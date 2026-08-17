//! Native jsonrpsee implementation of [`super::transport::RpcTransport`].
//!
//! Each request opens a fresh client. A dropped socket costs one call rather
//! than wedging a long-lived connection.

use super::transport::{BoxStream, RpcTransport};
use super::ChainError;
use async_trait::async_trait;
use serde_json::Value;
use std::time::Duration;

/// Budget for opening a session to one validator (TCP connect + WS handshake).
///
/// A wedged peer that accepts TCP and never speaks used to block the ordered
/// failover list forever. Five seconds is long enough for a slow host path and
/// short enough that the next endpoint is tried before the coordinator looks
/// hung at startup.
pub(crate) const RPC_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Budget for one JSON-RPC method after the client is up.
///
/// The jsonrpsee default is 60s, which multiplies badly across a validator list
/// when a peer is silent. Fifteen seconds still covers a loaded but working
/// node; it does not wait a full minute before moving on.
pub(crate) const RPC_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

pub(crate) async fn rpc_request(
    url: &str,
    method: &str,
    params: Value,
) -> Result<Value, ChainError> {
    // Support both ws:// and http(s):// via jsonrpsee.
    if url.starts_with("ws://") || url.starts_with("wss://") {
        rpc_ws(url, method, params).await
    } else {
        rpc_http(url, method, params).await
    }
}

async fn rpc_http(url: &str, method: &str, params: Value) -> Result<Value, ChainError> {
    use jsonrpsee::core::client::ClientT;
    use jsonrpsee::http_client::HttpClientBuilder;

    // HttpClientBuilder has no separate connection_timeout. request_timeout
    // wraps the full transport send (connect + response), so one budget covers
    // both phases and surfaces as Unavailable for failover.
    let client = HttpClientBuilder::default()
        .request_timeout(RPC_REQUEST_TIMEOUT)
        .build(url)
        .map_err(|e| ChainError::Unavailable(format!("http client: {e}")))?;
    let result: Value = client
        .request(method, rpc_params_from_value(params))
        .await
        .map_err(|e| ChainError::Unavailable(format!("rpc {method}: {e}")))?;
    Ok(result)
}

async fn rpc_ws(url: &str, method: &str, params: Value) -> Result<Value, ChainError> {
    use jsonrpsee::core::client::ClientT;
    use jsonrpsee::ws_client::WsClientBuilder;

    // connection_timeout only bounds TCP connect inside jsonrpsee. A peer that
    // accepts and never completes the WebSocket handshake still hangs build(),
    // so the whole build is wrapped in the same connect budget.
    let build = WsClientBuilder::default()
        .connection_timeout(RPC_CONNECT_TIMEOUT)
        .request_timeout(RPC_REQUEST_TIMEOUT)
        .build(url);
    let client = match tokio::time::timeout(RPC_CONNECT_TIMEOUT, build).await {
        Ok(Ok(client)) => client,
        Ok(Err(e)) => {
            return Err(ChainError::Unavailable(format!("ws client: {e}")));
        }
        Err(_) => {
            return Err(ChainError::Unavailable(format!(
                "ws connect timed out after {}s",
                RPC_CONNECT_TIMEOUT.as_secs()
            )));
        }
    };
    let result: Value = client
        .request(method, rpc_params_from_value(params))
        .await
        .map_err(|e| ChainError::Unavailable(format!("rpc {method}: {e}")))?;
    Ok(result)
}

fn rpc_params_from_value(params: Value) -> jsonrpsee::core::params::ArrayParams {
    match params {
        Value::Array(arr) => {
            let mut p = jsonrpsee::core::params::ArrayParams::new();
            for v in arr {
                let _ = p.insert(v);
            }
            p
        }
        other => {
            let mut p = jsonrpsee::core::params::ArrayParams::new();
            let _ = p.insert(other);
            p
        }
    }
}

/// Native transport over jsonrpsee. Opens a fresh connection per request, so a
/// dropped socket costs one call rather than wedging the client.
pub struct JsonrpseeTransport;

#[async_trait]
impl RpcTransport for JsonrpseeTransport {
    async fn request(&self, url: &str, method: &str, params: Value) -> Result<Value, ChainError> {
        rpc_request(url, method, params).await
    }

    async fn subscribe(
        &self,
        url: &str,
        sub: &str,
        params: Value,
        unsub: &str,
    ) -> Result<BoxStream<'static, Result<Value, ChainError>>, ChainError> {
        use jsonrpsee::core::client::{Subscription, SubscriptionClientT};
        use jsonrpsee::ws_client::WsClientBuilder;

        let build = WsClientBuilder::default()
            .connection_timeout(RPC_CONNECT_TIMEOUT)
            .request_timeout(RPC_REQUEST_TIMEOUT)
            .build(url);
        let client = match tokio::time::timeout(RPC_CONNECT_TIMEOUT, build).await {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => return Err(ChainError::Unavailable(format!("ws client: {e}"))),
            Err(_) => {
                return Err(ChainError::Unavailable(format!(
                    "ws connect timed out after {}s",
                    RPC_CONNECT_TIMEOUT.as_secs()
                )))
            }
        };
        let sub_stream: Subscription<Value> = match client
            .subscribe(sub, rpc_params_from_value(params), unsub)
            .await
        {
            Ok(s) => s,
            // A JSON-RPC error object is the node's answer about this request.
            // For `author_submitAndWatchExtrinsic` it means the node rejected
            // the extrinsic, and resubmitting the same bytes cannot change
            // that. Report it as a submit failure so the caller does not read a
            // permanent rejection as an unreachable node and retry forever.
            // Every other error is a transport fault and stays transient.
            Err(jsonrpsee::core::client::Error::Call(obj)) => {
                return Err(ChainError::Submit(format!("subscribe {sub}: {obj}")))
            }
            Err(e) => return Err(ChainError::Unavailable(format!("subscribe {sub}: {e}"))),
        };

        // The client must outlive the stream: dropping it closes the socket and
        // ends the subscription. Move it into the stream's state.
        let stream = futures::stream::unfold((sub_stream, client), |(mut s, c)| async move {
            let next = s.next().await?;
            let item = next.map_err(|e| ChainError::Unavailable(format!("subscription: {e}")));
            Some((item, (s, c)))
        });
        Ok(Box::pin(stream))
    }
}
