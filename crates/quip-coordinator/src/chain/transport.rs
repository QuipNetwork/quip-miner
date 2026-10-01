//! The transport seam between the chain client and a JSON-RPC endpoint.
//!
//! The client builds requests and decodes responses. It never opens a socket
//! itself. That split lets the same client run on a native WebSocket, on an
//! HTTP endpoint, or on a browser WebSocket under WebAssembly, where the
//! native jsonrpsee client does not build.

use super::ChainError;
use async_trait::async_trait;

/// A boxed stream, so the trait stays object safe.
pub type BoxStream<'a, T> = std::pin::Pin<Box<dyn futures::Stream<Item = T> + Send + 'a>>;

/// Issues JSON-RPC requests and subscriptions against one endpoint.
#[async_trait]
pub trait RpcTransport: Send + Sync {
    /// Issue one JSON-RPC method call against `url`.
    ///
    /// # Errors
    /// Returns [`ChainError::Unavailable`] when the endpoint cannot be reached
    /// or the node reports an error.
    async fn request(
        &self,
        url: &str,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, ChainError>;

    /// Open a JSON-RPC subscription against `url`.
    ///
    /// The stream yields one decoded notification per item. It ends when the
    /// server closes the subscription.
    ///
    /// # Errors
    /// Returns [`ChainError::Unavailable`] when the subscription cannot start.
    async fn subscribe(
        &self,
        url: &str,
        sub: &str,
        params: serde_json::Value,
        unsub: &str,
    ) -> Result<BoxStream<'static, Result<serde_json::Value, ChainError>>, ChainError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Transport that returns a scripted response and records the call.
    struct ScriptedTransport {
        calls: Mutex<Vec<(String, String)>>,
    }

    #[async_trait::async_trait]
    impl RpcTransport for ScriptedTransport {
        async fn request(
            &self,
            url: &str,
            method: &str,
            _params: serde_json::Value,
        ) -> Result<serde_json::Value, ChainError> {
            self.calls
                .lock()
                .unwrap()
                .push((url.to_string(), method.to_string()));
            Ok(serde_json::Value::String("0x00".into()))
        }

        async fn subscribe(
            &self,
            _url: &str,
            _sub: &str,
            _params: serde_json::Value,
            _unsub: &str,
        ) -> Result<BoxStream<'static, Result<serde_json::Value, ChainError>>, ChainError> {
            Err(ChainError::Unavailable(
                "no subscriptions in this test".into(),
            ))
        }
    }

    #[tokio::test]
    async fn a_transport_records_the_url_and_method_it_was_given() {
        let t = ScriptedTransport {
            calls: Mutex::new(Vec::new()),
        };
        let out = t
            .request(
                "ws://example:9944",
                "state_getStorage",
                serde_json::Value::Null,
            )
            .await
            .unwrap();
        assert_eq!(out.as_str(), Some("0x00"));
        assert_eq!(
            t.calls.lock().unwrap().as_slice(),
            &[(
                "ws://example:9944".to_string(),
                "state_getStorage".to_string()
            )]
        );
    }
}
