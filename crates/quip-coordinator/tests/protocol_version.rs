//! The coordinator accepts only `PROTOCOL_VERSION` in a miner `Hello`.

#![expect(
    clippy::expect_used,
    reason = "integration helper requires a working coordinator and socket"
)]

use quip_coordinator::chain::FakeChain;
use quip_coordinator::session::{CoordinatorService, CoordinatorState};
use quip_proto::v1::miner_service_client::MinerServiceClient;
use quip_proto::v1::miner_service_server::MinerServiceServer;
use quip_proto::v1::{coord_msg, miner_msg, Capabilities, Configure, Hello, JobKind, MinerMsg};
use quip_protocol::session::PROTOCOL_VERSION;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UnixListener;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::transport::{Endpoint, Server, Uri};

/// Serve one coordinator, send one `Hello`, and return the first reply.
/// `None` means the coordinator closed the stream without a reply.
async fn first_reply(tag: &str, protocol_version: u32) -> Option<coord_msg::Msg> {
    let sock = format!("/tmp/quip-proto-{tag}-{}.sock", std::process::id());
    let _ = std::fs::remove_file(&sock);
    let mut st = CoordinatorState::new();
    let _ = st.expected_tokens.insert("cpu-0".into(), "tok".into());
    let _ = st.configure.insert(
        "cpu-0".into(),
        Configure {
            queue_depth: 1,
            idle_timeout_s: 5,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        },
    );
    let chain = Arc::new(FakeChain::new(
        quip_coordinator::drive::parse_topology_spec(
            quip_coordinator::presets::preset_spec("smoke").expect("preset"),
        )
        .expect("spec")
        .to_snapshot(),
        None,
    ));
    let svc = CoordinatorService {
        state: Arc::new(Mutex::new(st)),
        chain,
        submit_notify: Arc::new(Mutex::new(None)),
    };
    let incoming = UnixListenerStream::new(UnixListener::bind(&sock).expect("bind"));
    let server = tokio::spawn(
        Server::builder()
            .add_service(MinerServiceServer::new(svc))
            .serve_with_incoming(incoming),
    );

    let path = sock.clone();
    let channel = Endpoint::try_from("http://[::]:50051")
        .expect("endpoint")
        .connect_with_connector(tower::service_fn(move |_: Uri| {
            let p = path.clone();
            async move {
                Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(
                    tokio::net::UnixStream::connect(p).await?,
                ))
            }
        }))
        .await
        .expect("connect");
    let mut client = MinerServiceClient::new(channel);
    let (tx, rx) = mpsc::channel::<MinerMsg>(4);
    tx.send(MinerMsg {
        msg: Some(miner_msg::Msg::Hello(Hello {
            miner_id: "cpu-0".into(),
            session_token: "tok".into(),
            capabilities: Some(Capabilities {
                protocol_version,
                supported_kinds: vec![JobKind::IsingSample as i32],
                stream_width: 1,
                ..Default::default()
            }),
        })),
    })
    .await
    .expect("hello");
    let mut inbound = client
        .session(ReceiverStream::new(rx))
        .await
        .expect("session")
        .into_inner();
    let reply = tokio::time::timeout(Duration::from_secs(5), inbound.message())
        .await
        .expect("the coordinator answers or closes within 5 s")
        .expect("stream status")
        .and_then(|m| m.msg);
    server.abort();
    let _ = std::fs::remove_file(&sock);
    reply
}

#[tokio::test]
async fn the_current_protocol_version_is_welcomed() {
    let reply = first_reply("current", PROTOCOL_VERSION).await;
    assert!(
        matches!(reply, Some(coord_msg::Msg::Welcome(_))),
        "got {reply:?}"
    );
}

#[tokio::test]
async fn protocol_version_1_is_rejected() {
    assert_eq!(first_reply("v1", 1).await, None);
}

#[tokio::test]
async fn an_unknown_protocol_version_is_rejected() {
    assert_eq!(first_reply("future", PROTOCOL_VERSION + 1).await, None);
}
