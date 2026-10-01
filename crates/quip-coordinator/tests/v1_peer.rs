//! A v1 miner (quip-solver-core 0.0.1 wire) handshakes with the dual-stack
//! edge and receives a plain job in v1 form. Result translation has unit tests.

#![expect(
    clippy::too_many_lines,
    reason = "one end-to-end handshake and job exchange"
)]

use quip_coordinator::chain::FakeChain;
use quip_coordinator::edge::DualMinerServer;
use quip_coordinator::session::{CoordinatorService, CoordinatorState};
use quip_proto::v1::{Configure, JobKind};
use quip_proto_v1::v1 as old;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UnixListener;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::transport::{Endpoint, Server, Uri};

fn om(msg: old::miner_msg::Msg) -> old::MinerMsg {
    old::MinerMsg { msg: Some(msg) }
}

#[tokio::test]
async fn v1_miner_mines_a_plain_job_through_the_edge() {
    let sock = format!("/tmp/quip-v1-peer-{}.sock", std::process::id());
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
    st.target = Some(quip_proto::v1::SetTarget {
        max_energy_milli: i64::MAX / 2,
        min_solutions: 1,
        max_proof_solutions: 32,
        ..Default::default()
    });
    let state = Arc::new(Mutex::new(st));
    let chain = Arc::new(FakeChain::new(
        quip_coordinator::drive::parse_topology_spec(
            quip_coordinator::presets::preset_spec("smoke").expect("preset"),
        )
        .expect("spec")
        .to_snapshot(),
        None,
    ));
    let svc = CoordinatorService {
        state: Arc::clone(&state),
        chain,
        submit_notify: Arc::new(Mutex::new(None)),
    };
    let incoming = UnixListenerStream::new(UnixListener::bind(&sock).expect("bind"));
    let server = tokio::spawn(
        Server::builder()
            .add_service(DualMinerServer::new(svc))
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
    let mut client = old::miner_service_client::MinerServiceClient::new(channel);
    let (tx, rx) = mpsc::channel::<old::MinerMsg>(8);
    tx.send(om(old::miner_msg::Msg::Hello(old::Hello {
        miner_id: "cpu-0".into(),
        session_token: "tok".into(),
        protocol_version: 1,
        backend: "cpu".into(),
        algorithm: "sa".into(),
        supported_kinds: vec![JobKind::IsingSample as i32],
        max_nodes: 0,
        max_edges: 0,
        native_topology_hash: None,
        features: vec![],
    })))
    .await
    .expect("hello");
    let mut inbound = client
        .session(ReceiverStream::new(rx))
        .await
        .expect("session")
        .into_inner();

    let welcome = inbound.message().await.expect("recv").expect("welcome");
    assert!(
        matches!(welcome.msg, Some(old::coord_msg::Msg::Welcome(w)) if w.protocol_version == 1)
    );
    assert!(!state
        .lock()
        .await
        .router
        .caps("cpu-0")
        .expect("registered")
        .accepts_leases());

    // Stage one plain inline job, then report ready so credits seed.
    let job = quip_proto::v1::Job {
        job_id: vec![7; 32],
        kind: JobKind::IsingSample as i32,
        generation: 0,
        ising: Some(quip_coordinator::producer::problem::milli_problem(
            Some(quip_proto::v1::ising_problem::Graph::Edges(
                quip_proto::v1::EdgeList {
                    u: vec![0],
                    v: vec![1],
                },
            )),
            &[1000, -1000],
            &[500],
        )),
        provenance: Some(quip_proto::v1::Provenance {
            is_pow: false,
            order_id: vec![1; 8],
        }),
        ..Default::default()
    };
    {
        let mut st = state.lock().await;
        assert!(st.router.stage_on("cpu-0", job));
    }
    tx.send(om(old::miner_msg::Msg::Ready(old::Ready {})))
        .await
        .expect("ready");

    let got = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let m = inbound.message().await.expect("recv").expect("msg");
            if let Some(old::coord_msg::Msg::Job(j)) = m.msg {
                return j;
            }
        }
    })
    .await
    .expect("job within 5s");
    let p = got.ising.expect("problem");
    assert_eq!(
        p.h_milli_le32,
        quip_protocol::wire::encode_i32_le(&[1000, -1000])
    );

    server.abort();
    let _ = std::fs::remove_file(&sock);
}
