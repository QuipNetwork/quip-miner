//! Session edge that serves v1 and v2 miners on one gRPC path during the
//! protocol v2 rollout. Remove it, and the `quip-proto-v1` dependency, once
//! the newest release of every miner repository speaks v2.

use bytes::{Buf, BufMut, Bytes};
use prost::Message;
use quip_proto::v1::{
    coord_msg, ising_problem, miner_msg, Algorithm, Backend, Capabilities, CoefficientEncoding,
    CoordMsg, Hello, JobKind, MinerMsg, Solution,
};
use quip_proto_v1::v1 as old;
use quip_protocol::session::{algorithm_from_name, backend_from_name, PROTOCOL_VERSION};
use quip_protocol::wire::{decode_spins, encode_spins_packed};
use tonic::Status;

/// Protocol a miner speaks on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerProtocol {
    /// quip-solver-core 0.0.1 and earlier, translated at this edge.
    V1,
    /// quip-solver-core 0.0.2-rc3 and later.
    V2,
}

/// Re-encode a message whose field numbers match in both versions.
fn recode<A: Message, B: Message + Default>(a: &A) -> Option<B> {
    B::decode(a.encode_to_vec().as_slice()).ok()
}

/// Classify a session from its first frame and return that frame as v2.
#[must_use = "the peer protocol selects the session codec"]
pub fn classify(first: &[u8]) -> (PeerProtocol, Result<MinerMsg, Status>) {
    if let Ok(msg) = MinerMsg::decode(first) {
        if matches!(&msg.msg, Some(miner_msg::Msg::Hello(h)) if h.capabilities.is_some()) {
            return (PeerProtocol::V2, Ok(msg));
        }
    }
    match old::MinerMsg::decode(first) {
        Ok(msg) if matches!(&msg.msg, Some(old::miner_msg::Msg::Hello(h)) if h.protocol_version == 1) => {
            (PeerProtocol::V1, Ok(v1_to_v2(msg)))
        }
        _ => (PeerProtocol::V2, decode_v2(first)),
    }
}

/// Decode one v2 frame.
///
/// # Errors
/// `InvalidArgument` when the frame is not a v2 `MinerMsg`.
#[expect(
    clippy::result_large_err,
    reason = "tonic streams require Status errors"
)]
pub fn decode_v2(frame: &[u8]) -> Result<MinerMsg, Status> {
    MinerMsg::decode(frame).map_err(|e| Status::invalid_argument(format!("decode MinerMsg: {e}")))
}

/// Decode one v1 frame and translate it.
///
/// # Errors
/// `InvalidArgument` when the frame is not a v1 `MinerMsg`.
#[expect(
    clippy::result_large_err,
    reason = "tonic streams require Status errors"
)]
pub fn decode_v1(frame: &[u8]) -> Result<MinerMsg, Status> {
    old::MinerMsg::decode(frame)
        .map(v1_to_v2)
        .map_err(|e| Status::invalid_argument(format!("decode v1 MinerMsg: {e}")))
}

/// v2 capabilities for a v1 identity. A v1 miner samples plain `I32` jobs
/// only, so it never lists `ISING_GENERATE` or a generator.
fn caps_from_v1(
    protocol_version: u32,
    backend: &str,
    algorithm: &str,
    kinds: &[i32],
    max_nodes: u32,
    max_edges: u32,
    stream_width: u32,
) -> Capabilities {
    Capabilities {
        supported_kinds: kinds
            .iter()
            .copied()
            .filter(|&k| k != JobKind::IsingGenerate as i32)
            .collect(),
        max_nodes,
        max_edges,
        protocol_version: if protocol_version == 1 {
            PROTOCOL_VERSION
        } else {
            0
        },
        stream_width,
        encodings: vec![CoefficientEncoding::I32 as i32],
        backend: backend_from_name(backend).unwrap_or(Backend::Unspecified) as i32,
        algorithm: algorithm_from_name(algorithm).unwrap_or(Algorithm::Unspecified) as i32,
        ..Default::default()
    }
}

/// Translate one v1 miner message to v2.
#[must_use]
pub fn v1_to_v2(msg: old::MinerMsg) -> MinerMsg {
    let msg = match msg.msg {
        Some(old::miner_msg::Msg::Hello(h)) => Some(miner_msg::Msg::Hello(Hello {
            capabilities: Some(caps_from_v1(
                h.protocol_version,
                &h.backend,
                &h.algorithm,
                &h.supported_kinds,
                h.max_nodes,
                h.max_edges,
                1,
            )),
            miner_id: h.miner_id,
            session_token: h.session_token,
        })),
        Some(old::miner_msg::Msg::Capabilities(c)) => {
            Some(miner_msg::Msg::Capabilities(caps_from_v1(
                c.protocol_version,
                &c.backend,
                &c.algorithm,
                &c.supported_kinds,
                c.max_nodes,
                c.max_edges,
                c.stream_width,
            )))
        }
        Some(old::miner_msg::Msg::Result(r)) => {
            Some(miner_msg::Msg::Result(quip_proto::v1::Result {
                job_id: r.job_id,
                solutions: r
                    .solutions
                    .into_iter()
                    .map(|s| Solution {
                        spins: decode_spins(&s.spins_bytes)
                            .map_or_else(|_| Vec::new(), |v| encode_spins_packed(&v)),
                        energy_milli: s.energy_milli,
                    })
                    .collect(),
                meta: r.meta.as_ref().and_then(recode),
                salt: Vec::new(),
                nonce: Vec::new(),
            }))
        }
        other => recode::<_, MinerMsg>(&old::MinerMsg { msg: other }).and_then(|m| m.msg),
    };
    MinerMsg { msg }
}

/// Translate one v2 coordinator message for a v1 miner, or `None` when v1
/// cannot express it. Only plain I32/1000 jobs have a v1 form.
/// The router must keep leases and other encodings away from v1 peers.
#[must_use]
pub fn v2_to_v1(msg: CoordMsg) -> Option<old::CoordMsg> {
    let msg = match msg.msg? {
        coord_msg::Msg::Welcome(_) => old::coord_msg::Msg::Welcome(old::Welcome {
            protocol_version: 1,
        }),
        coord_msg::Msg::Job(job) => old::coord_msg::Msg::Job(job_to_v1(job)?),
        other => return recode(&CoordMsg { msg: Some(other) }),
    };
    Some(old::CoordMsg { msg: Some(msg) })
}

fn job_to_v1(job: quip_proto::v1::Job) -> Option<old::Job> {
    if job.kind != JobKind::IsingSample as i32 {
        tracing::error!(
            kind = job.kind,
            "job kind has no v1 form; not sent to the v1 miner"
        );
        return None;
    }
    let ising = match job.ising {
        Some(p) => {
            if p.encoding != CoefficientEncoding::I32 as i32
                || p.scale != crate::producer::problem::MILLI_SCALE
            {
                tracing::error!(
                    encoding = p.encoding,
                    scale = p.scale,
                    "problem encoding has no v1 form; not sent"
                );
                return None;
            }
            Some(old::IsingProblem {
                graph: p.graph.map(|g| match g {
                    ising_problem::Graph::TopologyHash(h) => {
                        old::ising_problem::Graph::TopologyHash(h)
                    }
                    ising_problem::Graph::Edges(e) => {
                        old::ising_problem::Graph::Edges(old::EdgeList { u: e.u, v: e.v })
                    }
                }),
                h_milli_le32: p.h,
                j_milli_le32: p.j,
                num_reads: p.num_reads,
                num_sweeps: p.num_sweeps,
                anneal_time_us: p.anneal_time_us,
            })
        }
        None => None,
    };
    Some(old::Job {
        job_id: job.job_id,
        kind: job.kind,
        generation: job.generation,
        deadline_ms: job.deadline_ms,
        ising,
        provenance: job.provenance.as_ref().and_then(recode),
    })
}

use crate::chain::ChainClient;
use crate::session::CoordinatorService;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio_stream::Stream;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::codegen::{empty_body, http, Body, BoxFuture, StdError};

/// gRPC path both protocol versions dial.
const SESSION_PATH: &str = "/quip.v1.MinerService/Session";

/// A byte stream of encoded outbound frames.
pub(crate) type FrameStream = Pin<Box<dyn Stream<Item = Result<Bytes, Status>> + Send + 'static>>;

/// Passes gRPC frames through as bytes, so the edge can choose the schema.
#[derive(Debug, Clone, Copy, Default)]
struct RawCodec;
#[derive(Debug)]
struct RawEncoder;
#[derive(Debug)]
struct RawDecoder;

impl Encoder for RawEncoder {
    type Item = Bytes;
    type Error = Status;
    fn encode(&mut self, item: Bytes, dst: &mut EncodeBuf<'_>) -> Result<(), Status> {
        dst.put(item);
        Ok(())
    }
}

impl Decoder for RawDecoder {
    type Item = Bytes;
    type Error = Status;
    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Bytes>, Status> {
        Ok(Some(src.copy_to_bytes(src.remaining())))
    }
}

impl Codec for RawCodec {
    type Encode = Bytes;
    type Decode = Bytes;
    type Encoder = RawEncoder;
    type Decoder = RawDecoder;
    fn encoder(&mut self) -> RawEncoder {
        RawEncoder
    }
    fn decoder(&mut self) -> RawDecoder {
        RawDecoder
    }
}

/// The miner session service for v1 and v2 peers.
pub struct DualMinerServer<C: ChainClient + 'static> {
    svc: Arc<CoordinatorService<C>>,
}

impl<C: ChainClient + 'static> DualMinerServer<C> {
    /// Serve `svc` to miners of either protocol version.
    #[must_use]
    pub fn new(svc: CoordinatorService<C>) -> Self {
        Self { svc: Arc::new(svc) }
    }
}

impl<C: ChainClient + 'static> Clone for DualMinerServer<C> {
    fn clone(&self) -> Self {
        Self {
            svc: Arc::clone(&self.svc),
        }
    }
}

impl<C: ChainClient + 'static> tonic::server::NamedService for DualMinerServer<C> {
    const NAME: &'static str = quip_proto::v1::miner_service_server::SERVICE_NAME;
}

struct SessionFrames<C: ChainClient + 'static>(Arc<CoordinatorService<C>>);

impl<C: ChainClient + 'static> tonic::server::StreamingService<Bytes> for SessionFrames<C> {
    type Response = Bytes;
    type ResponseStream = FrameStream;
    type Future = BoxFuture<tonic::Response<FrameStream>, Status>;
    fn call(&mut self, request: tonic::Request<tonic::Streaming<Bytes>>) -> Self::Future {
        let svc = Arc::clone(&self.0);
        Box::pin(async move { svc.session_frames(request.into_inner()).await })
    }
}

impl<C, B> tonic::codegen::Service<http::Request<B>> for DualMinerServer<C>
where
    C: ChainClient + 'static,
    B: Body + Send + 'static,
    B::Error: Into<StdError> + Send + 'static,
{
    type Response = http::Response<tonic::body::BoxBody>;
    type Error = std::convert::Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: http::Request<B>) -> Self::Future {
        if req.uri().path() != SESSION_PATH {
            return Box::pin(async {
                let mut response = http::Response::new(empty_body());
                let headers = response.headers_mut();
                let _ = headers.insert(
                    Status::GRPC_STATUS,
                    (tonic::Code::Unimplemented as i32).into(),
                );
                let _ = headers.insert(
                    http::header::CONTENT_TYPE,
                    tonic::metadata::GRPC_CONTENT_TYPE,
                );
                Ok(response)
            });
        }
        let svc = Arc::clone(&self.svc);
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(RawCodec);
            Ok(grpc.streaming(SessionFrames(svc), req).await)
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "assert the supplied result fixture row"
)]
mod tests {
    use super::{classify, v1_to_v2, v2_to_v1, PeerProtocol};
    use prost::Message as _;
    use quip_proto::v1::{coord_msg, miner_msg, CoordMsg, Job, JobKind, Welcome};
    use quip_proto_v1::v1 as old;

    fn v1_hello() -> old::MinerMsg {
        old::MinerMsg {
            msg: Some(old::miner_msg::Msg::Hello(old::Hello {
                miner_id: "cpu-0".into(),
                session_token: "t".into(),
                protocol_version: 1,
                backend: "cpu".into(),
                algorithm: "sa".into(),
                supported_kinds: vec![1, 3],
                max_nodes: 10,
                max_edges: 20,
                native_topology_hash: None,
                features: vec![],
            })),
        }
    }

    #[test]
    fn classifies_a_v1_hello() {
        let (peer, msg) = classify(&v1_hello().encode_to_vec());
        assert_eq!(peer, PeerProtocol::V1);
        let Some(miner_msg::Msg::Hello(h)) = msg.unwrap().msg else {
            panic!("not a hello")
        };
        let caps = h.capabilities.unwrap();
        assert_eq!(
            caps.protocol_version,
            quip_protocol::session::PROTOCOL_VERSION
        );
        assert_eq!(caps.backend, quip_proto::v1::Backend::Cpu as i32);
        assert_eq!(caps.supported_kinds, vec![JobKind::IsingSample as i32]);
    }

    #[test]
    fn classifies_a_v2_hello() {
        let hello = quip_proto::v1::MinerMsg {
            msg: Some(miner_msg::Msg::Hello(quip_proto::v1::Hello {
                miner_id: "m".into(),
                session_token: "t".into(),
                capabilities: Some(quip_proto::v1::Capabilities {
                    protocol_version: 2,
                    ..Default::default()
                }),
            })),
        };
        assert_eq!(classify(&hello.encode_to_vec()).0, PeerProtocol::V2);
    }

    #[test]
    fn v1_caps_never_accept_leases() {
        let (_, msg) = classify(&v1_hello().encode_to_vec());
        let Some(miner_msg::Msg::Hello(h)) = msg.unwrap().msg else {
            panic!()
        };
        assert!(
            !crate::router::MinerCaps::from_capabilities(&h.capabilities.unwrap()).accepts_leases()
        );
    }

    #[test]
    fn v1_result_spins_are_packed() {
        let r = old::MinerMsg {
            msg: Some(old::miner_msg::Msg::Result(old::Result {
                job_id: vec![1],
                solutions: vec![old::Solution {
                    spins_bytes: vec![0x01, 0xFF, 0x01],
                    energy_milli: -7,
                }],
                meta: None,
            })),
        };
        let Some(miner_msg::Msg::Result(v2)) = v1_to_v2(r).msg else {
            panic!()
        };
        assert_eq!(
            v2.solutions[0].spins,
            quip_protocol::wire::encode_spins_packed(&[1, -1, 1])
        );
        assert_eq!(v2.solutions[0].energy_milli, -7);
    }

    #[test]
    fn welcome_goes_out_as_version_1() {
        let out = v2_to_v1(CoordMsg {
            msg: Some(coord_msg::Msg::Welcome(Welcome {
                protocol_version: 2,
            })),
        })
        .unwrap();
        assert!(
            matches!(out.msg, Some(old::coord_msg::Msg::Welcome(w)) if w.protocol_version == 1)
        );
    }

    #[test]
    fn plain_job_goes_out_with_v1_coefficients() {
        let job = Job {
            job_id: vec![9],
            kind: JobKind::IsingSample as i32,
            ising: Some(crate::producer::problem::milli_problem(
                None,
                &[1000, -1000],
                &[500],
            )),
            ..Default::default()
        };
        let out = v2_to_v1(CoordMsg {
            msg: Some(coord_msg::Msg::Job(job)),
        })
        .unwrap();
        let Some(old::coord_msg::Msg::Job(j)) = out.msg else {
            panic!()
        };
        let p = j.ising.unwrap();
        assert_eq!(
            p.h_milli_le32,
            quip_protocol::wire::encode_i32_le(&[1000, -1000])
        );
        assert_eq!(p.j_milli_le32, quip_protocol::wire::encode_i32_le(&[500]));
    }

    #[test]
    fn lease_job_is_not_translated_to_v1() {
        let snap = crate::chain::snapshot::MiningSnapshot {
            head_hash: [0; 32],
            last_proof_block_hash: [7; 32],
            topology_hash: vec![9; 32],
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![1000],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 1,
            max_energy_milli: 0,
            min_diversity_milli: 0,
            block_number: 1,
        };
        let lease = crate::lease::build_lease_job(&snap, [0; 32], 1, 4, 1);
        assert!(v2_to_v1(CoordMsg {
            msg: Some(coord_msg::Msg::Job(lease))
        })
        .is_none());
    }
}
