//! Published capability additions preserve protocol-2 wire semantics.

use prost::Message;
use quip_proto::v1::Capabilities;
use quip_protocol::session::PROTOCOL_VERSION;

#[test]
fn degree_presence_roundtrips_without_changing_protocol_version() {
    assert_eq!(PROTOCOL_VERSION, 2);
    for degree in [None, Some(0), Some(20)] {
        let caps = Capabilities {
            protocol_version: PROTOCOL_VERSION,
            max_degree: degree,
            ..Capabilities::default()
        };
        assert_eq!(caps.max_local_field_milli, None);
        assert_eq!(caps.exact_resolution_milli, None);
        let bytes = caps.encode_to_vec();
        assert_eq!(Capabilities::decode(bytes.as_slice()), Ok(caps));
    }
}
