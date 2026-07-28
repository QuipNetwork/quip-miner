//! `download` subcommand: pull winning qblocks from chain, redraw + self-verify
//! their Ising problems, and emit a `hardest_models` nonce-ref dataset.

pub mod record;

pub use record::{
    build_instance_record, hex_plain, DifficultyJson, InstanceRecord, ProvenanceJson, VerifyError,
};
