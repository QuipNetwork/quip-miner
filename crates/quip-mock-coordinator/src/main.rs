//! CLI entry for the mock coordinator: run one miner binary through the
//! full conformance session and print the driver report.

use quip_mock_coordinator::driver::drive_miner;
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(bin_path) = args.next() else {
        #[expect(
            clippy::print_stderr,
            reason = "CLI usage error is intentionally written to stderr"
        )]
        {
            eprintln!("usage: quip-mock-coordinator <miner-bin> <unix://socket>");
        }
        return ExitCode::from(64);
    };
    let Some(socket) = args.next() else {
        #[expect(
            clippy::print_stderr,
            reason = "CLI usage error is intentionally written to stderr"
        )]
        {
            eprintln!("usage: quip-mock-coordinator <miner-bin> <unix://socket>");
        }
        return ExitCode::from(64);
    };
    let report = drive_miner(&bin_path, &socket).await;
    #[expect(
        clippy::print_stdout,
        reason = "user-facing CLI prints the conformance report"
    )]
    {
        println!("{report:?}");
    }
    if report.is_conformant() {
        ExitCode::SUCCESS
    } else {
        #[expect(
            clippy::print_stderr,
            reason = "CLI failure summary is intentionally written to stderr"
        )]
        {
            eprintln!("conformance failed: {report:?}");
        }
        ExitCode::from(1)
    }
}
