//! `PyO3` bindings for the consensus primitives in `quip-protocol`.
//!
//! Exposes `quip_proto._core.scoring`, `quip_proto._core.wire`, and
//! `quip_proto._core.ExitCode`. The `quip_proto` package's `__init__` re-exports
//! these so `from quip_proto import scoring, wire` stays a drop-in. Because the
//! math is the Rust source, the Python side cannot drift from it.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// Ising energy in milli-units for the given spins, fields, and edges.
///
/// `PyO3` extracts owned `Vec`s from Python; the body only borrows them.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts owned Vec arguments from Python objects"
)]
fn energy_milli(spins: Vec<i8>, h: Vec<f64>, j: Vec<f64>, edges: Vec<(usize, usize)>) -> i64 {
    quip_protocol::scoring::energy_milli(&spins, &h, &j, &edges)
}

/// Pairwise solution-set diversity in \[0, 1\].
///
/// `PyO3` extracts an owned `Vec` of spin vectors from Python.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts owned Vec arguments from Python objects"
)]
fn set_diversity(solutions: Vec<Vec<i8>>) -> f64 {
    quip_protocol::scoring::set_diversity(&solutions)
}

/// Encode `i32` values as little-endian bytes.
///
/// `PyO3` extracts an owned `Vec` from Python.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts owned Vec arguments from Python objects"
)]
fn encode_i32_le(values: Vec<i32>) -> Vec<u8> {
    quip_protocol::wire::encode_i32_le(&values)
}

/// Decode little-endian `i32` bytes.
///
/// # Errors
///
/// Returns a Python `ValueError` when the byte length is not a multiple of 4.
///
/// `PyO3` extracts an owned `Vec` from Python.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts owned Vec arguments from Python objects"
)]
fn decode_i32_le(b: Vec<u8>) -> PyResult<Vec<i32>> {
    quip_protocol::wire::decode_i32_le(&b).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Encode spins as packed bytes (`+1`/`-1` → bit representation).
///
/// `PyO3` extracts an owned `Vec` from Python.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts owned Vec arguments from Python objects"
)]
fn encode_spins(spins: Vec<i8>) -> Vec<u8> {
    quip_protocol::wire::encode_spins(&spins)
}

/// Decode packed spin bytes.
///
/// # Errors
///
/// Returns a Python `ValueError` when the payload is malformed.
///
/// `PyO3` extracts an owned `Vec` from Python.
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts owned Vec arguments from Python objects"
)]
fn decode_spins(b: Vec<u8>) -> PyResult<Vec<i8>> {
    quip_protocol::wire::decode_spins(&b).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Python module `quip_proto._core`: scoring, wire, and `ExitCode` constants.
#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let scoring = PyModule::new(py, "scoring")?;
    scoring.add_function(wrap_pyfunction!(energy_milli, &scoring)?)?;
    scoring.add_function(wrap_pyfunction!(set_diversity, &scoring)?)?;
    m.add_submodule(&scoring)?;

    let wire = PyModule::new(py, "wire")?;
    wire.add_function(wrap_pyfunction!(encode_i32_le, &wire)?)?;
    wire.add_function(wrap_pyfunction!(decode_i32_le, &wire)?)?;
    wire.add_function(wrap_pyfunction!(encode_spins, &wire)?)?;
    wire.add_function(wrap_pyfunction!(decode_spins, &wire)?)?;
    m.add_submodule(&wire)?;

    let exit_code = PyModule::new(py, "ExitCode")?;
    exit_code.add("CLEAN", 0u8)?;
    exit_code.add("CONFIG_INVALID", 64u8)?;
    exit_code.add("ENV_INCOMPATIBLE", 69u8)?;
    exit_code.add("INTERNAL_FATAL", 70u8)?;
    exit_code.add("TOKEN_REJECTED", 77u8)?;
    m.add_submodule(&exit_code)?;

    Ok(())
}
