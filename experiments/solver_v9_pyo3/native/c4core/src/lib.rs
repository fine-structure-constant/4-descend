// Same search implementation as the standalone engine; no fork, pipes or files.
#![allow(dead_code)]
include!("main.rs");

#[cfg(feature = "python-extension")]
mod python_extension {
    use super::{Analysis, BOARD, Solver};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyclass(name = "Solver")]
    struct PySolver {
        solver: Solver,
    }

    #[pymethods]
    impl PySolver {
        #[new]
        fn new() -> Self {
            Self { solver: Solver::new() }
        }

        fn analyze(&mut self, py: Python<'_>, own: u64, mask: u64, budget_ms: u64)
            -> PyResult<(usize, i8, u64, u128, bool, Vec<i8>)> {
            if mask & !BOARD != 0 || own & !mask != 0 {
                return Err(PyValueError::new_err("Invalid bitboard"));
            }
            for col in 0..7 {
                let cells = (mask >> (7 * col)) & 63;
                if cells & (cells + 1) != 0 {
                    return Err(PyValueError::new_err("Non-contiguous gravity column"));
                }
            }
            let Analysis { col, score, nodes, elapsed_ms, solved, wdl } =
                py.allow_threads(|| self.solver.analyze(own, mask, budget_ms.clamp(1, 5000)));
            Ok((col, score, nodes, elapsed_ms, solved, wdl.to_vec()))
        }

        #[getter]
        fn table_entries(&self) -> usize {
            self.solver.tt.len()
        }

        fn resist(&mut self, py: Python<'_>, own: u64, mask: u64, budget_ms: u64)
            -> PyResult<(usize, u8, i16, u64, u128, bool)> {
            if mask & !BOARD != 0 || own & !mask != 0 {
                return Err(PyValueError::new_err("Invalid bitboard"));
            }
            for col in 0..7 {
                let cells = (mask >> (7 * col)) & 63;
                if cells & (cells + 1) != 0 {
                    return Err(PyValueError::new_err("Non-contiguous gravity column"));
                }
            }
            let result = py.allow_threads(||
                super::resistance::choose(own, mask, budget_ms.clamp(1, 5000)));
            Ok((result.col, result.depth, result.score, result.nodes,
                result.elapsed_ms, result.complete))
        }
    }

    #[pymodule]
    fn _c4solver(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<PySolver>()?;
        module.add("BACKEND", "rust-pyo3-abi3")?;
        module.add("ABI_MIN_PYTHON", "3.8")?;
        Ok(())
    }
}
