# Sources and licenses

The search engine is derived from Pascal Pons, Connect 4 solver (AGPL-3.0):
https://github.com/PascalPons/connect4
Corresponding search and Python binding source is in native/c4core, pinned
build dependencies in Cargo.lock, and build instructions in README.md.

Python extension binding uses PyO3 0.24.2 (MIT OR Apache-2.0):
https://github.com/PyO3/pyo3/tree/v0.24.2
https://pyo3.rs/v0.24.2/building-and-distribution.html
Third-party license notices are collected in THIRD_PARTY_LICENSES.txt.

Fixed books are inherited unchanged from v9: complete <=12-disc optimal-move
book, public 7x6.book exact scores, and 41,863 midgame WDL positions. The 202
v9 additional positions are development data labelled offline using the C++
solver, not learned weights and not an independent evaluation set. Full data
provenance remains in experiments/solver_v9/REFERENCES.md and
experiments/exact12_v7/REFERENCES.md in the full project.
