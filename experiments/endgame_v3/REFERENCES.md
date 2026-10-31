# Search references

- Pascal Pons, *Solving Connect Four*, especially bitboards, transposition tables,
  and anticipating direct losing moves:
  https://blog.gamesolver.org/solving-connect-four/06-bitboard/
  https://blog.gamesolver.org/solving-connect-four/07-transposition-table/
  https://blog.gamesolver.org/solving-connect-four/09-anticipate-losing-moves/

`support/tactics.py` implements these published ideas locally in Python.
The alpha-beta teacher is depth bounded; it is not a perfect-game oracle.
The policy/value network is initialized randomly and trained locally.
Official `strong` responses are recorded by `collect_strong.py` as supplementary
teacher data. No network call is made by exported inference code.


- Pascal Pons, iterative deepening and null-window exact search:
  https://blog.gamesolver.org/solving-connect-four/08-iterative-deepening/
  `support/endgame.py` implements two-sided exact transposition bounds and
  fair root probes. Timeouts retain unknown branches. Fully solved endgames
  account for move parity and forced moves through terminal play.
- Victor Allis (1988), *A Knowledge-based Approach of Connect-Four*:
  https://tromp.github.io/c4/connect4_thesis.pdf
  Sections 3 and 4 explain odd/even threats and control of zugzwang. These are
  useful for diagnosing forced losses; no unproved strategic rule is used to
  discard moves in this implementation.
- Tromp, J. (1995), *Connect-4* [Dataset]. UCI Machine Learning Repository.
  DOI: https://doi.org/10.24432/C59P43
  https://archive.ics.uci.edu/dataset/26/connect+4
  License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
  The 67,557 labelled 8-ply boards supervise network values. Exact opening
  lookups additionally use canonical bitboard encoding, reflection, and sound
  retrograde analysis; 116,010 proved positions are retained. Missing forced
  cases remain unknown. `build_opening_data.py` reproduces this transformation.
  The data is an additional search aid; the network weights are trained locally.
