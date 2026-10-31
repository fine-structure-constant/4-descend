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
