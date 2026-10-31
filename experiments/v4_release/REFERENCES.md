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

- Markus Thill, `bitbully-databases` 0.0.2, MIT license:
  https://github.com/MarkusThill/bitbully-databases
  https://pypi.org/project/bitbully-databases/0.0.2/
  The full 12-ply WDL database is stored as an untrained uint8 model buffer.
  It is search data, separate from the CNN's learned weights. Its signed
  Huffman key format and omitted-first-player-win convention are read by
  `support/deep_opening.py`. A short exact minimax backs up the frontier.
  The MIT notice is included in that module and with the local dataset.
  A 363-position independent check agreed with the UCI/Tromp 8-ply outcomes.


本轮还实现了已知胜负值下的候选走法排除，以及安全下界向后续回合的传播。
这些推断仅使用精确值或已证明的上下界。原始 C++ 教师只用于离线标注，
其所有可确定结果通过独立数组棋盘残局检验后才用于训练。

## v4 release: complete twelve-ply action memory and model search

- `support/movebook.py` contains the lookup implementation. The two sorted
  buffers in model.pt cover all 3,356,580 symmetric, nonterminal pre-move states
  in plies 1..12 (0..11 discs). Scores are WDL: +1/0/-1 for the current side.
  Every stored action is optimal for this objective. Fastest mate distance is
  not a secondary objective. The complete 12-disc BitBully frontier was backed
  up exactly; independent Pons checks and exhaustive verification are recorded
  under runs/. Missing or incomplete action memory is rejected by Bot startup.
- The full public Pascal Pons `7x6.book` supplies exact leaf scores to the
  bounded search after the opening. Source/release:
  https://github.com/PascalPons/connect4/releases/tag/book
  License: GNU AGPL v3 or later; text in tools/AGPL-3.0.txt. The corresponding
  solver sources (retaining notices) are provided under tools/pons_fast/.
- `tools/endgame_native.cpp` is the locally implemented, CPU-bounded exact
  search accelerator. It exposes a CPython stable ABI (3.8+) module. Its code
  and build inputs are included; the runtime needs no compiler or network.
- The learned CNN weights remain local training output (88,134 parameters).
  `movebook_keys`, `movebook_data`, `opening12`, and `deepbook` are untrained,
  explicitly labelled public/derived search-data buffers, not learned weights.
  The first twelve plies use the proved optimal action set; later decisions
  call the CNN and bounded search. Tests check that boundary explicitly.
