// Bitboard solver ported from Pascal Pons' Connect 4 solver (AGPL-3.0).
// https://github.com/PascalPons/connect4

use std::env;
use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

const WIDTH: usize = 7;
const HEIGHT: usize = 6;
const STRIDE: usize = HEIGHT + 1;
const CELLS: u32 = (WIDTH * HEIGHT) as u32;
const COL_ORDER: [usize; WIDTH] = [3, 4, 2, 5, 1, 6, 0];

const fn make_bottom() -> u64 {
    let mut mask = 0u64;
    let mut col = 0;
    while col < WIDTH {
        mask |= 1u64 << (col * STRIDE);
        col += 1;
    }
    mask
}

const BOTTOM: u64 = make_bottom();
const BOARD: u64 = BOTTOM * ((1u64 << HEIGHT) - 1);

const fn make_col_masks() -> [u64; WIDTH] {
    let mut masks = [0u64; WIDTH];
    let mut col = 0;
    while col < WIDTH {
        masks[col] = ((1u64 << HEIGHT) - 1) << (col * STRIDE);
        col += 1;
    }
    masks
}

const COL_MASKS: [u64; WIDTH] = make_col_masks();

const TT_BITS: usize = 22;
const TT_SIZE: usize = 1 << TT_BITS;

const FLAG_EMPTY: u8 = 0;
const FLAG_EXACT: u8 = 1;
const FLAG_LOWER: u8 = 2;
const FLAG_UPPER: u8 = 3;

#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    value: i8,
    flag: u8,
    best: u8,
}

const EMPTY_ENTRY: Entry = Entry {
    key: 0,
    value: 0,
    flag: FLAG_EMPTY,
    best: 255,
};

#[inline(always)]
fn possible(mask: u64) -> u64 {
    (mask + BOTTOM) & BOARD
}

#[inline(always)]
fn winning_position(position: u64, mask: u64) -> u64 {
    let mut r = (position << 1) & (position << 2) & (position << 3);

    let mut p = (position << STRIDE) & (position << (2 * STRIDE));
    r |= p & (position << (3 * STRIDE));
    r |= p & (position >> STRIDE);
    p = (position >> STRIDE) & (position >> (2 * STRIDE));
    r |= p & (position << STRIDE);
    r |= p & (position >> (3 * STRIDE));

    p = (position << HEIGHT) & (position << (2 * HEIGHT));
    r |= p & (position << (3 * HEIGHT));
    r |= p & (position >> HEIGHT);
    p = (position >> HEIGHT) & (position >> (2 * HEIGHT));
    r |= p & (position << HEIGHT);
    r |= p & (position >> (3 * HEIGHT));

    p = (position << (HEIGHT + 2)) & (position << (2 * (HEIGHT + 2)));
    r |= p & (position << (3 * (HEIGHT + 2)));
    r |= p & (position >> (HEIGHT + 2));
    p = (position >> (HEIGHT + 2)) & (position >> (2 * (HEIGHT + 2)));
    r |= p & (position << (HEIGHT + 2));
    r |= p & (position >> (3 * (HEIGHT + 2)));

    r & (BOARD ^ mask)
}

#[inline(always)]
fn possible_non_losing(own: u64, mask: u64) -> u64 {
    let mut moves = possible(mask);
    let opponent_win = winning_position(own ^ mask, mask);
    let forced = moves & opponent_win;
    if forced != 0 {
        if forced & (forced - 1) != 0 {
            return 0;
        }
        moves = forced;
    }
    moves & !(opponent_win >> 1)
}

#[inline(always)]
fn move_score(own: u64, mask: u64, bit: u64) -> u32 {
    winning_position(own | bit, mask).count_ones()
}

#[inline(always)]
fn first_column(bits: u64) -> usize {
    for &col in COL_ORDER.iter() {
        if bits & COL_MASKS[col] != 0 {
            return col;
        }
    }
    3
}

fn heuristic_choice(own: u64, mask: u64) -> usize {
    let legal = possible(mask);
    let safe = possible_non_losing(own, mask);
    let candidates = if safe != 0 { safe } else { legal };
    let mut best_col = 3usize;
    let mut best = i32::MIN;
    for &col in COL_ORDER.iter() {
        let bit = candidates & COL_MASKS[col];
        if bit == 0 {
            continue;
        }
        let threats = winning_position(own | bit, mask | bit).count_ones() as i32;
        let center = 3 - (col as i32 - 3).abs();
        let score = threats * 16 + center;
        if score > best {
            best = score;
            best_col = col;
        }
    }
    best_col
}

struct Solver {
    tt: Vec<Entry>,
    nodes: u64,
    deadline: Option<Instant>,
    timed_out: bool,
}

struct Analysis {
    col: usize,
    score: i8,
    wdl: [i8; WIDTH],
    nodes: u64,
    elapsed_ms: u128,
    solved: bool,
}

const UNKNOWN: i8 = 9;

impl Solver {
    fn new() -> Self {
        Solver {
            tt: vec![EMPTY_ENTRY; TT_SIZE],
            nodes: 0,
            deadline: None,
            timed_out: false,
        }
    }

    #[inline(always)]
    fn index(key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - TT_BITS)) as usize
    }

    #[inline(always)]
    fn tick(&mut self) {
        self.nodes += 1;
        if self.nodes & 1023 == 0 {
            if let Some(deadline) = self.deadline {
                if Instant::now() >= deadline {
                    self.timed_out = true;
                }
            }
        }
    }

    fn store(&mut self, key: u64, value: i8, flag: u8, best: u8) {
        let entry = &mut self.tt[Self::index(key)];
        entry.key = key;
        entry.value = value;
        entry.flag = flag;
        entry.best = best;
    }

    fn negamax(&mut self, own: u64, mask: u64, moves: u32, mut alpha: i8, mut beta: i8) -> i8 {
        self.tick();
        if self.timed_out {
            return 0;
        }

        let moves_mask = possible_non_losing(own, mask);
        if moves_mask == 0 {
            return -(((CELLS - moves) as i8) / 2);
        }
        if moves >= CELLS - 2 {
            return 0;
        }

        let min = -(((CELLS - 2 - moves) as i8) / 2);
        if alpha < min {
            alpha = min;
            if alpha >= beta {
                return alpha;
            }
        }
        let max = ((CELLS - 1 - moves) as i8) / 2;
        if beta > max {
            beta = max;
            if alpha >= beta {
                return beta;
            }
        }

        let key = own + mask;
        let idx = Self::index(key);
        let mut tt_best = 255usize;
        {
            let entry = self.tt[idx];
            if entry.flag != FLAG_EMPTY && entry.key == key {
                match entry.flag {
                    FLAG_EXACT => return entry.value,
                    FLAG_LOWER => {
                        if entry.value > alpha {
                            alpha = entry.value;
                            if alpha >= beta {
                                return alpha;
                            }
                        }
                    }
                    FLAG_UPPER => {
                        if entry.value < beta {
                            beta = entry.value;
                            if alpha >= beta {
                                return beta;
                            }
                        }
                    }
                    _ => {}
                }
                tt_best = entry.best as usize;
            }
        }

        let mut candidates: [(i8, u64, u8); WIDTH] = [(0, 0, 0); WIDTH];
        let mut count = 0;
        for &col in COL_ORDER.iter() {
            let bit = moves_mask & COL_MASKS[col];
            if bit == 0 {
                continue;
            }
            let mut score = move_score(own, mask, bit) as i8;
            if col == tt_best {
                score += 32;
            }
            let mut pos = count;
            while pos > 0 && candidates[pos - 1].0 < score {
                candidates[pos] = candidates[pos - 1];
                pos -= 1;
            }
            candidates[pos] = (score, bit, col as u8);
            count += 1;
        }

        let mut best_col = candidates[0].2;
        for i in 0..count {
            let bit = candidates[i].1;
            let col = candidates[i].2;
            let child_own = own ^ mask;
            let child_mask = mask | bit;
            let score = -self.negamax(child_own, child_mask, moves + 1, -beta, -alpha);
            if self.timed_out {
                return 0;
            }
            if score >= beta {
                self.store(key, score, FLAG_LOWER, col);
                return score;
            }
            if score > alpha {
                alpha = score;
                best_col = col;
            }
        }

        self.store(key, alpha, FLAG_UPPER, best_col);
        alpha
    }

    fn solve(&mut self, own: u64, mask: u64, moves: u32, weak: bool) -> i8 {
        if winning_position(own, mask) & possible(mask) != 0 {
            return ((CELLS + 1 - moves) as i8) / 2;
        }
        let mut min = -(((CELLS - moves) as i8) / 2);
        let mut max = ((CELLS + 1 - moves) as i8) / 2;
        if weak {
            min = -1;
            max = 1;
        }
        while min < max {
            let mut median = min + (max - min) / 2;
            if median <= 0 && min / 2 < median {
                median = min / 2;
            } else if median >= 0 && max / 2 > median {
                median = max / 2;
            }
            let result = self.negamax(own, mask, moves, median, median + 1);
            if self.timed_out {
                return 0;
            }
            if result <= median {
                max = result;
            } else {
                min = result;
            }
        }
        min
    }

    fn analyze(&mut self, own: u64, mask: u64, budget_ms: u64) -> Analysis {
        let start = Instant::now();
        self.nodes = 0;
        self.timed_out = false;
        self.deadline = Some(start + Duration::from_millis(budget_ms.max(1)));

        let moves = mask.count_ones();
        let legal = possible(mask);
        let mut wdl = [UNKNOWN; WIDTH];
        if legal == 0 {
            return Analysis {
                col: 3,
                score: 0,
                wdl,
                nodes: 0,
                elapsed_ms: start.elapsed().as_millis(),
                solved: true,
            };
        }

        let immediate = winning_position(own, mask) & legal;
        if immediate != 0 {
            let col = first_column(immediate);
            wdl[col] = 1;
            return Analysis {
                col,
                score: 1,
                wdl,
                nodes: self.nodes,
                elapsed_ms: start.elapsed().as_millis(),
                solved: true,
            };
        }

        let safe = possible_non_losing(own, mask);
        if safe == 0 {
            // Every legal drop loses immediately (the opponent wins next turn).
            for col in 0..WIDTH {
                if legal & COL_MASKS[col] != 0 {
                    wdl[col] = -1;
                }
            }
            return Analysis {
                col: heuristic_choice(own, mask),
                score: -1,
                wdl,
                nodes: self.nodes,
                elapsed_ms: start.elapsed().as_millis(),
                solved: true,
            };
        }

        for col in 0..WIDTH {
            if legal & !safe & COL_MASKS[col] != 0 {
                wdl[col] = -1;
            }
        }

        let mut best_col = heuristic_choice(own, mask);
        let mut best_score: i8 = -127;
        let mut legal_count = 0u32;
        let mut resolved = 0u32;

        for &col in COL_ORDER.iter() {
            let bit = safe & COL_MASKS[col];
            if bit == 0 {
                continue;
            }
            legal_count += 1;
            if self.timed_out {
                break;
            }
            let child_own = own ^ mask;
            let child_mask = mask | bit;
            let value = -self.solve(child_own, child_mask, moves + 1, true);
            if self.timed_out {
                break;
            }
            resolved += 1;
            let outcome = value.signum();
            wdl[col] = outcome;
            if outcome > best_score {
                best_score = outcome;
                best_col = col;
            }
            if outcome > 0 {
                return Analysis {
                    col,
                    score: 1,
                    wdl,
                    nodes: self.nodes,
                    elapsed_ms: start.elapsed().as_millis(),
                    solved: true,
                };
            }
        }

        if resolved == 0 {
            return Analysis {
                col: heuristic_choice(own, mask),
                score: -127,
                wdl,
                nodes: self.nodes,
                elapsed_ms: start.elapsed().as_millis(),
                solved: false,
            };
        }

        Analysis {
            col: best_col,
            score: best_score,
            wdl,
            nodes: self.nodes,
            elapsed_ms: start.elapsed().as_millis(),
            solved: resolved == legal_count,
        }
    }
}

fn parse_position(text: &str) -> Option<(u64, u64, u64)> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() < 3 {
        return None;
    }
    let own = fields[0].parse::<u64>().ok()?;
    let mask = fields[1].parse::<u64>().ok()?;
    let budget = fields[2].parse::<u64>().ok()?;
    Some((own, mask, budget))
}

fn write_result(result: &Analysis) {
    let mut out = io::stdout().lock();
    write!(
        out,
        "{} {} {} {} {}",
        result.col, result.score, result.nodes, result.elapsed_ms, result.solved as u8
    )
    .unwrap();
    for value in result.wdl {
        write!(out, " {}", value).unwrap();
    }
    writeln!(out).unwrap();
    out.flush().unwrap();
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--serve") {
        // Persistent search process: the transposition table survives across
        // moves, and the caller pays process startup only once per game.
        let stdin = io::stdin();
        let mut solver = Solver::new();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(value) => value,
                Err(_) => break,
            };
            if line.trim() == "quit" {
                break;
            }
            let Some((own, mask, budget)) = parse_position(&line) else {
                continue;
            };
            if mask & !BOARD != 0 || own & !mask != 0 {
                continue;
            }
            let result = solver.analyze(own, mask, budget);
            write_result(&result);
        }
        return;
    }

    let parsed = if args.len() >= 3 {
        parse_position(&args.join(" "))
    } else {
        let stdin = io::stdin();
        let line = stdin.lock().lines().next().and_then(|line| line.ok());
        line.as_deref().and_then(parse_position)
    };

    let (own, mask, budget) = match parsed {
        Some(values) => values,
        None => {
            eprintln!("usage: c4core <own-bits> <mask-bits> <budget-ms>");
            eprintln!("       c4core --serve");
            std::process::exit(2);
        }
    };

    if mask & !BOARD != 0 || own & !mask != 0 {
        eprintln!("invalid bitboards");
        std::process::exit(2);
    }

    let mut solver = Solver::new();
    let result = solver.analyze(own, mask, budget);
    write_result(&result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottom_matches_columns() {
        let mut expected = 0u64;
        for col in 0..WIDTH {
            expected |= 1u64 << (col * STRIDE);
        }
        assert_eq!(BOTTOM, expected);
        for col in 0..WIDTH {
            assert_eq!(COL_MASKS[col].count_ones(), HEIGHT as u32);
        }
    }

    #[test]
    fn detects_vertical_alignment() {
        let own = (1u64 << 0) | (1u64 << 1) | (1u64 << 2);
        let mask = own;
        let cells = winning_position(own, mask);
        assert_eq!(cells & COL_MASKS[0], 1u64 << 3);
    }

    #[test]
    fn takes_immediate_win() {
        let own = (1u64 << 0) | (1u64 << 1) | (1u64 << 2);
        let mask = own | (1u64 << 7) | (1u64 << 14) | (1u64 << 21);
        let mut solver = Solver::new();
        let result = solver.analyze(own, mask, 500);
        assert_eq!(result.col, 0);
        assert!(result.solved);
    }

    #[test]
    fn blocks_single_threat() {
        let opponent = (1u64 << 0) | (1u64 << 1) | (1u64 << 2);
        let own = (1u64 << 7) | (1u64 << 8);
        let mask = opponent | own;
        let mut solver = Solver::new();
        let result = solver.analyze(own, mask, 500);
        assert_eq!(result.col, 0);
    }
}
