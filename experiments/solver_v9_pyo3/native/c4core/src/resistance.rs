//! Distance-sensitive search used only after the caller has proved a loss.
//!
//! Keep horizon values and their table separate from the exact WDL solver.
//! A completed iteration is usable guidance; an interrupted one is discarded.
use super::{possible, possible_non_losing, winning_position, CELLS, COL_MASKS, COL_ORDER};
use std::time::{Duration, Instant};

const MATE: i16 = 10_000;
const INF: i16 = 20_000;
const TABLE_BITS: usize = 18;

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    depth: u8,
    score: i16,
    flag: u8,
    col: usize,
}

pub struct Resistance {
    pub col: usize,
    pub depth: u8,
    pub score: i16,
    pub nodes: u64,
    pub elapsed_ms: u128,
    pub complete: bool,
}

struct Search {
    table: Vec<Entry>,
    deadline: Instant,
    nodes: u64,
}

// Among equally long defences, leave fewer easy replies and create threats.
// In a loss next turn, block as many immediate winning replies as possible.
fn pressure(own: u64, mask: u64, bit: u64) -> i32 {
    let occupied = mask | bit;
    let replies = possible(occupied);
    let opponent = own ^ mask;
    let wins = (winning_position(opponent, occupied) & replies).count_ones() as i32;
    let safe = possible_non_losing(opponent, occupied).count_ones() as i32;
    let threats = winning_position(own | bit, occupied).count_ones() as i32;
    -wins * 1024 - safe * 32 + threats
}

fn evaluate(own: u64, mask: u64) -> i16 {
    let opponent = own ^ mask;
    let threats = winning_position(own, mask).count_ones() as i16;
    let other = winning_position(opponent, mask).count_ones() as i16;
    let centre = (own & COL_MASKS[3]).count_ones() as i16
        - (opponent & COL_MASKS[3]).count_ones() as i16;
    8 * (threats - other) + centre
}

impl Search {
    fn negamax(&mut self, own: u64, mask: u64, depth: u8,
               mut alpha: i16, mut beta: i16) -> Option<i16> {
        self.nodes += 1;
        if self.nodes & 255 == 0 && Instant::now() >= self.deadline {
            return None;
        }
        let remaining = (CELLS - mask.count_ones()) as i16;
        if remaining == 0 {
            return Some(0);
        }
        if winning_position(own, mask) & possible(mask) != 0 {
            return Some(MATE + remaining); // Win on the next drop.
        }
        let safe = possible_non_losing(own, mask);
        if safe == 0 {
            return Some(-(MATE + remaining - 1)); // Opponent wins in two plies.
        }
        if depth == 0 {
            return Some(evaluate(own, mask));
        }
        let key = own + mask;
        let index = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - TABLE_BITS)) as usize;
        let old = self.table[index];
        let mut preferred = 7;
        if old.flag != 0 && old.key == key {
            preferred = old.col;
            if old.depth >= depth {
                match old.flag {
                    1 => return Some(old.score),
                    2 => alpha = alpha.max(old.score),
                    3 => beta = beta.min(old.score),
                    _ => (),
                }
                if alpha >= beta {
                    return Some(old.score);
                }
            }
        }
        let original_alpha = alpha;
        let original_beta = beta;
        let mut order = COL_ORDER;
        order.sort_by_key(|&c| std::cmp::Reverse(
            (if c == preferred { 1024 } else { 0 })
                + winning_position(own | (safe & COL_MASKS[c]), mask).count_ones()));
        let mut best = -INF;
        let mut best_col = 7;
        for col in order {
            let bit = safe & COL_MASKS[col];
            if bit == 0 {
                continue;
            }
            let score = -self.negamax(own ^ mask, mask | bit, depth - 1, -beta, -alpha)?;
            if score > best {
                best = score;
                best_col = col;
            }
            alpha = alpha.max(best);
            if alpha >= beta {
                break;
            }
        }
        let flag = if best <= original_alpha { 3 }
                   else if best >= original_beta { 2 } else { 1 };
        self.table[index] = Entry { key, depth, score: best, flag, col: best_col };
        Some(best)
    }
}

pub fn choose(own: u64, mask: u64, budget_ms: u64) -> Resistance {
    let start = Instant::now();
    let legal = possible(mask);
    let immediate = winning_position(own, mask) & legal;
    let safe = possible_non_losing(own, mask);
    let candidates = if immediate != 0 { immediate }
                     else if safe != 0 { safe } else { legal };
    let mut order: Vec<usize> = COL_ORDER.into_iter()
        .filter(|&c| candidates & COL_MASKS[c] != 0).collect();
    order.sort_by_key(|&c| std::cmp::Reverse(pressure(own, mask, legal & COL_MASKS[c])));
    let remaining = (CELLS - mask.count_ones()) as i16;
    let mut result = Resistance {
        col: order.first().copied().unwrap_or(3), depth: 0, score: 0,
        nodes: 0, elapsed_ms: 0, complete: false,
    };
    if immediate != 0 || safe == 0 || order.len() <= 1 {
        result.complete = immediate != 0 || safe == 0;
        result.score = if immediate != 0 { MATE + remaining }
                       else if legal != 0 && safe == 0 { -(MATE + remaining - 1) } else { 0 };
        return result;
    }
    let mut search = Search {
        table: vec![Entry::default(); 1 << TABLE_BITS],
        deadline: start + Duration::from_millis(budget_ms.max(1)), nodes: 0,
    };
    for depth in 1..=(remaining as u8) {
        let mut best = -INF;
        let mut selected = result.col;
        let mut finished = true;
        // Full windows give comparable root scores, including distance ties.
        for &col in &order {
            if Instant::now() >= search.deadline {
                finished = false;
                break;
            }
            let bit = legal & COL_MASKS[col];
            let Some(value) = search.negamax(own ^ mask, mask | bit, depth - 1, -INF, INF)
                else { finished = false; break; };
            if -value > best {
                best = -value;
                selected = col;
            }
        }
        if !finished {
            break;
        }
        result.col = selected;
        result.score = best;
        result.depth = depth;
        // Once every root move is refuted, mate distance is exact. Until then
        // horizon estimates must never be advertised as terminal distances.
        result.complete = best.abs() >= MATE || depth == remaining as u8;
        if result.complete {
            break;
        }
    }
    result.nodes = search.nodes;
    result.elapsed_ms = start.elapsed().as_millis();
    result
}
