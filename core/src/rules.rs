// How a game ends, following the site (wincondition.ts): repetition, then checkmate/stalemate,
// then the move rule, then insufficient material. Results use the site's condition names.

use crate::moves::{in_check, is_empty, legal};
use crate::pos::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct End {
    /// "1-0", "0-1" or "1/2-1/2".
    pub result: &'static str,
    /// "repetition", "checkmate", "stalemate", "moverule" or "insuffmat".
    pub condition: &'static str,
}

pub fn game_end(pos: &Pos) -> Option<End> {
    let h = hash(pos);
    if pos.history.iter().filter(|&&old| old == h).count() >= 2 {
        return Some(End { result: "1/2-1/2", condition: "repetition" });
    }
    if is_empty(&legal(pos)) {
        if in_check(pos, pos.stm) {
            let result = if pos.stm == WHITE { "0-1" } else { "1-0" };
            return Some(End { result, condition: "checkmate" });
        }
        return Some(End { result: "1/2-1/2", condition: "stalemate" });
    }
    if pos.move_rule.map_or(false, |limit| pos.clock >= limit) {
        return Some(End { result: "1/2-1/2", condition: "moverule" });
    }
    if insufficient_material(pos) {
        return Some(End { result: "1/2-1/2", condition: "insuffmat" });
    }
    None
}

// ---------------------------------------------------------------- insufficient material
//
// The site lists material sets that can never be mated, even with help (insufficientmaterial.ts).
// Each listed set is maximal: any subset of it is also a draw. Bishops are counted per square
// colour, because two bishops on one colour cannot do what one bishop of each colour can.

const INF: u32 = u32::MAX;

/// Piece counts per colour. `n[color][kind]` for kinds other than bishops; `b[color]` is
/// (bishops on one square colour, bishops on the other).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Material {
    n: [[u32; 7]; 2],
    b: [[u32; 2]; 2],
}

/// Builds a table entry. White's pieces first, then Black's; both kings are always added.
fn set(white: &[(u8, u32)], wb: [u32; 2], black: &[(u8, u32)], bb: [u32; 2]) -> Material {
    let mut m = Material::default();
    m.n[0][KING as usize] = 1;
    m.n[1][KING as usize] = 1;
    for &(k, c) in white {
        m.n[0][k as usize] = c;
    }
    for &(k, c) in black {
        m.n[1][k as usize] = c;
    }
    m.b = [wb, bb];
    m
}

/// Unbounded board, one king each (the Classical rows of INSUFFMAT_SCENARIOS).
fn table_unbounded() -> Vec<Material> {
    let z = [0, 0];
    vec![
        set(&[(QUEEN, 1)], z, &[(QUEEN, 1)], z),
        set(&[(QUEEN, 1)], z, &[(ROOK, 1)], z),
        set(&[(QUEEN, 1)], z, &[(KNIGHT, 1)], [1, 0]),
        set(&[(QUEEN, 1)], z, &[], [1, 1]),
        set(&[(QUEEN, 1)], z, &[(KNIGHT, 2)], z),
        set(&[(QUEEN, 1)], z, &[(PAWN, 1)], z),
        set(&[(ROOK, 1)], [1, 0], &[(KNIGHT, 1)], z),
        set(&[(ROOK, 1), (KNIGHT, 1)], z, &[(KNIGHT, 1)], z),
        set(&[(ROOK, 1)], z, &[(ROOK, 1)], z),
        set(&[(ROOK, 1)], z, &[], [1, 1]),
        set(&[(ROOK, 1)], z, &[(KNIGHT, 1)], [1, 0]),
        set(&[(ROOK, 1)], z, &[(KNIGHT, 2)], z),
        set(&[(ROOK, 1)], z, &[(PAWN, 1)], z),
        set(&[], [INF, 1], &[], z),
        set(&[(KNIGHT, 1)], [INF, 0], &[], z),
        set(&[], [INF, 0], &[(PAWN, 1)], z),
        set(&[(KNIGHT, 1)], [1, 1], &[], z),
        set(&[], [1, 1], &[], [1, 0]),
        set(&[], [1, 1], &[(KNIGHT, 1)], z),
        set(&[], [1, 1], &[(PAWN, 1)], z),
        set(&[(KNIGHT, 2)], [1, 0], &[], z),
        set(&[(KNIGHT, 1)], [1, 0], &[], [1, 0]),
        set(&[(KNIGHT, 1)], [1, 0], &[], [0, 1]),
        set(&[(KNIGHT, 1)], [1, 0], &[(KNIGHT, 1)], z),
        set(&[(KNIGHT, 1)], [1, 0], &[(PAWN, 1)], z),
        set(&[], [1, 0], &[(KNIGHT, 2)], z),
        set(&[(KNIGHT, 3)], z, &[], z),
        set(&[(KNIGHT, 2)], z, &[(KNIGHT, 1)], z),
        set(&[(KNIGHT, 2)], z, &[(PAWN, 1)], z),
        set(&[(PAWN, 3)], z, &[(PAWN, 1)], z),
    ]
}

/// A world border closer than 1,000,000 squares helps mating (INSUFFMAT_SCENARIOS_FINITE).
fn table_bounded() -> Vec<Material> {
    vec![set(&[], [INF, 0], &[], [INF, 0]), set(&[(KNIGHT, 1)], [0, 0], &[], [0, 0])]
}

/// Is every piece of `a` also in `b`, with at least as many? (isSubsumedBy)
fn subsumed(a: &Material, b: &Material) -> bool {
    for c in 0..2 {
        for k in 1..7 {
            if k == BISHOP as usize {
                continue;
            }
            if a.n[c][k] > 0 && (b.n[c][k] == 0 || a.n[c][k] > b.n[c][k]) {
                return false;
            }
        }
        let (ab, bb) = (a.b[c], b.b[c]);
        if ab[0] + ab[1] > 0 && (bb[0].saturating_add(bb[1]) == 0 || ab[0] > bb[0] || ab[1] > bb[1]) {
            return false;
        }
    }
    true
}

/// Bishop colours are relative: sort White's pair descending and turn Black's the same way, so
/// "same colour" vs "opposite colour" between the sides survives (normalizeBishopParities).
fn normalize(m: &mut Material) {
    let mut swapped = false;
    if m.b[0][0] + m.b[0][1] > 0 && m.b[0][0] < m.b[0][1] {
        m.b[0].swap(0, 1);
        swapped = true;
    }
    if m.b[1][0] + m.b[1][1] > 0 {
        let white_none_or_even = m.b[0][0] + m.b[0][1] == 0 || m.b[0][0] == m.b[0][1];
        if swapped || (white_none_or_even && m.b[1][0] < m.b[1][1]) {
            m.b[1].swap(0, 1);
        }
    }
}

fn invert(m: &Material) -> Material {
    let mut out = Material { n: [m.n[1], m.n[0]], b: [m.b[1], m.b[0]] };
    normalize(&mut out);
    out
}

fn square_colour(x: i64, y: i64) -> usize {
    ((x as i128 + y as i128).abs() % 2) as usize
}

fn insufficient_material(pos: &Pos) -> bool {
    if !pos.check_material || pos.pieces.len() >= 11 {
        return false;
    }
    // Up to two pawns that can promote are tried as every piece they could become.
    let pawns: Vec<&Piece> = pos.pieces.iter().filter(|p| p.kind == PAWN && pos.promo_rank[p.color as usize].is_some()).collect();
    if pawns.len() > 2 {
        return false;
    }
    let mut base = Material::default();
    for p in pos.pieces.iter().filter(|p| !(p.kind == PAWN && pos.promo_rank[p.color as usize].is_some())) {
        if p.kind == BISHOP {
            base.b[p.color as usize][square_colour(p.x, p.y)] += 1;
        } else {
            base.n[p.color as usize][p.kind as usize] += 1;
        }
    }
    let mut scenarios = vec![base];
    for pawn in pawns {
        let c = pawn.color as usize;
        let mut next = Vec::new();
        for s in &scenarios {
            for outcome in 0..6 {
                let mut t = *s;
                match outcome {
                    0 => t.n[c][PAWN as usize] += 1,
                    1 => t.n[c][QUEEN as usize] += 1,
                    2 => t.n[c][ROOK as usize] += 1,
                    3 => t.n[c][KNIGHT as usize] += 1,
                    4 => t.b[c][0] += 1,
                    _ => t.b[c][1] += 1,
                }
                next.push(t);
            }
        }
        scenarios = next;
    }

    let b = pos.border;
    let near = 1_000_000;
    let bounded = b.bottom >= -near || b.left >= -near || b.right <= near || b.top <= near;
    let table = if bounded { table_bounded() } else { table_unbounded() };
    let kings_only = |m: &Material| (1..7).all(|k| k == KING as usize || (m.n[0][k] == 0 && m.n[1][k] == 0)) && m.b == [[0, 0], [0, 0]];
    scenarios.iter_mut().all(|s| {
        normalize(s);
        let inv = invert(s);
        kings_only(s) || table.iter().any(|t| subsumed(s, t) || subsumed(&inv, t))
    })
}
