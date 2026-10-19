// Turns positions into network input (PLAN D6, D7). Everything is relative: coordinates are seen
// from the side to move (Black's positions are flipped vertically) and optionally mirrored
// left-right; the net only ever sees differences between squares, log-bucketed.
//
// Per position:
//   tokens    [T][TOK_CH] u8    token 0 is the global token, then own pieces, then enemy pieces
//   pairs     [T][T][REL_CH] u8  relation of token i to token j
//   cands     [C][CAND_CH] u8   one row per candidate move
//   cand_rel  [C][T][REL_CH] u8  relation of the candidate's destination square to token j
//   target    index of the played move's candidate, or -1

use crate::moves::*;
use crate::pos::*;

pub const T: usize = 33; // global token + up to 32 pieces
pub const TOK_CH: usize = 10;
pub const REL_CH: usize = 5;
pub const CAND_CH: usize = 9;

/// Signed log2 bucket: 20 for 0, 20 ± (floor(log2|v|) + 1) otherwise, capped at 20 ± 20.
fn lg(v: i128) -> u8 {
    if v == 0 {
        return 20;
    }
    let b = (128 - v.unsigned_abs().leading_zeros()).min(20) as i128;
    (20 + v.signum() * b) as u8
}

pub struct Frame {
    sy: i128, // -1 when Black is to move: the board is seen upside down
    mx: i128, // -1 when mirrored left-right
}

impl Frame {
    fn d(&self, ax: i64, ay: i64, bx: i64, by: i64) -> (i128, i128) {
        ((bx as i128 - ax as i128) * self.mx, (by as i128 - ay as i128) * self.sy)
    }
}

/// Every piece as seen from one square: its direction (0-7, or NONE when not on a line or on the
/// square itself) and how many pieces stand strictly between. One pass over the pieces replaces a
/// scan of all pieces for every relation and every slider attack (PROGRESS A9). The results are
/// identical to `relation` / `attacks` / `count_attackers` (checked by `icore enccheck`).
const NONE: u8 = 8;

struct Rays {
    dir: Vec<u8>,
    between: Vec<u32>,
}

fn rays(pos: &Pos, x: i64, y: i64) -> Rays {
    let n = pos.pieces.len();
    let mut dir = vec![NONE; n];
    let mut dist = vec![0i128; n];
    for (i, p) in pos.pieces.iter().enumerate() {
        let (dx, dy) = (p.x as i128 - x as i128, p.y as i128 - y as i128);
        if (dx == 0 && dy == 0) || !(dx == 0 || dy == 0 || dx.abs() == dy.abs()) {
            continue;
        }
        dir[i] = ((dx.signum() + 1) * 3 + (dy.signum() + 1)) as u8; // 0..8 without 4 (the square itself)
        dir[i] = if dir[i] > 4 { dir[i] - 1 } else { dir[i] };
        dist[i] = dx.abs().max(dy.abs());
    }
    let mut between = vec![0u32; n];
    for i in 0..n {
        if dir[i] == NONE {
            continue;
        }
        between[i] = (0..n).filter(|&j| dir[j] == dir[i] && dist[j] < dist[i]).count() as u32;
    }
    Rays { dir, between }
}

/// `attacks(pos, p, x, y)` given the rays from (x, y): `d`/`b` are the direction and blockers of
/// p's square (p may be hypothetical, e.g. the moved piece; then d/b describe the target square).
fn attacks_ray(p: &Piece, x: i64, y: i64, d: u8, b: u32) -> bool {
    let ddx = x as i128 - p.x as i128;
    let ddy = y as i128 - p.y as i128;
    if ddx == 0 && ddy == 0 {
        return false;
    }
    match p.kind {
        PAWN => ddx.abs() == 1 && ddy == if p.color == WHITE { 1 } else { -1 },
        KNIGHT => (ddx.abs() == 1 && ddy.abs() == 2) || (ddx.abs() == 2 && ddy.abs() == 1),
        KING => ddx.abs() <= 1 && ddy.abs() <= 1,
        QUEEN => d != NONE && b == 0,
        ROOK => d != NONE && (ddx == 0 || ddy == 0) && b == 0,
        BISHOP => d != NONE && ddx != 0 && ddy != 0 && b == 0,
        _ => false,
    }
}

/// [dx bucket, dy bucket, line type, pieces between (+1), attack bits] for squares a -> b, with the
/// number of pieces between already known (`b`, from the rays of square a). Same output as
/// `encode_ref::relation`. (`b` from the rays of square a).
fn relation_ray(f: &Frame, ax: i64, ay: i64, bx: i64, by: i64, b: u32, attack: u8) -> [u8; REL_CH] {
    let (dx, dy) = f.d(ax, ay, bx, by);
    let line = if dx == 0 && dy == 0 {
        0
    } else if dy == 0 {
        1
    } else if dx == 0 {
        2
    } else if dx == dy {
        3
    } else if dx == -dy {
        4
    } else {
        0
    };
    let between = if line != 0 { 1 + b.min(2) as u8 } else { 0 };
    [lg(dx), lg(dy), line, between, attack]
}

/// `count_attackers` given the rays from (x, y).
fn count_attackers_ray(pos: &Pos, r: &Rays, x: i64, y: i64, color: u8, skip: Option<(i64, i64)>) -> u8 {
    pos.pieces
        .iter()
        .enumerate()
        .filter(|(i, q)| q.color == color && Some((q.x, q.y)) != skip && attacks_ray(q, x, y, r.dir[*i], r.between[*i]))
        .count()
        .min(3) as u8
}

pub struct Encoded {
    pub tokens: Vec<u8>,
    pub pairs: Vec<u8>,
    pub n: usize,          // number of candidates
    pub cands: Vec<u8>,    // n rows of CAND_CH
    pub cand_rel: Vec<u8>, // n blocks of T*REL_CH
    pub target: i32,
}

/// Encodes a position. `rep` = how often it occurred before; `played` = move to use as the
/// policy target (if any); `mirror` flips left-right (an exact symmetry of the rules).
/// `max_cands` caps the candidate list (training memory): the target always stays, the rest
/// are thinned evenly.
pub fn encode(pos: &Pos, rep: u32, played: Option<Move>, mirror: bool, max_cands: usize) -> Encoded {
    encode_keep(pos, rep, played, mirror, max_cands, None)
}

/// Like `encode`, but with `keep` the candidates are exactly those indices of the full candidate
/// list, in that order (RL training: all visited moves plus some unvisited, PROGRESS E12/A7).
pub fn encode_keep(pos: &Pos, rep: u32, played: Option<Move>, mirror: bool, max_cands: usize, keep: Option<&[u32]>) -> Encoded {
    let us = pos.stm;
    let f = Frame { sy: if us == WHITE { 1 } else { -1 }, mx: if mirror { -1 } else { 1 } };

    // Token order: global, own pieces, enemy pieces.
    let mut order: Vec<usize> = (0..pos.pieces.len()).collect();
    order.sort_by_key(|&i| (pos.pieces[i].color != us, pos.pieces[i].kind));
    order.truncate(T - 1);
    let pcs: Vec<Piece> = order.iter().map(|&i| pos.pieces[i]).collect();
    let tok_rays: Vec<Rays> = pcs.iter().map(|p| rays(pos, p.x, p.y)).collect();
    let pinned: Vec<(i64, i64)> = [WHITE, BLACK].iter().flat_map(|&c| pins(pos, c)).map(|p| (p.0, p.1)).collect();
    let checked = in_check(pos, us);

    let mut tokens = vec![0u8; T * TOK_CH];
    tokens[0] = 1; // global
    tokens[8] = (pos.clock / 10).min(11) as u8;
    tokens[9] = rep.min(2) as u8 + 3 * checked as u8;
    for (t, p) in pcs.iter().enumerate() {
        let row = &mut tokens[(t + 1) * TOK_CH..(t + 2) * TOK_CH];
        let own = p.color == us;
        row[0] = if own { 1 + p.kind } else { 7 + p.kind };
        row[1] = p.right as u8;
        if p.kind == PAWN {
            if let Some(r) = pos.promo_rank[p.color as usize] {
                let dir = if p.color == WHITE { 1 } else { -1 };
                row[2] = ((r as i128 - p.y as i128) * dir).clamp(1, 8) as u8;
            }
        }
        row[3] = count_attackers_ray(pos, &tok_rays[t], p.x, p.y, 1 - p.color, None);
        row[4] = count_attackers_ray(pos, &tok_rays[t], p.x, p.y, p.color, Some((p.x, p.y)));
        row[5] = pinned.contains(&(p.x, p.y)) as u8;
        row[6] = king(pos, 1 - p.color).map_or(false, |k| attacks(pos, p, k.x, k.y)) as u8;
        if p.kind == KING {
            let mut mask = 0u8;
            for (i, &(ox, oy)) in KING_STEPS.iter().enumerate() {
                let (Some(x), Some(y)) = (p.x.checked_add(ox), p.y.checked_add(oy)) else { continue };
                let blocked = !in_border(pos, x, y)
                    || piece_at(pos, x, y).map_or(false, |j| pos.pieces[j].color == p.color)
                    || {
                        let r = rays(pos, x, y);
                        pos.pieces.iter().enumerate().any(|(i, q)| q.color != p.color && attacks_ray(q, x, y, r.dir[i], r.between[i]))
                    };
                // Bits follow the canonical frame, so a mirrored or flipped board stays consistent.
                let (cx, cy) = (ox as i128 * f.mx, oy as i128 * f.sy);
                let bit = KING_STEPS.iter().position(|&(a, b)| a as i128 == cx && b as i128 == cy).unwrap_or(i);
                mask |= (blocked as u8) << bit;
            }
            row[7] = mask;
        }
    }

    let n = pcs.len() + 1;
    let mut pairs = vec![0u8; T * T * REL_CH];
    for i in 0..n {
        for j in 0..n {
            let rel = if i == 0 || j == 0 {
                [0, 0, 5, 0, 0]
            } else {
                let (a, b) = (&pcs[i - 1], &pcs[j - 1]);
                let r = &tok_rays[i - 1]; // rays from a's square; b is piece order[j - 1]
                let (d, bl) = (r.dir[order[j - 1]], r.between[order[j - 1]]);
                let att = attacks_ray(a, b.x, b.y, d, bl) as u8 | (attacks_ray(b, a.x, a.y, d, bl) as u8) << 1;
                relation_ray(&f, a.x, a.y, b.x, b.y, bl, att)
            };
            pairs[(i * T + j) * REL_CH..(i * T + j + 1) * REL_CH].copy_from_slice(&rel);
        }
    }

    let set = legal(pos);
    let mut moves = candidates(pos, &set, &TREE);
    let target_move = played.and_then(|m| representative(pos, &set, &moves, &m, &TREE));
    if let Some(k) = keep {
        moves = k.iter().filter_map(|&i| moves.get(i as usize).copied()).collect();
    } else if moves.len() > max_cands.max(1) {
        let keep = max_cands.max(1);
        let stride = moves.len() as f64 / keep as f64;
        let mut thinned: Vec<Move> = (0..keep).map(|i| moves[(i as f64 * stride) as usize]).collect();
        if let Some(t) = target_move {
            if !thinned.contains(&t) {
                thinned[keep - 1] = t;
            }
        }
        moves = thinned;
    }
    let mut target = -1;
    let mut cands = Vec::with_capacity(moves.len() * CAND_CH);
    let mut cand_rel = vec![0u8; moves.len() * T * REL_CH];
    for (ci, m) in moves.iter().enumerate() {
        if Some(*m) == target_move {
            target = ci as i32;
        }
        let from_tok = pcs.iter().position(|p| p.x == m.fx && p.y == m.fy).map_or(0, |t| t + 1);
        let mover = pcs.get(from_tok.wrapping_sub(1)).copied().unwrap_or(pcs[0]);
        let victim = piece_at(pos, m.tx, m.ty).map(|j| pos.pieces[j].kind);
        let is_ep = mover.kind == PAWN && m.fx != m.tx && victim.is_none();
        let castle = mover.kind == KING && (m.tx as i128 - m.fx as i128).abs() == 2 && m.fy == m.ty;
        let double = mover.kind == PAWN && (m.ty as i128 - m.fy as i128).abs() == 2;
        let (dx, dy) = f.d(m.fx, m.fy, m.tx, m.ty);
        let dir = if dx.abs() == dy.abs() || dx == 0 || dy == 0 { 1 + (dx.signum() + 1) * 3 + (dy.signum() + 1) } else { 0 };
        let dest = rays(pos, m.tx, m.ty);
        cands.extend_from_slice(&[
            from_tok as u8,
            m.promo,
            victim.unwrap_or(if is_ep { PAWN } else { 0 }),
            if castle { 1 } else if is_ep { 2 } else if double { 3 } else { 0 },
            count_attackers_ray(pos, &dest, m.tx, m.ty, 1 - us, None),
            count_attackers_ray(pos, &dest, m.tx, m.ty, us, Some((m.fx, m.fy))),
            gives_check(pos, m) as u8,
            lg(dx.abs().max(dy.abs())),
            dir as u8,
        ]);
        // The moving piece as it would stand on the destination square.
        let moved = Piece { x: m.tx, y: m.ty, kind: if m.promo != 0 { m.promo } else { mover.kind }, ..mover };
        let block = &mut cand_rel[ci * T * REL_CH..(ci + 1) * T * REL_CH];
        for j in 0..n {
            let rel = if j == 0 {
                [0, 0, 5, 0, 0]
            } else {
                let b = &pcs[j - 1];
                let (d, bl) = (dest.dir[order[j - 1]], dest.between[order[j - 1]]);
                let att = attacks_ray(b, m.tx, m.ty, d, bl) as u8 | (attacks_ray(&moved, b.x, b.y, d, bl) as u8) << 1;
                relation_ray(&f, m.tx, m.ty, b.x, b.y, bl, att)
            };
            block[j * REL_CH..(j + 1) * REL_CH].copy_from_slice(&rel);
        }
    }
    Encoded { tokens, pairs, n: moves.len(), cands, cand_rel, target }
}
