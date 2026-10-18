// Exact legal moves. A slider's moves in one direction are kept as an interval of steps 1..=hi,
// the way the site stores them, because an open ray can hold ~2^64 squares. Every other move
// (pawn, knight, king, castling, and blocks/captures while in check) is a single move.

use crate::pos::*;

pub const KNIGHT_JUMPS: [(i64, i64); 8] = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
pub const KING_STEPS: [(i64, i64); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
pub const ORTHO: [(i64, i64); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];
pub const DIAG: [(i64, i64); 4] = [(1, 1), (-1, 1), (-1, -1), (1, -1)];
const PROMOTIONS: [u8; 4] = [QUEEN, ROOK, BISHOP, KNIGHT];

/// `hi` steps along a ray are legal. `UNBOUNDED` means the ray never ends (no border that side).
pub const UNBOUNDED: i128 = i128::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slide {
    pub fx: i64,
    pub fy: i64,
    pub dx: i64,
    pub dy: i64,
    pub hi: i128,
}

#[derive(Clone, Debug, Default)]
pub struct MoveSet {
    pub singles: Vec<Move>,
    pub slides: Vec<Slide>,
}

pub fn directions(kind: u8) -> &'static [(i64, i64)] {
    match kind {
        ROOK => &ORTHO,
        BISHOP => &DIAG,
        QUEEN => &KING_STEPS, // the same 8 unit vectors
        _ => &[],
    }
}

fn slides_along(kind: u8, dx: i64, dy: i64) -> bool {
    let diagonal = dx != 0 && dy != 0;
    match kind {
        QUEEN => true,
        ROOK => !diagonal,
        BISHOP => diagonal,
        _ => false,
    }
}

/// Steps from (x, y) to the last square inside the border along (dx, dy).
fn steps_to_border(pos: &Pos, x: i64, y: i64, dx: i64, dy: i64) -> i128 {
    let b = pos.border;
    let mut steps = UNBOUNDED;
    let mut limit = |room: i128, open: bool| {
        if !open {
            steps = steps.min(room);
        }
    };
    if dx > 0 {
        limit(b.right as i128 - x as i128, b.right == i64::MAX);
    }
    if dx < 0 {
        limit(x as i128 - b.left as i128, b.left == i64::MIN);
    }
    if dy > 0 {
        limit(b.top as i128 - y as i128, b.top == i64::MAX);
    }
    if dy < 0 {
        limit(y as i128 - b.bottom as i128, b.bottom == i64::MIN);
    }
    steps
}

/// How many steps a slider of `color` standing on (x, y) may go along (dx, dy):
/// up to the first piece (included if it is an enemy) or the border.
pub fn ray_limit(pos: &Pos, x: i64, y: i64, dx: i64, dy: i64, color: u8) -> i128 {
    let border = steps_to_border(pos, x, y, dx, dy);
    match first_on_ray(pos, x, y, dx, dy) {
        Some((i, k)) => {
            let k = if pos.pieces[i].color == color { k - 1 } else { k };
            k.min(border)
        }
        None => border,
    }
}

/// Is any piece strictly between (x, y) and n steps away along (dx, dy)?
fn blocked(pos: &Pos, x: i64, y: i64, dx: i64, dy: i64, n: i128) -> bool {
    first_on_ray(pos, x, y, dx, dy).map_or(false, |(_, k)| k < n)
}

/// Does piece `p` attack the square (x, y)?
pub fn attacks(pos: &Pos, p: &Piece, x: i64, y: i64) -> bool {
    let ddx = x as i128 - p.x as i128;
    let ddy = y as i128 - p.y as i128;
    if ddx == 0 && ddy == 0 {
        return false;
    }
    match p.kind {
        PAWN => ddx.abs() == 1 && ddy == if p.color == WHITE { 1 } else { -1 },
        KNIGHT => (ddx.abs() == 1 && ddy.abs() == 2) || (ddx.abs() == 2 && ddy.abs() == 1),
        KING => ddx.abs() <= 1 && ddy.abs() <= 1,
        _ => {
            if !(ddx == 0 || ddy == 0 || ddx.abs() == ddy.abs()) {
                return false;
            }
            let (dx, dy) = (ddx.signum() as i64, ddy.signum() as i64);
            slides_along(p.kind, dx, dy) && !blocked(pos, p.x, p.y, dx, dy, ddx.abs().max(ddy.abs()))
        }
    }
}

pub fn attacked(pos: &Pos, x: i64, y: i64, by: u8) -> bool {
    pos.pieces.iter().any(|p| p.color == by && attacks(pos, p, x, y))
}

pub fn king(pos: &Pos, color: u8) -> Option<Piece> {
    pos.pieces.iter().copied().find(|p| p.kind == KING && p.color == color)
}

pub fn in_check(pos: &Pos, color: u8) -> bool {
    king(pos, color).map_or(false, |k| attacked(pos, k.x, k.y, 1 - color))
}

/// First piece along (dx, dy) from (x, y), ignoring the piece on `skip`: (index, steps).
fn first_on_ray_except(pos: &Pos, x: i64, y: i64, dx: i64, dy: i64, skip: (i64, i64)) -> Option<(usize, i128)> {
    let mut best: Option<(usize, i128)> = None;
    for (i, p) in pos.pieces.iter().enumerate() {
        if (p.x, p.y) == skip {
            continue;
        }
        let ddx = p.x as i128 - x as i128;
        let ddy = p.y as i128 - y as i128;
        let k = if dx != 0 { ddx * dx as i128 } else { ddy * dy as i128 };
        if k > 0 && ddx == k * dx as i128 && ddy == k * dy as i128 && best.map_or(true, |(_, b)| k < b) {
            best = Some((i, k));
        }
    }
    best
}

/// Unit direction and step count from a to b, if they share a rank, file or diagonal.
fn line_between(ax: i64, ay: i64, bx: i64, by: i64) -> Option<(i64, i64, i128)> {
    let (dx, dy) = (bx as i128 - ax as i128, by as i128 - ay as i128);
    if (dx == 0 && dy == 0) || !(dx == 0 || dy == 0 || dx.abs() == dy.abs()) {
        return None;
    }
    Some((dx.signum() as i64, dy.signum() as i64, dx.abs().max(dy.abs())))
}

/// Does the legal move `m` give check? Same answer as playing it and testing, but without
/// copying the position: a direct check from the destination (the origin square counts as
/// empty), or a discovered check through the origin. Castling and en passant are played.
pub fn gives_check(pos: &Pos, m: &Move) -> bool {
    let Some(i) = piece_at(pos, m.fx, m.fy) else { return false };
    let mover = pos.pieces[i];
    let Some(k) = king(pos, 1 - mover.color) else { return false };
    let castle = mover.kind == KING && m.fy == m.ty && (m.tx as i128 - m.fx as i128).abs() == 2;
    let ep = mover.kind == PAWN && m.fx != m.tx && piece_at(pos, m.tx, m.ty).is_none();
    if castle || ep {
        return in_check(&play(pos, m), 1 - mover.color);
    }
    let kind = if m.promo != 0 { m.promo } else { mover.kind };
    let origin = (m.fx, m.fy);

    // Direct check from the destination.
    let (ddx, ddy) = (k.x as i128 - m.tx as i128, k.y as i128 - m.ty as i128);
    let direct = match kind {
        PAWN => ddx.abs() == 1 && ddy == if mover.color == WHITE { 1 } else { -1 },
        KNIGHT => (ddx.abs() == 1 && ddy.abs() == 2) || (ddx.abs() == 2 && ddy.abs() == 1),
        KING => ddx.abs() <= 1 && ddy.abs() <= 1,
        _ => match line_between(m.tx, m.ty, k.x, k.y) {
            Some((dx, dy, n)) => {
                slides_along(kind, dx, dy)
                    && first_on_ray_except(pos, m.tx, m.ty, dx, dy, origin).map_or(true, |(_, b)| b >= n)
            }
            None => false,
        },
    };
    if direct {
        return true;
    }

    // Discovered check: the origin was the only piece between the king and our slider.
    let Some((dx, dy, n_origin)) = line_between(k.x, k.y, m.fx, m.fy) else { return false };
    if first_on_ray(pos, k.x, k.y, dx, dy).map_or(true, |(_, b)| b < n_origin) {
        return false; // something else already blocks before the origin
    }
    let Some((j, n_slider)) = first_on_ray_except(pos, k.x, k.y, dx, dy, origin) else { return false };
    let slider = pos.pieces[j];
    if slider.color != mover.color || !slides_along(slider.kind, dx, dy) {
        return false;
    }
    // Unless the mover lands back on that same line between the king and the slider.
    let on_line = line_between(k.x, k.y, m.tx, m.ty).map_or(false, |(lx, ly, n)| (lx, ly) == (dx, dy) && n < n_slider);
    !on_line
}

// ---------------------------------------------------------------- pseudo-legal single moves

fn push_pawn_move(pos: &Pos, p: &Piece, tx: i64, ty: i64, out: &mut Vec<Move>) {
    if pos.promo_rank[p.color as usize] == Some(ty) {
        for kind in PROMOTIONS {
            out.push(Move { fx: p.x, fy: p.y, tx, ty, promo: kind });
        }
    } else {
        out.push(Move { fx: p.x, fy: p.y, tx, ty, promo: 0 });
    }
}

fn pawn_moves(pos: &Pos, p: &Piece, out: &mut Vec<Move>) {
    let dir = if p.color == WHITE { 1 } else { -1 };
    let empty = |x: i64, y: i64| in_border(pos, x, y) && piece_at(pos, x, y).is_none();
    let (x, y1, y2) = (p.x, p.y.saturating_add(dir), p.y.saturating_add(2 * dir));
    if empty(x, y1) {
        push_pawn_move(pos, p, x, y1, out);
        if p.right && empty(x, y2) {
            push_pawn_move(pos, p, x, y2, out);
        }
    }
    for cx in [x.saturating_sub(1), x.saturating_add(1)] {
        if !in_border(pos, cx, y1) {
            continue;
        }
        match piece_at(pos, cx, y1) {
            Some(j) if pos.pieces[j].color != p.color => push_pawn_move(pos, p, cx, y1, out),
            Some(_) => {}
            None => {
                if let Some(e) = pos.ep {
                    let victim = piece_at(pos, e.pawn_x, e.pawn_y);
                    if e.x == cx && e.y == y1 && victim.map_or(false, |j| pos.pieces[j].color != p.color) {
                        push_pawn_move(pos, p, cx, y1, out);
                    }
                }
            }
        }
    }
}

fn jump_moves(pos: &Pos, p: &Piece, offsets: &[(i64, i64)], out: &mut Vec<Move>) {
    for &(ox, oy) in offsets {
        let (Some(tx), Some(ty)) = (p.x.checked_add(ox), p.y.checked_add(oy)) else { continue };
        if !in_border(pos, tx, ty) {
            continue;
        }
        if piece_at(pos, tx, ty).map_or(true, |j| pos.pieces[j].color != p.color) {
            out.push(Move { fx: p.x, fy: p.y, tx, ty, promo: 0 });
        }
    }
}

/// Castling: two squares toward the nearest piece on the king's rank, if that piece is a
/// friendly non-pawn non-king with its special right, at least 3 squares away, and the king
/// is not in check and does not pass over an attacked square. The landing square is tested
/// later like any king move.
fn castling_moves(pos: &Pos, k: &Piece, out: &mut Vec<Move>) {
    if !k.right || attacked(pos, k.x, k.y, 1 - k.color) {
        return;
    }
    for dir in [-1i64, 1] {
        let Some((j, dist)) = first_on_ray(pos, k.x, k.y, dir, 0) else { continue };
        let partner = pos.pieces[j];
        if dist < 3 || partner.color != k.color || !partner.right || partner.kind == PAWN || partner.kind == KING {
            continue;
        }
        let over = Move { fx: k.x, fy: k.y, tx: k.x + dir, ty: k.y, promo: 0 };
        if in_check(&play(pos, &over), k.color) {
            continue;
        }
        out.push(Move { fx: k.x, fy: k.y, tx: k.x + 2 * dir, ty: k.y, promo: 0 });
    }
}

// ---------------------------------------------------------------- legality

/// Pieces of `color` pinned to their king: (x, y) of the pinned piece and the pin direction.
pub fn pins(pos: &Pos, color: u8) -> Vec<(i64, i64, i64, i64)> {
    let mut out = Vec::new();
    let Some(k) = king(pos, color) else { return out };
    for &(dx, dy) in &KING_STEPS {
        let Some((i, _)) = first_on_ray(pos, k.x, k.y, dx, dy) else { continue };
        let own = pos.pieces[i];
        if own.color != color {
            continue;
        }
        if let Some((j, _)) = first_on_ray(pos, own.x, own.y, dx, dy) {
            let enemy = pos.pieces[j];
            if enemy.color != color && slides_along(enemy.kind, dx, dy) {
                out.push((own.x, own.y, dx, dy));
            }
        }
    }
    out
}

/// While in single check, the one square on this slide (if any) that captures the checker or
/// blocks its line to the king.
fn resolve_check(s: &Slide, k: &Piece, checker: &Piece) -> Option<Move> {
    let to = |steps: i128| Move {
        fx: s.fx,
        fy: s.fy,
        tx: (s.fx as i128 + steps * s.dx as i128) as i64,
        ty: (s.fy as i128 + steps * s.dy as i128) as i64,
        promo: 0,
    };
    // Capture: the checker sits on the ray within reach.
    let cx = checker.x as i128 - s.fx as i128;
    let cy = checker.y as i128 - s.fy as i128;
    let kc = if s.dx != 0 { cx * s.dx as i128 } else { cy * s.dy as i128 };
    if kc >= 1 && kc <= s.hi && cx == kc * s.dx as i128 && cy == kc * s.dy as i128 {
        return Some(to(kc));
    }
    if !matches!(checker.kind, BISHOP | ROOK | QUEEN) {
        return None; // a knight or pawn check can only be answered by capturing it
    }
    // Block: solve  from + a*d = king + t*u  for a in 1..=hi and t strictly between king and checker.
    let (ux, uy) = ((checker.x as i128 - k.x as i128).signum(), (checker.y as i128 - k.y as i128).signum());
    let n = (checker.x as i128 - k.x as i128).abs().max((checker.y as i128 - k.y as i128).abs());
    let (dx, dy) = (s.dx as i128, s.dy as i128);
    let (rx, ry) = (k.x as i128 - s.fx as i128, k.y as i128 - s.fy as i128);
    let det = ux * dy - dx * uy;
    if det == 0 {
        return None; // parallel lines never meet strictly between king and checker
    }
    let a_num = ux * ry - uy * rx;
    let t_num = dx * ry - dy * rx;
    if a_num % det != 0 || t_num % det != 0 {
        return None; // the lines cross between squares
    }
    let (a, t) = (a_num / det, t_num / det);
    if a >= 1 && a <= s.hi && t >= 1 && t < n {
        Some(to(a))
    } else {
        None
    }
}

/// All legal moves of the side to move.
pub fn legal(pos: &Pos) -> MoveSet {
    let us = pos.stm;
    let mut singles = Vec::new();
    let mut slides = Vec::new();
    for p in pos.pieces.iter().filter(|p| p.color == us) {
        match p.kind {
            PAWN => pawn_moves(pos, p, &mut singles),
            KNIGHT => jump_moves(pos, p, &KNIGHT_JUMPS, &mut singles),
            KING => {
                jump_moves(pos, p, &KING_STEPS, &mut singles);
                castling_moves(pos, p, &mut singles);
            }
            _ => {
                for &(dx, dy) in directions(p.kind) {
                    let hi = ray_limit(pos, p.x, p.y, dx, dy, us);
                    if hi >= 1 {
                        slides.push(Slide { fx: p.x, fy: p.y, dx, dy, hi });
                    }
                }
            }
        }
    }

    let Some(k) = king(pos, us) else { return MoveSet { singles, slides } };
    let checkers: Vec<Piece> = pos.pieces.iter().filter(|q| q.color != us && attacks(pos, q, k.x, k.y)).copied().collect();
    if checkers.len() >= 2 {
        slides.clear(); // double check: only the king can move
        singles.retain(|m| m.fx == k.x && m.fy == k.y);
    } else if checkers.len() == 1 {
        for s in slides.drain(..) {
            singles.extend(resolve_check(&s, &k, &checkers[0]));
        }
    } else {
        // A slider that is not pinned can go anywhere on its rays; a pinned one only along the pin.
        let pinned = pins(pos, us);
        slides.retain(|s| match pinned.iter().find(|p| p.0 == s.fx && p.1 == s.fy) {
            None => true,
            Some(&(_, _, lx, ly)) => s.dx * ly == s.dy * lx,
        });
    }
    // Single moves are few, so each is verified by playing it.
    singles.retain(|m| !in_check(&play(pos, m), us));
    MoveSet { singles, slides }
}

pub fn is_empty(set: &MoveSet) -> bool {
    set.singles.is_empty() && set.slides.is_empty()
}

/// Is `m` in the legal set?
pub fn contains(set: &MoveSet, m: &Move) -> bool {
    if set.singles.contains(m) {
        return true;
    }
    m.promo == 0
        && set.slides.iter().any(|s| {
            if s.fx != m.fx || s.fy != m.fy {
                return false;
            }
            let ddx = m.tx as i128 - s.fx as i128;
            let ddy = m.ty as i128 - s.fy as i128;
            let k = if s.dx != 0 { ddx * s.dx as i128 } else { ddy * s.dy as i128 };
            k >= 1 && k <= s.hi && ddx == k * s.dx as i128 && ddy == k * s.dy as i128
        })
}

// ---------------------------------------------------------------- candidate moves (PLAN D4)
//
// The search can only play moves from a finite list. Single moves are always all included.
// On a slide, a square matters if it relates the slider to another piece, so the list takes:
//   - every square within `near` steps, and the squares just short of the blocker;
//   - every crossing of the ray with a rank, file or diagonal through any other piece, ±`spread`
//     (attacks, defences, pins, blocks, interpositions, checks, squares beside the king's lines);
//   - squares within 2 of either king;
//   - one square in the middle of every gap between those: inside a gap every square relates to
//     every piece in the same way, so one stands for all of them (PLAN §9 R1);
//   - two "far" squares beyond the last of those, relative to the position (not absolute
//     coordinates): last + 4 and last + 65536, for running away or relocating.
// So every legal slide square is either a candidate or shares its gap with one.

pub struct CandidateRule {
    pub near: i128,
    pub spread: i128,
}

pub const TREE: CandidateRule = CandidateRule { near: 8, spread: 1 };
pub const ROOT: CandidateRule = CandidateRule { near: 16, spread: 2 };

const FAR_STEPS: [i128; 2] = [4, 65536];

/// Steps `a` with from + a*d on the line through `p` of the given family, if the ray crosses it.
fn crossings(s: &Slide, p: &Piece) -> [Option<i128>; 4] {
    let (fx, fy, dx, dy) = (s.fx as i128, s.fy as i128, s.dx as i128, s.dy as i128);
    let (px, py) = (p.x as i128, p.y as i128);
    let solve = |num: i128, den: i128| if den != 0 && num % den == 0 { Some(num / den) } else { None };
    [
        solve(px - fx, dx),                     // file x = px
        solve(py - fy, dy),                     // rank y = py
        solve((px - py) - (fx - fy), dx - dy),  // diagonal x - y = const
        solve((px + py) - (fx + fy), dx + dy),  // anti-diagonal x + y = const
    ]
}

/// Steps along the slide's ray to the square (x, y), if the ray passes through it.
pub fn steps_to(s: &Slide, x: i128, y: i128) -> Option<i128> {
    let (ddx, ddy) = (x - s.fx as i128, y - s.fy as i128);
    let k = if s.dx != 0 { ddx * s.dx as i128 } else { ddy * s.dy as i128 };
    (k > 0 && ddx == k * s.dx as i128 && ddy == k * s.dy as i128).then_some(k)
}

/// The distances on one slide that the candidate rule keeps, sorted, without the far squares.
pub fn slide_breakpoints(pos: &Pos, s: &Slide, rule: &CandidateRule) -> Vec<i128> {
    let mut d: Vec<i128> = (1..=rule.near.min(s.hi)).collect();
    if let Some((_, k)) = first_on_ray(pos, s.fx, s.fy, s.dx, s.dy) {
        d.extend([k - 2, k - 1, k]); // k itself is legal only as a capture; filtered below
    }
    for p in &pos.pieces {
        if p.x == s.fx && p.y == s.fy {
            continue;
        }
        for a in crossings(s, p).into_iter().flatten() {
            d.extend((a - rule.spread)..=(a + rule.spread));
        }
        if p.kind == KING {
            for ox in -2..=2i128 {
                for oy in -2..=2i128 {
                    d.extend(steps_to(s, p.x as i128 + ox, p.y as i128 + oy));
                }
            }
        }
    }
    d.retain(|&k| k >= 1 && k <= s.hi);
    d.sort_unstable();
    d.dedup();
    d
}

/// The finite move list the search considers.
pub fn candidates(pos: &Pos, set: &MoveSet, rule: &CandidateRule) -> Vec<Move> {
    let mut out = set.singles.clone();
    for s in &set.slides {
        let mut d = slide_breakpoints(pos, s, rule);
        let last = d.last().copied().unwrap_or(0);
        let gaps: Vec<i128> = d.windows(2).filter(|w| w[1] - w[0] >= 2).map(|w| (w[0] + w[1]) / 2).collect();
        d.extend(gaps);
        for far in FAR_STEPS {
            if last + far <= s.hi {
                d.push(last + far);
            }
        }
        for k in d {
            out.push(Move {
                fx: s.fx,
                fy: s.fy,
                tx: (s.fx as i128 + k * s.dx as i128) as i64,
                ty: (s.fy as i128 + k * s.dy as i128) as i64,
                promo: 0,
            });
        }
    }
    out
}

/// The candidate that stands for a legal move `m`: `m` itself, the middle square of the gap it
/// lies in, or the near far square (last + 4) when it lies beyond every relevant square.
pub fn representative(pos: &Pos, set: &MoveSet, cands: &[Move], m: &Move, rule: &CandidateRule) -> Option<Move> {
    if cands.contains(m) {
        return Some(*m);
    }
    for s in &set.slides {
        if s.fx != m.fx || s.fy != m.fy || m.promo != 0 {
            continue;
        }
        let Some(k) = steps_to(s, m.tx as i128, m.ty as i128) else { continue };
        if k > s.hi {
            return None;
        }
        let bp = slide_breakpoints(pos, s, rule);
        let last = bp.last().copied().unwrap_or(0);
        let rep = if k > last {
            last + FAR_STEPS[0]
        } else {
            let w = bp.windows(2).find(|w| w[0] < k && k < w[1])?;
            (w[0] + w[1]) / 2
        };
        let r = Move {
            fx: s.fx,
            fy: s.fy,
            tx: (s.fx as i128 + rep * s.dx as i128) as i64,
            ty: (s.fy as i128 + rep * s.dy as i128) as i64,
            promo: 0,
        };
        return cands.contains(&r).then_some(r);
    }
    None
}

/// Every legal move as a list. Only possible when every ray is short (a bounded board).
pub fn expand(set: &MoveSet, max_steps: i128) -> Vec<Move> {
    let mut out = set.singles.clone();
    for s in &set.slides {
        assert!(s.hi <= max_steps, "expand: a ray of {} steps; use a bounded board", s.hi);
        for k in 1..=s.hi as i64 {
            out.push(Move { fx: s.fx, fy: s.fy, tx: s.fx + k * s.dx, ty: s.fy + k * s.dy, promo: 0 });
        }
    }
    out
}

/// Number of move sequences of length `depth` (game end is ignored, as in standard perft).
pub fn perft(pos: &Pos, depth: u32) -> u64 {
    let moves = expand(&legal(pos), 1 << 20);
    if depth <= 1 {
        return if depth == 1 { moves.len() as u64 } else { 1 };
    }
    moves.iter().map(|m| perft(&play(pos, m), depth - 1)).sum()
}
