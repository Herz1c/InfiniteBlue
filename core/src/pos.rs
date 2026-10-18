// A position is a flat list of pieces plus the game state. Classical pieces only (K Q R B N P).
// Coordinates are i64. Differences between coordinates are always computed in i128, because two
// pieces can stand almost 2^64 squares apart inside the engine border.

pub const WHITE: u8 = 0;
pub const BLACK: u8 = 1;

pub const PAWN: u8 = 1;
pub const KNIGHT: u8 = 2;
pub const BISHOP: u8 = 3;
pub const ROOK: u8 = 4;
pub const QUEEN: u8 = 5;
pub const KING: u8 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub x: i64,
    pub y: i64,
    pub kind: u8,
    pub color: u8,
    /// Special right ("+" in ICN): a pawn's double step, or a king's / rook's castling right.
    pub right: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub fx: i64,
    pub fy: i64,
    pub tx: i64,
    pub ty: i64,
    /// 0, or the kind a pawn promotes to.
    pub promo: u8,
}

/// En passant: the square a capturing pawn lands on, and where the pawn it takes stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ep {
    pub x: i64,
    pub y: i64,
    pub pawn_x: i64,
    pub pawn_y: i64,
}

/// The inclusive playable box. i64::MIN / i64::MAX on a side means there is no border there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Border {
    pub left: i64,
    pub right: i64,
    pub bottom: i64,
    pub top: i64,
}

pub const NO_BORDER: Border = Border { left: i64::MIN, right: i64::MAX, bottom: i64::MIN, top: i64::MAX };

#[derive(Clone, Debug)]
pub struct Pos {
    pub pieces: Vec<Piece>,
    pub stm: u8,
    pub ep: Option<Ep>,
    /// Plies since the last capture or pawn move.
    pub clock: u32,
    /// The game is drawn when `clock` reaches this (100 = the 50-move rule).
    pub move_rule: Option<u32>,
    pub fullmove: u32,
    /// Rank on which White's / Black's pawns promote, if they can.
    pub promo_rank: [Option<i64>; 2],
    pub border: Border,
    /// Hashes of earlier positions that can still repeat: those since the last pawn move,
    /// capture or lost special right. This is the site's repetition window.
    pub history: Vec<u64>,
    /// The site tests insufficient material only right after a capture or a promotion,
    /// or when no move has been played yet.
    pub check_material: bool,
}

pub fn piece_at(pos: &Pos, x: i64, y: i64) -> Option<usize> {
    pos.pieces.iter().position(|p| p.x == x && p.y == y)
}

pub fn in_border(pos: &Pos, x: i64, y: i64) -> bool {
    let b = pos.border;
    x >= b.left && x <= b.right && y >= b.bottom && y <= b.top
}

/// Nearest piece from (x, y) in unit direction (dx, dy): its index and how many steps away.
pub fn first_on_ray(pos: &Pos, x: i64, y: i64, dx: i64, dy: i64) -> Option<(usize, i128)> {
    let mut best: Option<(usize, i128)> = None;
    for (i, p) in pos.pieces.iter().enumerate() {
        let ddx = p.x as i128 - x as i128;
        let ddy = p.y as i128 - y as i128;
        let k = if dx != 0 { ddx * dx as i128 } else { ddy * dy as i128 };
        if k <= 0 || ddx != k * dx as i128 || ddy != k * dy as i128 {
            continue;
        }
        if best.map_or(true, |(_, bk)| k < bk) {
            best = Some((i, k));
        }
    }
    best
}

// ---------------------------------------------------------------- hashing

// splitmix64: a cheap, well-mixed 64-bit hash. Coordinates are unbounded, so a table of random
// keys per square (classic Zobrist) is impossible; each piece's key is computed instead.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn piece_key(p: &Piece) -> u64 {
    let code = p.kind as u64 * 4 + p.color as u64 * 2 + p.right as u64;
    mix(mix(mix(p.x as u64) ^ p.y as u64) ^ code)
}

/// Position hash: pieces (with their special rights), side to move and en-passant state.
/// XOR makes it independent of the order of the piece list.
pub fn hash(pos: &Pos) -> u64 {
    let mut h = if pos.stm == BLACK { 0x6A09_E667_F3BC_C909 } else { 0 };
    for p in &pos.pieces {
        h ^= piece_key(p);
    }
    if let Some(e) = pos.ep {
        h ^= mix(mix(e.x as u64 ^ 0xE9) ^ e.y as u64);
    }
    h
}

// ---------------------------------------------------------------- making a move

/// The site revokes castling rights that can no longer be used. When a king or rook with its
/// right moves or is captured, each friendly piece of the other role (king vs. non-king) on that
/// rank loses its right too, unless it still has a partner: a piece of the other role on its rank
/// with a right, at least 3 squares away (movepiece.ts cascadeDeleteSpecialRights). This never
/// changes which moves are legal, but it decides which positions count as repetitions.
fn orphaned_rights(pos: &Pos, gone: &Piece) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    if !gone.right || gone.kind == PAWN {
        return out;
    }
    let is_king = |p: &Piece| p.kind == KING;
    let castler = |p: &Piece, color: u8, y: i64| p.color == color && p.kind != PAWN && p.right && p.y == y;
    let is_gone = |p: &Piece| p.x == gone.x && p.y == gone.y;
    for c in pos.pieces.iter().filter(|c| castler(c, gone.color, gone.y) && is_king(c) != is_king(gone) && !is_gone(c)) {
        let has_partner = pos.pieces.iter().any(|q| {
            castler(q, c.color, c.y) && is_king(q) != is_king(c) && !is_gone(q) && (q.x as i128 - c.x as i128).abs() >= 3
        });
        if !has_partner {
            out.push((c.x, c.y));
        }
    }
    out
}

/// Returns the position after `m`. `m` must be legal (see `moves::legal`); nothing is checked here.
pub fn play(pos: &Pos, m: &Move) -> Pos {
    let mut next = pos.clone();
    let i = piece_at(pos, m.fx, m.fy).expect("play: no piece on the from-square");
    let mover = pos.pieces[i];
    let dx = m.tx as i128 - m.fx as i128;
    let dy = m.ty as i128 - m.fy as i128;

    // The captured piece stands on the destination, or beside it for en passant.
    let mut victim = piece_at(pos, m.tx, m.ty);
    if victim.is_none() && mover.kind == PAWN && dx != 0 {
        if let Some(e) = pos.ep {
            if e.x == m.tx && e.y == m.ty {
                victim = piece_at(pos, e.pawn_x, e.pawn_y);
            }
        }
    }

    // Castling: the king moves two squares along its rank, and the nearest piece on that side
    // lands on the square the king passed over.
    let mut lost_right = mover.right;
    let mut revoked = orphaned_rights(pos, &mover);
    if let Some(j) = victim {
        revoked.extend(orphaned_rights(pos, &pos.pieces[j]));
    }
    if mover.kind == KING && dy == 0 && dx.abs() == 2 {
        let dir = dx.signum() as i64;
        let (j, _) = first_on_ray(pos, m.fx, m.fy, dir, 0).expect("castling without a partner");
        revoked.extend(orphaned_rights(pos, &pos.pieces[j]));
        next.pieces[j].x = m.fx + dir;
        next.pieces[j].right = false;
        lost_right = true;
    }
    // Pieces whose rights are revoked do not move in this turn, so they are found by square.
    for (x, y) in revoked {
        if let Some(j) = piece_at(&next, x, y) {
            next.pieces[j].right = false;
        }
    }

    let p = &mut next.pieces[i];
    p.x = m.tx;
    p.y = m.ty;
    p.right = false;
    if m.promo != 0 {
        p.kind = m.promo;
    }
    let capture = victim.is_some();
    if let Some(j) = victim {
        next.pieces.swap_remove(j); // done last: it moves another piece into slot j
    }

    next.ep = if mover.kind == PAWN && dy.abs() == 2 {
        Some(Ep { x: m.fx, y: ((m.fy as i128 + m.ty as i128) / 2) as i64, pawn_x: m.tx, pawn_y: m.ty })
    } else {
        None
    };
    next.clock = if capture || mover.kind == PAWN { 0 } else { pos.clock + 1 };
    if capture || mover.kind == PAWN || lost_right {
        next.history.clear(); // nothing before an irreversible move can repeat
    } else {
        next.history.push(hash(pos));
    }
    next.check_material = capture || m.promo != 0;
    if pos.stm == BLACK {
        next.fullmove += 1;
    }
    next.stm = 1 - pos.stm;
    next
}

// ---------------------------------------------------------------- ICN

const LETTERS: [(char, u8); 6] = [('p', PAWN), ('n', KNIGHT), ('b', BISHOP), ('r', ROOK), ('q', QUEEN), ('k', KING)];

fn kind_from_letter(c: char) -> Option<u8> {
    LETTERS.iter().find(|(l, _)| *l == c.to_ascii_lowercase()).map(|(_, k)| *k)
}

pub fn letter(kind: u8) -> char {
    LETTERS.iter().find(|(_, k)| *k == kind).map(|(l, _)| *l).unwrap()
}

fn int(s: &str) -> Result<i64, String> {
    s.parse::<i64>().map_err(|_| format!("bad integer {s:?}"))
}

/// "x,y>x,y" with an optional "=Q" (either case) for a promotion.
pub fn parse_move(tok: &str) -> Result<Move, String> {
    let (squares, promo) = match tok.split_once('=') {
        Some((sq, p)) => (sq, p.chars().next().and_then(kind_from_letter).ok_or(format!("bad promotion in {tok:?}"))?),
        None => (tok, 0),
    };
    let (from, to) = squares.split_once('>').ok_or(format!("bad move {tok:?}"))?;
    let (fx, fy) = from.split_once(',').ok_or(format!("bad move {tok:?}"))?;
    let (tx, ty) = to.split_once(',').ok_or(format!("bad move {tok:?}"))?;
    Ok(Move { fx: int(fx)?, fy: int(fy)?, tx: int(tx)?, ty: int(ty)?, promo })
}

/// The site's compact move token. The promotion letter is uppercase for White.
pub fn move_token(m: &Move, color: u8) -> String {
    let mut s = format!("{},{}>{},{}", m.fx, m.fy, m.tx, m.ty);
    if m.promo != 0 {
        let c = letter(m.promo);
        s.push('=');
        s.push(if color == WHITE { c.to_ascii_uppercase() } else { c });
    }
    s
}

/// Drops `[Key "value"]` metadata, whose values may contain spaces.
fn strip_metadata(icn: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for c in icn.chars() {
        match c {
            '[' => depth += 1,
            ']' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

fn is_numbers(tok: &str) -> bool {
    tok.split(',').all(|v| v == "_" || int(v).is_ok())
}

/// Parses an ICN into the start position and the move list that follows it.
/// Only Classical pieces are supported; win-condition tokens are accepted and ignored.
pub fn parse_icn(icn: &str) -> Result<(Pos, Vec<Move>), String> {
    let mut pos = Pos {
        pieces: Vec::new(),
        stm: WHITE,
        ep: None,
        clock: 0,
        move_rule: None,
        fullmove: 1,
        promo_rank: [None, None],
        border: NO_BORDER,
        history: Vec::new(),
        check_material: true,
    };
    let mut moves = Vec::new();
    let mut ep_square: Option<(i64, i64)> = None;

    for tok in strip_metadata(icn).split_whitespace() {
        if tok == "w" || tok == "b" {
            pos.stm = if tok == "w" { WHITE } else { BLACK };
        } else if tok.contains('>') {
            for m in tok.split('|') {
                moves.push(parse_move(m)?);
            }
        } else if let Some(inner) = tok.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
            // "(8|1)" or "(8;q,r,b,n|1;q,r,b,n)" are promotions; "(checkmate|checkmate)" is not.
            if inner.starts_with(|c: char| c.is_ascii_digit() || c == '-') {
                for (side, part) in inner.split('|').take(2).enumerate() {
                    let ranks = part.split(';').next().unwrap();
                    if !ranks.is_empty() {
                        pos.promo_rank[side] = Some(int(ranks.split(',').next().unwrap())?);
                    }
                }
            }
        } else if let Some((clock, limit)) = tok.split_once('/') {
            pos.clock = int(clock)? as u32;
            pos.move_rule = Some(int(limit)? as u32);
        } else if tok.chars().all(|c| c.is_ascii_digit()) {
            pos.fullmove = int(tok)? as u32;
        } else if tok.chars().all(|c| c.is_ascii_alphabetic() || c == ',') {
            // win conditions such as "checkmate"
        } else if is_numbers(tok) {
            let v: Vec<&str> = tok.split(',').collect();
            if v.len() == 2 {
                ep_square = Some((int(v[0])?, int(v[1])?));
            } else if v.len() == 4 {
                let side = |s: &str, inf: i64| if s == "_" { Ok(inf) } else { int(s) };
                pos.border = Border {
                    left: side(v[0], i64::MIN)?,
                    right: side(v[1], i64::MAX)?,
                    bottom: side(v[2], i64::MIN)?,
                    top: side(v[3], i64::MAX)?,
                };
            } else {
                return Err(format!("unknown numeric token {tok:?}"));
            }
        } else {
            for entry in tok.split('|') {
                pos.pieces.push(parse_piece(entry)?);
            }
        }
    }
    // The ICN gives only the en-passant square; the pawn that can be taken stands one step
    // past it, in the direction the opponent's pawns move.
    if let Some((x, y)) = ep_square {
        let pawn_y = if pos.stm == WHITE { y - 1 } else { y + 1 };
        pos.ep = Some(Ep { x, y, pawn_x: x, pawn_y });
    }
    Ok((pos, moves))
}

fn parse_piece(entry: &str) -> Result<Piece, String> {
    let split = entry.find(|c: char| c == '-' || c.is_ascii_digit()).ok_or(format!("bad piece {entry:?}"))?;
    let (abbr, rest) = entry.split_at(split);
    let mut chars = abbr.chars();
    let c = chars.next().ok_or(format!("bad piece {entry:?}"))?;
    if chars.next().is_some() {
        return Err(format!("unsupported piece {entry:?} (Classical only)"));
    }
    let kind = kind_from_letter(c).ok_or(format!("unsupported piece {entry:?}"))?;
    let right = rest.ends_with('+');
    let (x, y) = rest.trim_end_matches('+').split_once(',').ok_or(format!("bad piece {entry:?}"))?;
    Ok(Piece { x: int(x)?, y: int(y)?, kind, color: if c.is_ascii_uppercase() { WHITE } else { BLACK }, right })
}

/// The position as a stand-alone ICN (no move list; the repetition history is not representable).
pub fn to_icn(pos: &Pos) -> String {
    let mut parts = vec![if pos.stm == WHITE { "w".to_string() } else { "b".to_string() }];
    if let Some(e) = pos.ep {
        parts.push(format!("{},{}", e.x, e.y));
    }
    if let Some(limit) = pos.move_rule {
        parts.push(format!("{}/{}", pos.clock, limit));
    }
    parts.push(pos.fullmove.to_string());
    let rank = |r: Option<i64>| r.map_or(String::new(), |v| v.to_string());
    parts.push(format!("({}|{})", rank(pos.promo_rank[0]), rank(pos.promo_rank[1])));
    let b = pos.border;
    if b != NO_BORDER {
        let side = |v: i64, inf: i64| if v == inf { "_".to_string() } else { v.to_string() };
        parts.push(format!(
            "{},{},{},{}",
            side(b.left, i64::MIN),
            side(b.right, i64::MAX),
            side(b.bottom, i64::MIN),
            side(b.top, i64::MAX)
        ));
    }
    let pieces: Vec<String> = pos
        .pieces
        .iter()
        .map(|p| {
            let c = letter(p.kind);
            let c = if p.color == WHITE { c.to_ascii_uppercase() } else { c };
            format!("{c}{},{}{}", p.x, p.y, if p.right { "+" } else { "" })
        })
        .collect();
    parts.push(pieces.join("|"));
    parts.join(" ")
}

/// Parses an ICN and plays its move list (without checking legality).
pub fn parse_icn_and_play(icn: &str) -> Result<Pos, String> {
    let (mut pos, moves) = parse_icn(icn)?;
    for m in &moves {
        pos = play(&pos, m);
    }
    Ok(pos)
}
