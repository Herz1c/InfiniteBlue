// Test and debugging CLI. Reads one ICN per line on stdin.
//   icore perft <depth>   perft counts for depths 1..=depth (bounded boards only)
//   icore moves           the exact legal move set as one JSON line per position
//   icore replay          checks every move of each game for legality, then prints how it ends
//   icore positions       every position of each game, as a stand-alone ICN per line
//   icore random <games> <max_plies> <seed>   random playouts from each input ICN, printed as
//                         positions; slides go 1-8 steps or up to 2^62 (stress test for far moves)
//   icore bench           times legal moves, candidates, game_end and the full encoder per position
//   icore enccheck        the fast encoder must equal the frozen reference (encode_ref) byte for byte
use icore::moves::{
    candidates, contains, legal, perft, slide_breakpoints, steps_to, CandidateRule, MoveSet, ROOT, TREE, UNBOUNDED,
};
use icore::pos::{parse_icn, parse_icn_and_play, parse_move, play, to_icn, Move, Pos};
use icore::rules::game_end;
use std::io::{BufRead, Write};
use std::time::Instant;

fn moves_json(set: &MoveSet) -> String {
    let singles: Vec<String> = set
        .singles
        .iter()
        .map(|m| format!("[{},{},{},{},{}]", m.fx, m.fy, m.tx, m.ty, m.promo))
        .collect();
    let slides: Vec<String> = set
        .slides
        .iter()
        .map(|s| {
            let hi = if s.hi == UNBOUNDED { "null".to_string() } else { s.hi.to_string() };
            format!("[{},{},{},{},{}]", s.fx, s.fy, s.dx, s.dy, hi)
        })
        .collect();
    format!("{{\"singles\":[{}],\"slides\":[{}]}}", singles.join(","), slides.join(","))
}

fn end_json(pos: &icore::pos::Pos) -> String {
    match game_end(pos) {
        Some(e) => format!("{{\"result\":\"{}\",\"condition\":\"{}\"}}", e.result, e.condition),
        None => "null".to_string(),
    }
}

/// 0 = in the candidate list, 1 = a far slide beyond every relevant square (a far candidate
/// stands in for it), 2 = inside a gap whose middle square is a candidate, 3 = missed,
/// 4 = not even legal.
fn classify(pos: &Pos, set: &MoveSet, cands: &[Move], m: &Move, rule: &CandidateRule) -> usize {
    if !contains(set, m) {
        return 4;
    }
    if cands.contains(m) {
        return 0;
    }
    for s in &set.slides {
        if s.fx != m.fx || s.fy != m.fy {
            continue;
        }
        if let Some(k) = steps_to(s, m.tx as i128, m.ty as i128) {
            let bp = slide_breakpoints(pos, s, rule);
            let last = bp.last().copied().unwrap_or(0);
            if k > last {
                return if last + 4 <= s.hi { 1 } else { 3 };
            }
            if bp.windows(2).any(|w| w[0] < k && k < w[1]) {
                return 2;
            }
        }
    }
    3
}

/// xorshift64*: a tiny deterministic random generator for the stress playouts.
fn rand(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// A random legal move; a slide's length is short half the time and log-uniform otherwise.
fn random_move(pos: &Pos, rng: &mut u64) -> Option<Move> {
    let set = legal(pos);
    let options = set.singles.len() + set.slides.len();
    if options == 0 {
        return None;
    }
    let i = (rand(rng) % options as u64) as usize;
    if i < set.singles.len() {
        return Some(set.singles[i]);
    }
    let s = set.slides[i - set.singles.len()];
    let hi = s.hi.min(1 << 62);
    let k = if rand(rng) % 2 == 0 {
        1 + (rand(rng) as i128 % hi.min(8))
    } else {
        let bits = rand(rng) % 63;
        (1 + (rand(rng) as i128 & ((1i128 << bits) - 1))).min(hi)
    };
    Some(Move {
        fx: s.fx,
        fy: s.fy,
        tx: (s.fx as i128 + k * s.dx as i128) as i64,
        ty: (s.fy as i128 + k * s.dy as i128) as i64,
        promo: 0,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    if cmd == "enccheck" {
        // Every position, mirrored and not, without a target and with each 7th candidate as the
        // played move and a 64-candidate cap (the training path): all outputs must be identical.
        let (mut n, mut bad) = (0usize, 0usize);
        for line in stdin.lock().lines() {
            let line = line.unwrap();
            let Ok(pos) = parse_icn_and_play(line.trim()) else { continue };
            let cands = candidates(&pos, &legal(&pos), &TREE);
            let mut cases: Vec<(Option<Move>, bool, usize)> = vec![(None, false, usize::MAX), (None, true, usize::MAX)];
            for (i, m) in cands.iter().enumerate().step_by(7).take(3) {
                cases.push((Some(*m), i % 2 == 1, 64));
            }
            for (played, mirror, cap) in cases {
                let a = icore::encode::encode(&pos, 1, played, mirror, cap);
                let b = icore::encode_ref::encode(&pos, 1, played, mirror, cap);
                n += 1;
                if a.tokens != b.tokens || a.pairs != b.pairs || a.n != b.n || a.cands != b.cands || a.cand_rel != b.cand_rel || a.target != b.target {
                    bad += 1;
                    if bad <= 5 {
                        let what: Vec<&str> = [("tokens", a.tokens != b.tokens), ("pairs", a.pairs != b.pairs), ("cands", a.cands != b.cands), ("cand_rel", a.cand_rel != b.cand_rel), ("target", a.target != b.target)]
                            .iter().filter(|x| x.1).map(|x| x.0).collect();
                        writeln!(out, "MISMATCH {:?} mirror={} played={:?}: {}", what, mirror, played.is_some(), line.trim()).unwrap();
                    }
                }
            }
        }
        writeln!(out, "enccheck: {n} encodings compared, {bad} mismatches").unwrap();
        return;
    }
    if cmd == "bench" {
        let lines: Vec<String> = stdin.lock().lines().map(|l| l.unwrap()).filter(|l| !l.trim().is_empty()).collect();
                // Where does per-leaf CPU time go in self-play? (PROGRESS A7)
                let positions: Vec<Pos> = lines.iter().filter_map(|l| parse_icn_and_play(l).ok()).collect();
                let reps = 3;
                let time = |f: &dyn Fn(&Pos) -> usize| {
                    let t = Instant::now();
                    let mut acc = 0;
                    for _ in 0..reps {
                        for p in &positions {
                            acc += f(p);
                        }
                    }
                    (t.elapsed().as_secs_f64() * 1e6 / (reps * positions.len()) as f64, acc)
                };
                let (t_legal, _) = time(&|p| legal(p).singles.len());
                let (t_cands, n_cands) = time(&|p| candidates(p, &legal(p), &TREE).len());
                let (t_end, _) = time(&|p| game_end(p).is_some() as usize);
                let (t_enc, _) = time(&|p| icore::encode::encode(p, 0, None, false, usize::MAX).n);
                println!(
                    "{} positions, {:.0} candidates on average: legal {:.0} us, legal+candidates {:.0} us, game_end {:.0} us, full encode {:.0} us",
                    positions.len(),
                    n_cands as f64 / (reps * positions.len()) as f64,
                    t_legal, t_cands, t_end, t_enc
                );
                    return;
    }
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let icn = line.trim();
        if icn.is_empty() || icn.starts_with('#') {
            continue;
        }
        match cmd {
            "perft" => {
                let depth: u32 = args.get(2).and_then(|d| d.parse().ok()).unwrap_or(3);
                let pos = parse_icn_and_play(icn).unwrap();
                let counts: Vec<String> = (1..=depth)
                    .map(|d| {
                        let t = Instant::now();
                        let n = perft(&pos, d);
                        format!("D{d}={n} ({} ms)", t.elapsed().as_millis())
                    })
                    .collect();
                writeln!(out, "{}", counts.join(" ")).unwrap();
            }
            "moves" => {
                let pos = parse_icn_and_play(icn).unwrap();
                let set = legal(&pos);
                let check = icore::moves::in_check(&pos, pos.stm);
                writeln!(out, "{{\"moves\":{},\"check\":{check},\"end\":{}}}", moves_json(&set), end_json(&pos)).unwrap();
            }
            "replay" => {
                let (mut pos, moves) = parse_icn(icn).unwrap();
                let mut bad = None;
                for (ply, m) in moves.iter().enumerate() {
                    if game_end(&pos).is_some() || !contains(&legal(&pos), m) {
                        bad = Some(ply);
                        break;
                    }
                    pos = play(&pos, m);
                }
                match bad {
                    Some(ply) => writeln!(out, "{{\"ok\":false,\"ply\":{ply},\"end\":{}}}", end_json(&pos)).unwrap(),
                    None => writeln!(out, "{{\"ok\":true,\"plies\":{},\"end\":{}}}", moves.len(), end_json(&pos)).unwrap(),
                }
            }
            "positions" => {
                let (mut pos, moves) = parse_icn(icn).unwrap();
                writeln!(out, "{}", to_icn(&pos)).unwrap();
                for m in &moves {
                    pos = play(&pos, m);
                    writeln!(out, "{}", to_icn(&pos)).unwrap();
                }
            }
            "random" => {
                let arg = |i: usize, default: u64| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(default);
                let (games, max_plies, seed) = (arg(2, 100), arg(3, 200), arg(4, 1));
                let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                let start = parse_icn_and_play(icn).unwrap();
                for _ in 0..games {
                    let mut pos = start.clone();
                    for _ in 0..max_plies {
                        if game_end(&pos).is_some() {
                            break;
                        }
                        let Some(m) = random_move(&pos, &mut rng) else { break };
                        pos = play(&pos, &m);
                        writeln!(out, "{}", to_icn(&pos)).unwrap();
                    }
                }
            }
            "cover" => {
                // "ICN<TAB>move|move|..." -> how each move relates to our candidate lists.
                let (icn, list) = icn.split_once('\t').expect("cover: ICN<TAB>moves");
                let pos = parse_icn_and_play(icn).unwrap();
                let set = legal(&pos);
                let mut fields = Vec::new();
                for (name, rule) in [("tree", &TREE), ("root", &ROOT)] {
                    let cands = candidates(&pos, &set, rule);
                    let mut missed = Vec::new();
                    let mut counts = [0u32; 5]; // exact, far, gap, missed, illegal
                    for tok in list.split('|').filter(|t| !t.is_empty()) {
                        let m = parse_move(tok).unwrap();
                        let c = classify(&pos, &set, &cands, &m, rule);
                        counts[c] += 1;
                        if c == 3 {
                            missed.push(format!("\"{tok}\""));
                        }
                    }
                    fields.push(format!(
                        "\"{name}\":{{\"exact\":{},\"far\":{},\"gap\":{},\"missed\":{},\"illegal\":{},\"n_candidates\":{},\"missed_moves\":[{}]}}",
                        counts[0], counts[1], counts[2], counts[3], counts[4], cands.len(), missed.join(",")
                    ));
                }
                writeln!(out, "{{{}}}", fields.join(",")).unwrap();
            }
            "checktest" => {
                // gives_check must equal play-and-test for every candidate move.
                let pos = parse_icn_and_play(icn).unwrap();
                if icore::moves::in_check(&pos, 1 - pos.stm) {
                    writeln!(out, "0 0").unwrap(); // illegal position: the side not to move is in check
                    continue;
                }
                let set = legal(&pos);
                let (mut n, mut bad) = (0, 0);
                for m in candidates(&pos, &set, &ROOT) {
                    let slow = icore::moves::in_check(&play(&pos, &m), 1 - pos.stm);
                    n += 1;
                    if icore::moves::gives_check(&pos, &m) != slow {
                        bad += 1;
                        eprintln!("MISMATCH {} {} slow={slow}", icn, icore::pos::move_token(&m, pos.stm));
                    }
                }
                writeln!(out, "{n} {bad}").unwrap();
            }
            _ => {
                eprintln!("usage: icore perft <depth> | moves | replay | positions | random <games> <plies> <seed> | cover");
                std::process::exit(2);
            }
        }
    }
}
