// Python module `icore` (built with maturin, feature "python"). Two functions:
//   replay(start_icn, moves)  -> [(position ICN, times seen before)] for the position before each move
//   encode(icns, reps, moves, mirror, max_cands, threads) -> {name: (bytes, shape)}; wrap with numpy.frombuffer
//   position(icn)    -> (stand-alone ICN of the position after the move list, times seen before)
//   candidates(icn)  -> candidate move tokens, in the same order as encode's candidates (no cap)
use crate::encode::{encode as encode_one, Encoded, CAND_CH, REL_CH, T, TOK_CH};
use crate::mcts::{advance, best, expand_and_backup, new_tree, repetitions, select, Tree};
use crate::moves::{candidates as cand_list, contains, legal, representative, slide_breakpoints, steps_to, TREE};
use crate::pos::{hash, move_token, parse_icn, parse_icn_and_play, parse_move, play, to_icn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

#[pyfunction]
fn replay(start_icn: &str, moves: Vec<String>) -> PyResult<Vec<(String, u32)>> {
    let (mut pos, pre) = parse_icn(start_icn).map_err(err)?;
    for m in &pre {
        pos = play(&pos, m);
    }
    let mut out = Vec::with_capacity(moves.len());
    for tok in &moves {
        let m = parse_move(tok).map_err(err)?;
        if !contains(&legal(&pos), &m) {
            return Err(err(format!("illegal move {tok} at ply {}", out.len())));
        }
        let h = hash(&pos);
        out.push((to_icn(&pos), pos.history.iter().filter(|&&x| x == h).count() as u32));
        pos = play(&pos, &m);
    }
    Ok(out)
}

/// Runs f(0..n) on `threads` threads, keeping the order.
fn parallel_map<R: Send>(n: usize, threads: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    let chunk = n.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|start| {
                let f = &f;
                s.spawn(move || (start..(start + chunk).min(n)).map(f).collect::<Vec<R>>())
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

#[pyfunction]
fn encode<'py>(
    py: Python<'py>,
    icns: Vec<String>,
    reps: Vec<u32>,
    moves: Vec<String>,
    mirror: Vec<bool>,
    max_cands: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let n = icns.len();
    let results: Vec<Result<Encoded, String>> = py.allow_threads(|| {
        parallel_map(n, threads, |i| {
            let pos = parse_icn_and_play(&icns[i])?;
            let played = if moves[i].is_empty() { None } else { Some(parse_move(&moves[i])?) };
            Ok(encode_one(&pos, reps[i], played, mirror[i], max_cands))
        })
    });
    let encoded: Vec<Encoded> = results.into_iter().collect::<Result<_, _>>().map_err(err)?;
    encoded_dict(py, &encoded)
}

/// Encoded positions as {name: (bytes, shape)}; candidates are padded to the longest list.
fn encoded_dict<'py>(py: Python<'py>, encoded: &[Encoded]) -> PyResult<Bound<'py, PyDict>> {
    let n = encoded.len();
    let c = encoded.iter().map(|e| e.cands.len()).max().unwrap_or(1).max(1);

    let mut tokens = Vec::with_capacity(n * T * TOK_CH);
    let mut pairs = Vec::with_capacity(n * T * T * REL_CH);
    let mut cands = vec![0u8; n * c * CAND_CH];
    let mut cand_rel = vec![0u8; n * c * T * REL_CH];
    let mut n_cands = Vec::with_capacity(n * 4);
    let mut target = Vec::with_capacity(n * 4);
    for (i, e) in encoded.iter().enumerate() {
        tokens.extend_from_slice(&e.tokens);
        pairs.extend_from_slice(&e.pairs);
        for (k, (row, block)) in e.cands.iter().zip(&e.cand_rel).enumerate() {
            let at = (i * c + k) * CAND_CH;
            cands[at..at + CAND_CH].copy_from_slice(row);
            let at = (i * c + k) * T * REL_CH;
            cand_rel[at..at + T * REL_CH].copy_from_slice(block);
        }
        n_cands.extend_from_slice(&(e.cands.len() as i32).to_le_bytes());
        target.extend_from_slice(&e.target.to_le_bytes());
    }

    let out = PyDict::new(py);
    let put = |name: &str, data: &[u8], shape: Vec<usize>| out.set_item(name, (PyBytes::new(py, data), shape));
    put("tokens", &tokens, vec![n, T, TOK_CH])?;
    put("pairs", &pairs, vec![n, T, T, REL_CH])?;
    put("cands", &cands, vec![n, c, CAND_CH])?;
    put("cand_rel", &cand_rel, vec![n, c, T, REL_CH])?;
    put("n_cands", &n_cands, vec![n])?;
    put("target", &target, vec![n])?;
    Ok(out)
}

#[pyfunction]
fn position(icn: &str) -> PyResult<(String, u32)> {
    let pos = parse_icn_and_play(icn).map_err(err)?;
    let h = hash(&pos);
    Ok((to_icn(&pos), pos.history.iter().filter(|&&x| x == h).count() as u32))
}

#[pyfunction]
fn candidates(icn: &str) -> PyResult<Vec<String>> {
    let pos = parse_icn_and_play(icn).map_err(err)?;
    let set = legal(&pos);
    Ok(cand_list(&pos, &set, &TREE).iter().map(|m| move_token(m, pos.stm)).collect())
}

/// A search tree. Python loop: d = s.select(64, threads); evaluate d with the net;
/// s.backup(priors, values); ... ; s.best().
#[pyclass(unsendable)]
struct Search {
    tree: Tree,
}

#[pymethods]
impl Search {
    #[new]
    fn new(icn: &str) -> PyResult<Self> {
        Ok(Search { tree: new_tree(parse_icn_and_play(icn).map_err(err)?) })
    }

    /// Up to `n` leaves for the net, encoded like `encode` (may be fewer, or none).
    fn select<'py>(&mut self, py: Python<'py>, n: usize, threads: usize) -> PyResult<(Bound<'py, PyDict>, usize)> {
        let leaves = select(&mut self.tree, n);
        let positions: Vec<_> = leaves.iter().map(|&i| self.tree.nodes[i as usize].pos.clone()).collect();
        let encoded = py.allow_threads(|| {
            parallel_map(positions.len(), threads, |i| encode_one(&positions[i], repetitions(&positions[i]), None, false, usize::MAX))
        });
        Ok((encoded_dict(py, &encoded)?, leaves.len()))
    }

    /// Priors (softmax over each leaf's candidates) and values ([-1, 1], side to move) for the
    /// oldest select whose results are still outstanding.
    fn backup(&mut self, priors: Vec<Vec<f32>>, values: Vec<f32>) {
        expand_and_backup(&mut self.tree, priors, values);
    }

    /// (move token, visits, average value) of the most visited root move.
    fn best(&self) -> Option<(String, u32, f32)> {
        let stm = self.tree.nodes[self.tree.root as usize].pos.stm;
        best(&self.tree).map(|(m, n, q)| (move_token(&m, stm), n, q))
    }

    fn visits(&self) -> u32 {
        self.tree.nodes[self.tree.root as usize].visits
    }

    /// Re-roots onto the position after `moves` (tokens). False: the subtree is unknown.
    fn advance(&mut self, moves: Vec<String>) -> PyResult<bool> {
        let parsed: Vec<_> = moves.iter().map(|t| parse_move(t)).collect::<Result<_, _>>().map_err(err)?;
        Ok(advance(&mut self.tree, &parsed))
    }

    fn set_params(&mut self, c_puct: f32, fpu: f32, contempt: f32) {
        self.tree.c_puct = c_puct;
        self.tree.fpu = fpu;
        self.tree.contempt = contempt;
    }
}

/// For the H6 experiment: Some((kind, representative, random)) when `mv` is a legal slide that is
/// not itself a candidate. kind is "gap" or "far"; representative is the candidate that stands for
/// it; random is a uniformly random legal square on the same ray in a different gap (or region).
#[pyfunction]
fn h6_case(icn: &str, mv: &str, seed: u64) -> PyResult<Option<(String, String, String)>> {
    let pos = parse_icn_and_play(icn).map_err(err)?;
    let m = parse_move(mv).map_err(err)?;
    let set = legal(&pos);
    let cands = cand_list(&pos, &set, &TREE);
    if cands.contains(&m) || !contains(&set, &m) {
        return Ok(None);
    }
    let Some(rep) = representative(&pos, &set, &cands, &m, &TREE) else { return Ok(None) };
    let s = set.slides.iter().find(|s| s.fx == m.fx && s.fy == m.fy && steps_to(s, m.tx as i128, m.ty as i128).is_some()).unwrap();
    let k = steps_to(s, m.tx as i128, m.ty as i128).unwrap();
    let mut bp = slide_breakpoints(&pos, s, &TREE);
    let last = *bp.last().unwrap_or(&0);
    let kind = if k > last { "far" } else { "gap" };
    // Regions: the gaps between breakpoints plus the region beyond the last one (capped at 2^20 steps).
    bp.insert(0, 0);
    let mut regions: Vec<(i128, i128)> = bp.windows(2).filter(|w| w[1] - w[0] >= 2).map(|w| (w[0] + 1, w[1] - 1)).collect();
    if last < s.hi {
        regions.push((last + 1, s.hi.min(last + (1 << 20))));
    }
    let mine = regions.iter().position(|&(a, b)| a <= k && k <= b);
    let others: Vec<(i128, i128)> = regions.iter().enumerate().filter(|(i, _)| Some(*i) != mine).map(|(_, r)| *r).collect();
    if others.is_empty() {
        return Ok(None);
    }
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || { x ^= x >> 12; x ^= x << 25; x ^= x >> 27; x.wrapping_mul(0x2545_F491_4F6C_DD1D) };
    let (a, b) = others[(next() % others.len() as u64) as usize];
    let r = a + (next() as i128).rem_euclid(b - a + 1);
    let to = |k: i128| crate::pos::Move { fx: s.fx, fy: s.fy, tx: (s.fx as i128 + k * s.dx as i128) as i64, ty: (s.fy as i128 + k * s.dy as i128) as i64, promo: 0 };
    Ok(Some((kind.to_string(), move_token(&rep, pos.stm), move_token(&to(r), pos.stm))))
}

/// (result, condition) if the game is over in this position, else None.
#[pyfunction]
fn game_end(icn: &str) -> PyResult<Option<(String, String)>> {
    let pos = parse_icn_and_play(icn).map_err(err)?;
    Ok(crate::rules::game_end(&pos).map(|e| (e.result.to_string(), e.condition.to_string())))
}

#[pymodule]
fn icore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(replay, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(position, m)?)?;
    m.add_function(wrap_pyfunction!(candidates, m)?)?;
    m.add_class::<Search>()?;
    m.add_function(wrap_pyfunction!(h6_case, m)?)?;
    m.add_function(wrap_pyfunction!(game_end, m)?)?;
    Ok(())
}
