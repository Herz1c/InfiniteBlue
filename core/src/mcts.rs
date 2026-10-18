// Monte Carlo tree search with PUCT (PLAN D9), driven in batches from Python:
//   1. `select(n)` walks the tree up to n times and returns the leaves that need the net;
//      each walk adds a "virtual loss" on its path so the next walk tries something else.
//   2. Python evaluates the leaves and calls `backup(priors, values)` for the oldest batch still
//      waiting. Batches are first-in first-out, so the next batch can be selected while the GPU
//      still works on the previous one (pipelining).
//   3. `best()` is the most visited root move. `advance(moves)` re-roots the tree after moves
//      were played, keeping the subtree that was already searched.
// Values are in [-1, 1] from the side to move's point of view (win 1, draw 0, loss -1).
// Game ends (mate, stalemate, repetition, move rule, insufficient material) are exact.

use crate::moves::{candidates, legal, TREE};
use crate::pos::{hash, play, Move, Pos};
use crate::rules::game_end;

const NONE: u32 = u32::MAX;

pub struct Node {
    pub pos: Pos,
    pub moves: Vec<Move>,
    pub prior: Vec<f32>,
    pub child: Vec<u32>,
    pub n: Vec<u32>,
    pub w: Vec<f32>, // sum of values from this node's side to move, per move
    pub vloss: Vec<u32>,
    pub visits: u32,
    pub expanded: bool,
    pub pending: bool, // waiting for the net
    pub terminal: Option<f32>,
    pub draw: bool, // terminal by a draw (scored with contempt, see Tree::contempt)
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub c_puct: f32,
    pub fpu: f32, // unvisited moves start at (parent value - fpu)
    /// A draw counts as -contempt for the side to move at the root (and +contempt for the
    /// opponent), so the engine avoids repetitions against weaker opponents. 0 = neutral.
    pub contempt: f32,
    pub root: u32,
    // Per batch: (leaf, path of (node, move index), needs the net). A walk that ran into a leaf
    // already waiting is kept with needs=false: its virtual loss stays until the batch is backed
    // up, so the following walks spread out instead of repeating the same path.
    pending: std::collections::VecDeque<Vec<(u32, Vec<(u32, usize)>, bool)>>,
}

fn new_node(pos: Pos) -> Node {
    let end = game_end(&pos);
    let draw = end.is_some_and(|e| e.result == "1/2-1/2");
    // A decisive end happened on the opponent's move, so the side to move has lost.
    let terminal = end.map(|_| if draw { 0.0 } else { -1.0 });
    Node {
        pos,
        moves: Vec::new(),
        prior: Vec::new(),
        child: Vec::new(),
        n: Vec::new(),
        w: Vec::new(),
        vloss: Vec::new(),
        visits: 0,
        expanded: false,
        pending: false,
        terminal,
        draw,
    }
}

pub fn new_tree(root: Pos) -> Tree {
    Tree { nodes: vec![new_node(root)], c_puct: 2.0, fpu: 0.3, contempt: 0.0, root: 0, pending: Default::default() }
}

/// Times the position occurred before (the encoder's repetition input).
pub fn repetitions(pos: &Pos) -> u32 {
    let h = hash(pos);
    pos.history.iter().filter(|&&x| x == h).count() as u32
}

fn pick(node: &Node, c_puct: f32, fpu: f32) -> usize {
    let total: u32 = node.n.iter().sum::<u32>() + node.vloss.iter().sum::<u32>();
    let sqrt_total = (total.max(1) as f32).sqrt();
    let parent_q = if node.visits > 0 { node.w.iter().sum::<f32>() / node.visits as f32 } else { 0.0 };
    let mut best = (f32::NEG_INFINITY, 0);
    for i in 0..node.moves.len() {
        let n = node.n[i] + node.vloss[i];
        // A virtual loss counts as a lost visit.
        let q = if n == 0 { parent_q - fpu } else { (node.w[i] - node.vloss[i] as f32) / n as f32 };
        let u = q + c_puct * node.prior[i] * sqrt_total / (1 + n) as f32;
        if u > best.0 {
            best = (u, i);
        }
    }
    best.1
}

fn backup(tree: &mut Tree, path: &[(u32, usize)], leaf_value: f32, virtual_loss: bool) {
    let mut v = leaf_value;
    for &(ni, mi) in path.iter().rev() {
        v = -v; // the parent sees the child's value with the opposite sign
        let node = &mut tree.nodes[ni as usize];
        node.n[mi] += 1;
        node.w[mi] += v;
        node.visits += 1;
        if virtual_loss {
            node.vloss[mi] -= 1;
        }
    }
}

/// Walks the tree up to `n` times. Returns the indices of leaves the net must evaluate.
/// Walks that end in a finished game are backed up immediately.
pub fn select(tree: &mut Tree, n: usize) -> Vec<u32> {
    let mut leaves = Vec::new();
    let mut batch = Vec::new();
    let root = tree.root;
    for _ in 0..n {
        let mut path: Vec<(u32, usize)> = Vec::new();
        let mut cur = root;
        loop {
            let node = &tree.nodes[cur as usize];
            if let Some(mut v) = node.terminal {
                if node.draw {
                    let root_side = node.pos.stm == tree.nodes[root as usize].pos.stm;
                    v = if root_side { -tree.contempt } else { tree.contempt };
                }
                backup(tree, &path, v, false);
                // undo the virtual losses added on the way down (backup did not remove them)
                for &(ni, mi) in &path {
                    tree.nodes[ni as usize].vloss[mi] -= 1;
                }
                break;
            }
            if !node.expanded {
                if node.pending {
                    batch.push((cur, path, false)); // collision: keep the virtual loss for now
                } else {
                    tree.nodes[cur as usize].pending = true;
                    batch.push((cur, path, true));
                    leaves.push(cur);
                }
                break;
            }
            let mi = pick(node, tree.c_puct, tree.fpu);
            tree.nodes[cur as usize].vloss[mi] += 1;
            path.push((cur, mi));
            let next = tree.nodes[cur as usize].child[mi];
            cur = if next == NONE {
                let pos = play(&tree.nodes[cur as usize].pos, &tree.nodes[cur as usize].moves[mi]);
                tree.nodes.push(new_node(pos));
                let id = (tree.nodes.len() - 1) as u32;
                tree.nodes[cur as usize].child[mi] = id;
                id
            } else {
                next
            };
        }
    }
    tree.pending.push_back(batch); // always one entry per select, so backups pair up in order
    leaves
}

/// Candidate moves of a node, in the encoder's order (the priors refer to this list).
pub fn node_moves(pos: &Pos) -> Vec<Move> {
    candidates(pos, &legal(pos), &TREE)
}

/// Gives the leaves of the oldest waiting batch (in `select`'s order) their priors and values,
/// then backs up.
pub fn expand_and_backup(tree: &mut Tree, priors: Vec<Vec<f32>>, values: Vec<f32>) {
    let pending = tree.pending.pop_front().unwrap_or_default();
    let mut evals = priors.into_iter().zip(values);
    for (leaf, path, needs) in pending {
        if !needs {
            for &(ni, mi) in &path {
                tree.nodes[ni as usize].vloss[mi] -= 1;
            }
            continue;
        }
        let Some((prior, value)) = evals.next() else { break };
        let node = &mut tree.nodes[leaf as usize];
        node.moves = node_moves(&node.pos);
        let k = node.moves.len();
        node.prior = prior.into_iter().take(k).collect();
        node.prior.resize(k, 0.0);
        node.child = vec![NONE; k];
        node.n = vec![0; k];
        node.w = vec![0.0; k];
        node.vloss = vec![0; k];
        node.expanded = true;
        node.pending = false;
        if k == 0 {
            node.terminal = Some(0.0); // cannot happen: no moves is mate or stalemate, caught earlier
        }
        backup(tree, &path, value, true);
    }
}

/// Moves the root along `moves` (just played). False when the tree does not contain them, or
/// when it has grown too large; then the caller starts a fresh tree.
pub fn advance(tree: &mut Tree, moves: &[Move]) -> bool {
    if !tree.pending.is_empty() || tree.nodes.len() > 4_000_000 {
        return false;
    }
    let mut r = tree.root;
    for m in moves {
        let node = &tree.nodes[r as usize];
        if !node.expanded {
            return false;
        }
        let Some(i) = node.moves.iter().position(|x| x == m) else { return false };
        if node.child[i] == NONE {
            return false;
        }
        r = node.child[i];
    }
    tree.root = r;
    true
}

/// The most visited root move, its visit count, and its average value for the side to move.
pub fn best(tree: &Tree) -> Option<(Move, u32, f32)> {
    let root = &tree.nodes[tree.root as usize];
    let i = (0..root.moves.len()).max_by_key(|&i| root.n[i])?;
    let q = if root.n[i] > 0 { root.w[i] / root.n[i] as f32 } else { 0.0 };
    Some((root.moves[i], root.n[i], q))
}
