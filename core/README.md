# core: rules and move generation (Rust)

About 1,070 lines, no dependencies. Built and tested in WSL.

| file | what it does |
|---|---|
| `src/pos.rs` | `Pos` (a flat list of ≤ 32 pieces plus side to move, en passant, clock, border, repetition history), ICN parsing and printing, hashing, `play` (make a move) |
| `src/moves.rs` | exact legal moves (`legal`): single moves plus one step interval per slider direction; pins and checks; `perft`; the candidate rule (`candidates`) |
| `src/rules.rs` | game end (`game_end`) in the site's order: repetition, mate/stalemate, move rule, insufficient material |
| `src/main.rs` | `icore` test CLI: `perft`, `moves`, `replay`, `positions`, `random`, `cover` |
| `tests/perft8x8.rs` | 8×8-border perft against the standard chess numbers |

## Reading Rust here (just enough)

- `struct` is plain data; `fn` is a free function. There are no classes or methods with hidden state.
- `&Pos` means "read-only view of a Pos"; `play(&pos, &m)` returns a new position and leaves the old one untouched.
- `Option<T>` is "a T or nothing" (`Some(x)` / `None`).
- `Vec<T>` is a growable array.
- `i128` is used for every coordinate difference, because two pieces can be ~2^64 squares apart.

## Commands (in WSL, from the repo root)

```bash
cd core && CARGO_TARGET_DIR=~/build/icore cargo build --release
```

```bash
cd core && CARGO_TARGET_DIR=~/build/icore cargo test --release
```

```bash
bash tools/m1_positions.sh && for f in probes random games probe_random; do python3 tools/compare_moves.py data/m1/$f.icn; done
```

```bash
python3 tools/compare_ends.py data/m1/games_full.icn
```

```bash
python3 tools/coverage.py data/m1/games_full.icn ref/perft/positions.icn
```

The build directory sits in WSL's own filesystem (`~/build/icore`) because building on `/mnt/c` is slow.
