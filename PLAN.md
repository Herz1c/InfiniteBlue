# PLAN: an infinite-chess engine (Classical) that beats Apeiron

Research date 2026-10-17. Nothing implemented yet.

**Goal (in priority order):**
1. An engine **no human can beat** at 5 s per move, with the highest possible quality of play.
2. Beat the **latest** Apeiron.
3. Produce new research findings along the way (§9).

### Decisions (2026-10-17)

| topic | decision |
|---|---|
| Engine | Our own. Apeiron is not cloned. Apeiron is used only as an opponent and, in M2, as a label source. |
| Where it runs | Locally (GPU + CPU), never in the browser. Net size is limited only by evaluations per second. |
| Platform | WSL2 Ubuntu for everything that spawns engines. |
| Time control | **5 s per move** for milestone confirmations. Development SPRTs run at 1 s for speed. |
| Opponent | Latest Apeiron. Frozen at v7.10.0 for development; the newest release is re-tested at the end (§5). |
| Success | Measured both against Apeiron and for robustness against humans and adversaries (M5, §9 R4). |
| Who writes code | Claude writes all code (Rust and Python), kept flat and commented; you steer the research and review. The `[YOU]` marks in §3 are historical. |

Sources were read locally (all under `ref/`, which is gitignored and never imported by our code):

- `ref/apeiron`: FirePlank/infinite-chess-engine at `b928c46` (HEAD, 2026-10-15). The latest release tag, **v7.10.0**, is `4e1aba5` (2026-10-11).
- `ref/apeiron-v7.10.0`: a worktree of that tag. Built in WSL from `~/src/apeiron-v7.10.0`; the frozen opponent binary is `~/engines/apeiron-v7.10.0/sprt` (see `PROGRESS.md`).
- `ref/site`: Infinite-Chess/infinitechess.org at `67b2691` (2026-10-14).

Line references point at those commits. Paths with `apeiron:` or `site:` are relative to `ref/apeiron/` and `ref/site/`.

---

## 0. Verdict up front

**What holds.** The direction is right: one variant, our own movegen, a transformer over piece tokens with pairwise attention bias, a pointer policy, distillation then RL, Rust plus PyTorch. The closest proven analogue is Lc0, which uses a transformer over squares, learned attention bias ("smolgen"/GAB) and a query·key policy head. Its transformer BT4 is about 270 Elo stronger in raw policy than its best CNN ([lczero blog](https://lczero.org/blog/2024/02/transformer-progress/); [Chessformer, arXiv 2409.12272](https://arxiv.org/abs/2409.12272); [ICLR 2026 version, arXiv 2605.19091](https://arxiv.org/abs/2605.19091)).

**What I would change, and why:**

1. **Variant name.** On the site, "Chess on an Infinite Plane" is **CoaIP**: guards, chancellors, hawks and extra pawns (`site:src/shared/chess/variants/variantdictionary.ts:186-189`). "Classical" is the normal 8×8 army on an unbounded board (`:136-139`). This plan assumes **Classical**.
2. **M1 "perft identical to Apeiron" measures the wrong thing.** Apeiron's perft counts its own *heuristic candidate list*, not legal moves; an open ray has about 9·10^18 legal destinations. The list depends on:
   - wiggle constants;
   - check squares against the *enemy* king;
   - a "far escape" square on an **absolute** shell at |coord| ≈ 4063, which comes from its 13-bit TT move encoding (`apeiron:src/moves.rs:2844-2851`, `src/search/tt_defs.rs:276-278`).

   Matching it exactly would mean re-implementing their GPL heuristic. The replacement criterion (§4, M1) is:
   - exact legal sets equal to the site's own generator;
   - perft inside a bounded 8×8 world equal to standard chess numbers;
   - coverage of every move Apeiron actually plays.

   I still ran Apeiron's perft to depth 4 on 24 positions (§1.7) and keep it as a diff and debug tool. Our counts are expected to differ by design.
3. **No set-transformer machinery.** With ≤32 pieces, full self-attention is free. ISAB/PMA exist for large sets ([Set Transformer, arXiv 1810.00825](https://arxiv.org/abs/1810.00825)). Use a plain encoder plus additive pairwise bias. Graphormer found edge features work best as attention bias ([arXiv 2106.05234](https://arxiv.org/abs/2106.05234)), and Chessformer found geometric bias adds more than 100 Elo inside Lc0.
4. **Replace egocentric crops with king-zone tokens.** Use the same transformer and add one token per square around each king, carrying exact attack counts. This gives the same information as a crop without a second (CNN) branch.
5. **Board = a flat array of ≤32 pieces.** Apeiron's sparse tiles and line indices exist for 200-piece fairy positions and a search running at 0.3–0.4M nodes/s per thread. Our search does about 10^4 net evaluations/s, so movegen speed is irrelevant and the simplest code wins.
6. **Run Apeiron, and probably develop, in WSL2/Linux.**
   - The Windows GNU toolchain cannot build it: `dlltool` needs an assembler that isn't shipped.
   - There is no MSVC on this machine.
   - Apeiron's own notes say Windows antivirus scanning per process spawn causes timeout losses (`apeiron:.claude/skills/sprt-testing/SKILL.md:51-62`).
7. **Target version.** You wrote v7.1.0. The latest release is **v7.10.0**, nine minor releases later (each minor release is ≥ +30 nominal SPRT Elo, `apeiron:CHANGELOG.md:9-15`). Freeze v7.10.0 for M4 and keep v7.1.0 as an intermediate rung.
8. **The real risk is compute, not architecture.** At 400k nodes per move (1–2 s on one thread), Apeiron reaches **median depth 24** (measured, §0.1). Our MCTS will get about 10^4 net evaluations per move, so it can only win through a much better evaluation and policy. That is the AlphaZero bet. It is plausible here because:
   - root branching is 70–100+ moves (§1.7), which is hard for alpha-beta;
   - Apeiron's own roadmap says its evaluation lacks infinite-board "smart metrics", such as long-range attacks (`apeiron:docs/ROADMAP.md:33-36`).

   Kill and pivot criteria are in §7.

### 0.1 Measured on this machine (2026-10-17)

| item | value |
|---|---|
| GPU | RTX 5060 Ti 16 GB, driver 616.92. bf16 matmul 48.7 TFLOPS when idle (measured in WSL). The first Windows runs were slowed by another app using about 75% of the GPU. |
| CPU / RAM / disk | Core Ultra 7 265KF, 20 cores; 31.6 GB RAM; 548 GB free |
| Software present | Python 3.13 with PyTorch 2.14+cu130 (CUDA works), Node 24, WSL2 Ubuntu 24.04 with gcc |
| Software I installed | rustup in user scope: Windows GNU host (`~/.cargo`) and in WSL. `nightly-2025-11-15` comes from Apeiron's `rust-toolchain.toml`. |
| Net throughput (idle GPU, WSL, bf16, 40 tokens, batch 256, `nn.TransformerEncoder` with float bias mask, no CUDA graphs) | 0.8M params: about 133k pos/s · 4.2M params: about 50k pos/s · 11.8M params: about 25k pos/s · 25M params: about 14k pos/s |
| Apeiron `data_gen`, Classical, 18 threads | depth 9: **1.6M positions/h** (6.8k nodes/move) · depth 12: **0.58M positions/h** (17.7k nodes/move) |
| Apeiron, 400k nodes per move | depth median 24, p10 19, p90 31 |
| Apeiron Classical self-play | 82% draws at depth 9, 75% at depth 12; mostly threefold repetition (44/72 and 40/72 games) |

Scripts: `ref/tools/gpu_bench.py` (rerun with the GPU idle), `ref/perft/dg_bench.sh`, `ref/perft/positions.icn`, `ref/apeiron-v7.10.0/examples/perft_icn.rs`.

Compute verdict: local hardware is enough through M3. M4 is feasible locally in weeks; a cloud burst shortens it (§4, M4).

---

## 1. Findings

### 1.1 Rules of Classical

The site is the source of truth; Apeiron agrees with it everywhere I checked.

**Start position.**
- ICN: `w 0/100 1 (8|1) P1,2+|P2,2+|…|K5,1+|k5,8+` (`apeiron:src/lib.rs:60-61`, `site:…/variantdictionary.ts:96-97`). `+` marks a special right (pawn double step, castling).
- **Nothing stops pieces behind their own back rank.** Apeiron's root list for the start position has **73** moves, not chess's 20.
- At full strength with 1 s, Apeiron's first move was `4,1 → 9,-4`: the queen goes out behind its own pawns (verified with `sprt serve`).

**World border.**
- Human Classical games have none: there is no `worldBorderDist` (`variantdictionary.ts:136-146`).
- Engine games on the site get a border at distance `I64_MAX − 2000` around the start box (`site:src/client/scripts/esm/game/chess/engines/engine.ts:49`, `site:src/shared/chess/logic/gamefile.ts:219-233`). That gives x ∈ [1 − (2^63 − 2001), 8 + (2^63 − 2001)], and the same for y.
- Apeiron defaults to ±(i64::MAX − 1000) (`apeiron:src/moves.rs:78`).
- So **i64 coordinates suffice, but every difference must be computed in i128 or with saturation** (see the overflow traps in §1.4).

**Promotion.**
- White promotes on rank 8, Black on rank 1, to N, B, R or Q (`variantdictionary.ts:105-106,139`; `apeiron:src/moves.rs:4617-4660`).

**Pawns.**
- Single push. Double push only with the special right and both squares empty (`moves.rs:4688-4702`).
- En passant targets the skipped square and is available only on the next ply (`site:src/shared/chess/logic/specialdetect.ts:265-311`).

**Castling.**
- The king needs its special right and moves 2 squares toward the *nearest* piece on its rank in that direction.
- That piece must be friendly, not a pawn, not royal, still hold its special right, and stand ≥ 3 squares away. It may be arbitrarily far.
- The king may not be in check and may not pass an attacked square. The landing square is checked like any king move. The partner lands on the square the king passed.
- Sources: `specialdetect.ts:45-170`, `apeiron:src/moves.rs:4743-4805`.

**Move rule (50-move rule).**
- `0/100` means 100 plies. The counter resets on a capture or a pawn move (`site:…/movepiece.ts:376-386`).
- The game is drawn when the counter reaches 100, but checkmate on that same move still wins. Checkmate is tested before the move rule (`site:…/wincondition.ts:38-51,163-169`).

**Repetition.**
- Threefold.
- The history scan stops at any pawn move, capture or loss of a special right. Positions that differ only in en-passant state count as different (`site:…/repetition.ts:24-143`).
- Apeiron's equivalent is a repetition window (`apeiron:src/game.rs:1370-1381`).

**Stalemate** is a draw (`site:…/checkmate.ts:56-63`).

**Insufficient material.**
- The site uses a table in which each listed set is *maximal*: any subset of a listed set is also a draw (`site:…/insufficientmaterial.ts:43-90`).
- **K+Q vs K and K+R vs K are immediate draws on the unbounded board**, because both are subsets of K+Q vs K+q and K+R vs K+r.
- K+R+R vs K is not a draw. The value net must learn all of this.

**Game-end check order:** repetition → checkmate/stalemate → move rule → insufficient material (`wincondition.ts:38-51`).

### 1.2 ICN

Site parser: `site:src/shared/chess/logic/icn/icnconverter.ts:353-454`.

```
[Meta "v"]… <turn> [<ep x,y>] <clock>/<limit> <fullmove> (<ranksW>[;types]|<ranksB>[;types]) [<l,r,b,t> (_ = ∞)] [winconditions] <pieces> [<moves>]
```

- Piece entry: `<abbr><x>,<y>[+]`, uppercase for White, entries separated by `|`.
- Moves: `x,y>x,y[=Q]`, separated by `|` (compact form, `icnconverter.ts:314-319`).
- Apeiron also replays a move list appended to the ICN (`apeiron:src/game.rs:4036-4062`).
- Apeiron's `serve` returns moves as `"x,y x,y[ q]"` (`apeiron:src/bin/sprt.rs:973-979`).

### 1.3 How Apeiron picks slider candidates

Source: `apeiron:src/moves.rs:3311-3949`.

For each ray, the legal set is distances 1..max_dist, up to the first blocker (captured if it is an enemy) or the border. From it, Apeiron emits a finite list:

1. **Short ray or small board.** If the blocker is within 8 squares, or the world is ≤ 32×32, every square is emitted (`:3364-3367, :3440-3452`).
2. **Otherwise, the union of these distances:**
   - (a) Pieces on the ray's own line, with a "wiggle" around each: enemy ±2, friend ±1. Beyond 10 squares the wiggle drops to enemy ±1 and friend 0; royals always get the full wiggle (`:3323-3338`).
   - (b) Cross-ray squares: where the ray crosses a row, column or diagonal through another piece that is the nearest piece on that line. Orthogonal crossings get the wiggle; diagonal crossings are exact (`find_cross_ray_targets_into`, `:2961-3247`).
   - (c) Check squares against the enemy king (`:3719-3782`).
   - (d) Wall squares beside a bare enemy king, only in lone-king conversion mode (`:3784-3812`).
   - (e) Always d = 1 and d = 2, plus the capture square (`:3817-3825`).
3. **Added outside the cache:**
   - a capture beyond i64 distance (`:3898-3909`);
   - **one far escape per fully open ray**, to the absolute shell at `MAX_TT_COORD − 32 = 4063`, when there are ≥ 50 squares of room (`:2844-2851, :3911-3920`);
   - fresh check squares for the king's current square (`:3924-3947`).
4. **Interior search uses a cache** keyed by (square, piece, direction) that is **deliberately never invalidated** (`apeiron:AGENTS.md`, "Architecture gotchas").
   - Root, perft and API calls bypass it (`moves.rs:24`, `game.rs:1727-1734`).
   - At shallow depth, a per-ray quiet cap keeps only the N nearest squares within 1024 (`moves.rs:52-58, 3879-3891`).

Example from the dump of position 15 (KQ vs K): the queen's 68 candidates are the 8-neighbourhood, squares on the enemy king's lines ±2, crossings with the king's file and rank, and far-shell squares such as `4063,-12` and `-4064,-12`.

### 1.4 Movegen traps from Apeiron's changelog

Each of these becomes a test for our movegen.

| trap | ref (`apeiron:CHANGELOG.md`) |
|---|---|
| Stale candidate cache: in 84% of 2401 positions, root lists both lost legal moves and gained impossible ones | :686 |
| Cache keyed by square and direction only, so a black queen reused a white rook's list | :105 |
| Cached check squares from where the king *used to be* | :106 |
| Startpos perft D3 gave 8842 instead of 8902 (stale cache used in exact lists) | :785 |
| Public legal-move API filtered the cached, incomplete list | :418 |
| Check from more than 64 squares away: slider blocks were missing, so a blockable check read as mate | :34 |
| Square beside a bare king was never generated past the 16-square filter, so the mating wall was impossible | :457 |
| Rook could not see a winning attack 23 squares away | :716 |
| Sliders could not make the long run to the far shell, hiding escapes | :569 |
| Overflow: a play border defaulting to ±1e15 hid a wrap in `ray_border_distance` (126 legal moves became 92) | :372 |
| Overflow: diagonal cross product gave a false zero past 2^31 | :421 |
| Overflow: i64→i32 in tropism pushed scores past the mate score | :757 |
| Cache flag stored on bit 48: captures more than 2^48 away landed short | :65 |
| Distance caps removed (a bishop 4000 squares away pinning a rook), then a 1024 cap re-added for shallow quiets | :23, commit `6365f8e` |
| Castling: partner ≥ 3 squares away not enforced, so the king could land on an occupied square | :760 |
| Castling validation demanded an empty third square, which is the rook itself | :459 |
| Bishop square-colour parity missing from the material hash, so insufficient-material verdicts were cached wrongly | :810 |
| ICN parse inherited the previous position's border and rules | :420, :669 |
| Repetition counted across a lost special right, or onto the en-passant position after a clock reset | :25 |

Lessons:
- An incomplete *root* list is fatal; an incomplete *interior* list is tolerable (`AGENTS.md`, rejected root pruning: −26.5 Elo).
- Compute every distance in i128.
- Keep exact legality strictly separate from candidate selection.

### 1.5 What Apeiron tried that failed or was reverted

- **InfNNUE-v1**, a full NNUE replacement: 6.9M parameters, an absolute eval target, one-hot P/N/B/R/Q/K pieces only. It hit a data wall and was removed (`apeiron:docs/hybrid-eval-design.md:22-42`; `CHANGELOG.md:311`).
- **Pawn-king net (Stage B):** 0.00% gain on the holdout. A joint version with pawn cells memorized its training data (`docs/eval-net-plan.md`, log 2026-09-22/23).
- **Sparse piece-king embeddings:** piece placements from archive games did not transfer to on-policy games. With WDL-mixed targets the net learned to identify the game and its result (−25% on the holdout). Their rule: **high-cardinality positional inputs need eval-only or relabelled targets and on-policy data** (same log).
- **Distilling a larger net into a smaller one** added nothing. Label quality beat volume: fixed-depth relabels beat archive labels (same log).
- **Search ideas that lost:** pruning root quiets by rank, gentle aspiration windows, wider history tables, and removing the stale cache (`apeiron:AGENTS.md`, "Already tried and rejected").
- **What it means for us.** Our net is bigger than InfNNUE, so data is the core risk.
  - Distillation labels from Apeiron are cheap: about 0.58M positions/h at depth 12.
  - Value targets come from search scores (not raw WDL) while distilling.
  - Data is moved on-policy (our own games) as soon as M3.

### 1.6 Running Apeiron as an opponent

**Build** (WSL2): `cargo build --release --features sprt --bin sprt` (`apeiron:sprt/README.md:24-26`). Done here with `tools/apeiron.sh`; the frozen binary is `~/engines/apeiron-v7.10.0/sprt` in WSL.

**Protocol: `sprt serve`.** One JSON request per line on stdin, one JSON line back on stdout (`apeiron:src/bin/sprt.rs:170-173, 366-395, 4085-4155`).
- Request: `{icn, variant, wtime, btime, winc, binc, fixed_time, max_depth, noise_amp, seed, strength}`.
- Response: `{bestmove:"x,y x,y[ q]", score, elapsed_ms, panic}`.
- The process stays warm across a game (TT and history persist). `bestmove` is null in a terminal position. Verified working.

**UCI** exists only for 8×8 chess (`apeiron:src/bin/README.md:75-81`), so it is useless here.

**Strength levels 1–8.**
- Level 8 is full strength (`apeiron:src/search/skill.rs:5`).
- Levels 1–7 sample from MultiPV with depth caps 2/3/4/5/7/9/13, a best-move chance from 0.09 to 0.68, eval distortion and "reply blindness" (`skill.rs:47-180`).
- The site's UI currently exposes only levels 1–3 (`site:…/engines/engine.ts:52`). **Apeiron level N is not the same as site UI level N.**

**Time and threads.**
- `fixed_time` gives milliseconds per move; clocks are also supported.
- Natively the engine runs on 1 thread unless `APEIRON_THREADS=N` is set (Lazy SMP; `apeiron:src/search.rs:2715-2729`).
- Trap: `max_depth` with time 0 does not search deep (`SKILL.md:40-45`).

**Training data.**
- `data_gen --variants Classical --depth D [--nodes N] --keep-noisy` writes one JSONL record per game.
- Each record has the start ICN, the moves, and per position: static eval, search score, depth, nodes, best move, plus the game result (`apeiron:src/bin/data_gen.rs:190-362`).
- Openings are randomized: 12 plies of MultiPV-5 picks within 120 cp.

**Their SPRT.**
- Pentanomial GSPRT matching fastchess/Fishtest, with normalized or logistic Elo (`sprt.rs:617-640, 882-912`).
- α = β = 0.05, so the LLR bounds are ±2.94. Bounds are [0, 5] for gainers and [−10, 0] for non-regression (`sprt/README.md:80-89`, `SKILL.md:105-118`).
- Openings: each pair shares 8 plies searched at depth 7 with noise, then the game is replayed with colours swapped (`README.md:69-70`).
- Default cap is 300 plies with 1000 cp adjudication. That is **too short for Classical**: the median game is 224 plies (`CHANGELOG.md:508`).

### 1.7 Apeiron perft reference (v7.10.0, local harness `examples/perft_icn.rs`)

Positions are in `ref/perft/positions.icn`; raw output is in `ref/perft/d4.txt`.
- Apeiron's `perft()` counted the same as a perft filtered by make/`is_move_illegal`/undo in all 96 cases, so its root list is already legal.
- These counts are **candidate counts**, not legal-move counts.
- Perft ignores game end, so positions 15 and 16 are insufficient-material draws on the site.

| # | position | D1 | D2 | D3 | D4 | D4 time |
|---|---|---:|---:|---:|---:|---:|
| 01 | start | 73 | 5,442 | 495,660 | 47,245,434 | 9.9 s |
| 02 | 1.e4 e5 | 90 | 8,449 | 928,280 | 108,662,230 | 24.5 s |
| 03 | Ruy Lopez | 86 | 7,098 | 747,018 | 76,784,537 | 27.0 s |
| 04 | QGD | 83 | 7,911 | 782,366 | 91,987,708 | 24.2 s |
| 05 | en passant available (e5 f5) | 92 | 7,174 | 786,303 | 77,068,089 | 19.7 s |
| 06 | kingside castling ready for both | 75 | 5,744 | 536,807 | 52,482,715 | 18.4 s |
| 07 | early Qh5 (open queen rays) | 87 | 9,052 | 962,977 | 119,104,809 | 33.8 s |
| 08 | Nimzo with castled black king | 82 | 6,026 | 609,959 | 58,080,360 | 13.8 s |
| 09 | bare castling, all four rooks, open board | 41 | 1,578 | 66,882 | 2,887,543 | 1.3 s |
| 10 | far pin on a file (queen ~1e6 away) | 26 | 1,656 | 61,096 | 4,798,080 | 1.3 s |
| 11 | check from a bishop 1000 squares away (blocks) | 8 | 184 | 28,872 | 618,556 | 0.2 s |
| 12 | double check (far rook + far bishop) | 4 | 266 | 15,142 | 908,629 | 0.2 s |
| 13 | promotion by push and capture | 16 | 389 | 6,821 | 174,180 | 0.0 s |
| 14 | black en passant + black promotion | 14 | 125 | 2,373 | 20,363 | 0.0 s |
| 15 | KQ vs K far apart (site: draw) | 76 | 552 | 39,432 | 276,280 | 0.2 s |
| 16 | KRR vs K | 76 | 576 | 41,152 | 301,312 | 0.1 s |
| 17 | scattered pieces, long distances | 7 | 2,524 | 777,987 | 258,784,474 | 65.0 s |
| 18 | pawns outside the 8×8 box | 52 | 555 | 27,238 | 279,700 | 0.1 s |
| 19 | king at 1,1 is not a corner | 3 | 118 | 622 | 30,371 | 0.0 s |
| 20 | knights | 29 | 626 | 17,353 | 391,392 | 0.1 s |
| 21 | rooks with several blockers on one rank | 81 | 5,642 | 485,879 | 38,347,122 | 7.6 s |
| 22 | knight check while a bishop is pinned | 6 | 132 | 6,910 | 588,853 | 0.1 s |
| 23 | castling through an attacked square | 49 | 1,685 | 88,037 | 2,872,415 | 0.7 s |
| 24 | discovered check, far queen behind own knight | 8 | 355 | 38,406 | 2,083,652 | 0.5 s |

---

## 2. Architecture decisions

**D1 · Board representation**
- Options: (a) sparse 8×8 tiles plus per-line sorted indices, like Apeiron and the site; (b) a hash map from coordinate to piece; (c) a flat array of ≤ 32 pieces with linear scans.
- **Choice: (c).** `struct Pos { pcs: [Pc; 32], n: u8, stm, ep, clock, rights in Pc.flags, hash }`.
- Rationale: Classical never has more than 32 pieces. Every "nearest blocker on this ray" query is ≤ 31 comparisons, with no index to keep in sync. Movegen will be at least 100× faster than the net, so speed is irrelevant.
- Risk: none for Classical. It does not scale to 200-piece fairy positions, which are out of scope.

**D2 · Coordinates and overflow**
- **Choice:** store i64; compute every difference, distance and cross product in **i128**.
- Enforce the site's engine border (`2^63 − 2001` around the start box) as a hard limit on interval ends.
- Never feed absolute coordinates to the net (translation invariance).
- Risk: low, provided the overflow traps of §1.4 become fuzz tests (positions near ±2^62).

**D3 · Exact move representation**
- Options: enumerate squares (impossible); exact *intervals*; candidates only (Apeiron).
- **Choice:** exact legal set = individual moves (K, N, P, castling, en passant, promotions) plus **one interval [1, L] per slider direction**.
- Legality: a pin restricts a slider to the pin line. In check, intervals are intersected with the block and capture squares; with a double check, only king moves remain.
- Rationale: it is the site's own representation (`SlideLimits`, `site:src/shared/chess/logic/legalmoves.ts:31-53`), so it can be compared 1:1 against the site.
- Risk: the edge cases are pins plus en passant on one rank, and castling through attacked squares. Covered by M1 tests.

**D4 · Candidate selection** (finite moves the search may play)
- Options: (a) copy Apeiron's rule; (b) our own rule over the exact intervals; (c) sampled actions (Sampled MuZero, [arXiv 2104.06303](https://arxiv.org/abs/2104.06303)).
- **Choice: (b)**, designed and written by you. Per ray, take the union of:
  - all of d ≤ 8;
  - the capture square;
  - every crossing of the ray with a row, column or diagonal through any other piece, plus that crossing ±1;
  - check squares and the squares adjacent to the enemy king's lines;
  - the square just short of the blocker;
  - for an open ray, **relative** far representatives: last breakpoint + 4, and +2^16, all clipped to the border.

  The root uses a wider version (d ≤ 16, crossings ±2). Pawn, knight and king moves are always complete.
- Rationale: translation-invariant and independent of any TT encoding, and the policy head absorbs width cheaply.
- Risk: a long-range quiet move outside the set can never be played. Mitigation: the coverage metric against Apeiron's played moves (M1-d), plus mining every miss into a regression test.
- Research alternative: R1 (§9) replaces this heuristic with a complete, segment-based move abstraction. D4 is the baseline it has to beat.

**D5 · Hashing and repetition**
- **Choice:** Zobrist over (x, y, piece, special right) using `splitmix64(x ^ rot(y) ^ piece)`, because coordinates are unbounded and no key table can cover them. Plus a side key and an en-passant key.
- The repetition window follows the site's rule: it resets on a pawn move, a capture or a lost special right. Draw detection, both on the board and inside the tree, uses the hash stack within that window.
- Risk: collisions, which a full-position compare on twofold catches.

**D6 · Net input**
- Options: (a) dense crops; (b) piece tokens only; (c) piece tokens + relational bias + king-zone tokens.
- **Choice: (c).**
  - **Tokens.** Pieces (≤ 32), the 8 squares around each king (16 tokens), and one global token, so ≤ 49 in total.
  - **Piece features:** type, own/enemy, special right, en-passant-capturable, log-bucketed distance to each side's promotion rank, attacked-by-enemy and defended-by-own counts, pinned, gives check, log-bucketed mobility per direction.
  - **King-zone features:** offset, occupied, attack counts for each side.
  - **Global features:** move-rule bucket, repetition count.
  - **Pair relation (i, j):** signed log2 buckets of dx and dy; line type (row, column, diagonal, anti-diagonal, none); "j is the first piece seen from i along that line"; number of pieces between (0, 1, 2+); i attacks j; j attacks i; knight offset; king adjacency. Embedded and projected to an additive **per-head attention bias**, as in Graphormer and Chessformer's GAB/smolgen.
  - **Canonicalization:** always from the side to move (flip y → 9 − y and swap colours, which exactly maps promotion ranks 8 ↔ 1). Random x-mirroring is an exact symmetry of the rules (castling is symmetric), so it doubles the data.
- Rationale: rules-derived tactical facts (attacks, pins) are cheap and exact. Apeiron's experience says scalar facts transfer well, while raw placements need on-policy data.
- Risk: the relation vocabulary misses something. Mitigation: ablations on held-out policy accuracy.

**D7 · Policy head**
- Options: a fixed move index (impossible); a pointer over candidates ([Pointer Networks, arXiv 1506.03134](https://arxiv.org/abs/1506.03134)).
- **Choice:** one **query token per candidate move**. Its features are the moving piece's encoder state, direction, log-distance, capture or promotion type, gives check, attack and defence counts on the destination, and relations from the destination square to every piece (same vocabulary as D6).
- Two cross-attention layers run from move queries to encoder tokens with that bias, then an MLP produces a logit; the softmax is over candidates only.
- This generalizes Lc0's attention policy, where the logit is query(from)·key(to) over 64×64 ([arXiv 2409.12272](https://arxiv.org/abs/2409.12272), App. A-C): here the destination is not a token, so it gets a query.
- Risk: cost grows with the number of candidates. With about 150 candidates × 49 tokens it is small.

**D8 · Value head and targets**
- **Choice:** WDL softmax plus a moves-left scalar from the global token.
- In distillation: the main value target is Apeiron's search score mapped through a logistic (K fitted on the data). The WDL loss is small and positions are subsampled per game, because Apeiron saw nets fingerprint game results.
- Auxiliaries during RL: short-term value, opponent reply policy and a soft policy (KataGo; [arXiv 1902.10565](https://arxiv.org/abs/1902.10565); [KataGoMethods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)).
- Risk: insufficient-material draws (K+Q vs K) must be terminal in our rules, so the net never has to learn them.

**D9 · Search**
- Options: alpha-beta with a net eval (Apeiron's home turf, two years of SPRT tuning); PUCT MCTS (Lc0); Gumbel MCTS ([Danihelka et al., ICLR 2022](https://iclr.cc/virtual/2022/spotlight/6419)).
- **Choice:**
  - PUCT for matches: subtree reuse, FPU, root policy temperature, virtual loss for batching, exact terminals, twofold-in-tree counted as a draw.
  - **Gumbel root with sequential halving** for self-play, because it guarantees policy improvement even with very few simulations.
  - No alpha-beta.
- Rationale: the policy prunes a 70–100+ root branching factor, which hurts alpha-beta most; it is our asymmetric advantage.
- Risk: tactical blindness at about 10^4 nodes against depth 24. See §7.

**D10 · Rust ↔ Python**
- Options: (a) a separate inference server over pipes; (b) ONNX Runtime/TensorRT inside Rust (`ort`); (c) tch-rs; (d) **PyO3: Python drives the loop, Rust owns games and trees.**
- **Choice: (d)** via maturin:

  ```python
  sp = core.SelfPlay(n=512, ...)
  x = sp.leaves()          # numpy arrays
  while True:
      p, v = net(x)        # torch
      x = sp.step(p, v)    # Rust: backup, select, play moves, write samples
  ```

- One process, no IPC, and the batch size is simply the number of games. Match play uses the same loop with virtual-loss leaves.
- Revisit (b) only if Python overhead becomes measurable.
- Risk: PyO3 build friction on Windows. Mitigation: build in WSL (D12).

**D11 · Training**
- PyTorch, bf16 autocast, AdamW with cosine schedule, batch 1024.
- Features are computed **on the fly by Rust** (`core.encode(positions)`) from compact position records, so changing features never forces data regeneration.

**D12 · Platform**
- **Choice:** WSL2 Ubuntu for Rust, Apeiron and match runs.
- Training: either PyTorch inside WSL (CUDA in WSL is not verified yet; it's an M0 check) or Windows PyTorch, which works today, reading the same files.
- Risk: WSL memory is limited to 15 GB by default; raise it in `.wslconfig`.

---

## 3. Repo structure (flat)

```
InfiniteChess/
  PLAN.md
  core/                    Rust crate, built as a Python module with maturin
    Cargo.toml
    src/lib.rs             PyO3 glue only                          [tooling]
    src/pos.rs             Pos, ICN parse/print, make/undo, hash  [YOU]
    src/moves.rs           exact intervals, legality, candidates   [YOU]
    src/rules.rs           terminal: mate/stalemate/rep/50/insuffmat [YOU]
    src/encode.rs          tokens, pair relations, move queries    [YOU]
    src/mcts.rs            PUCT + Gumbel root, tree reuse          [YOU]
    src/selfplay.rs        parallel games, sample writer           [YOU, small]
    tests/                 perft8x8, oracle diff, fuzz             [tooling]
  py/
    net.py                 model + losses                          [YOU]
    train.py               loop, checkpoints                       [YOU core; logging = tooling]
    data.py                shard reading, batching                 [tooling]
    match.py               engine vs engine over the `serve` protocol, referee-checked [tooling, done M0]
    sprt.py                pentanomial LLR (about 40 lines)        [YOU]
  tools/
    referee/referee.ts     site rules as arbiter: legality + game end (imports ref/site, AGPL, local only) [tooling, done M0]
    apeiron.sh             build a frozen Apeiron release into ~/engines (no Apeiron code here) [tooling, done M0]
    make_openings.py       builds the frozen opening suite                               [tooling, done M0]
    check_games.py         replays engine games through the referee                      [tooling, done M0]
    setup_wsl.sh, setup_wsl_py.sh   one-time WSL setup (Node; Rust stable, venv, torch)    [tooling, done M0]
  bench/openings_classical_v1.jsonl   frozen opening suite (sha256 in PROGRESS.md)
  PROGRESS.md          dated experiment log: hypothesis, setup, numbers, verdict (§9) [YOU]
  data/                    gitignored
  ref/                     gitignored: apeiron, site, perft probes, benchmarks
```

---

## 4. Milestones

Times are focused hours of your own work. Compute figures come from §0.1.

### M0 · Harness (10–15 h) · **done 2026-10-17** (details in `PROGRESS.md`)

1. WSL setup: Rust stable for `core`, Apeiron's nightly for Apeiron, maturin, and a check that PyTorch sees CUDA in WSL.
2. `match.py`: plays two `serve` processes against each other.
   - **Sanity check:** Apeiron level 8 vs level 1 must score above 90%.
   - Verify the 1000-ply cap and that terminal states agree with Apeiron's.
3. Opening suite (§5), frozen to a file.

- You write: nothing critical. This milestone is tooling.
- Compute: about 1 h of CPU.

### M1 · Movegen (50–90 h) · **done 2026-10-17** (results in `PROGRESS.md`)

Steps:
1. `pos.rs`: ICN parsing and printing, make/undo, hash.
2. `moves.rs`: intervals, pins, checks, double check, castling, en passant, promotion.
3. `rules.rs`: all terminal conditions per §1.1.
4. Candidate rule from D4.

**Done when all of these pass:**
- **a) Bounded perft.** With world border [1,8]², perft equals standard chess:
  - start position D1–D5 = 20 / 400 / 8,902 / 197,281 / 4,865,609;
  - plus the usual tricky positions (Kiwipete and similar) through your own FEN→ICN converter.

  Inside an 8×8 border the site's rules reduce to chess. Apeiron used exactly this check (`CHANGELOG.md:785`).
- **b) Exact sets equal the site.** On ≥ 10,000 positions, our individual moves and per-direction intervals equal the site's `legalmoves.calculateAll` output (a `moves` command added to `tools/referee/referee.ts` at the start of M1). Positions: the 24 probes, positions from Apeiron games, and random playouts with coordinates up to ±2^62.
- **c) Replay.** Every move of ≥ 10,000 Apeiron `data_gen` games is legal for us, and our terminal verdicts match Apeiron's `termination` field.
- **d) Coverage.** ≥ 99.9% of Apeiron's played moves are in our candidate set (Apeiron's far-shell moves map to our far representative of the same piece and direction). Report the size of (Apeiron root list minus ours) on the 24 probes.
- Not a criterion: Apeiron's perft table (§1.7). Our counts will differ by design, since our candidate rule is our own. Use the table only to investigate a coverage miss from (d).

- You write: `pos.rs`, `moves.rs`, `rules.rs` and the candidate rule.
- Tooling: the oracle script and test harness.
- Compute: CPU only.

### M2 · Distillation: policy alone (no search) beats Apeiron level 1 at > 60% (60–100 h) · **done 2026-10-18** (E6: top-1 0.506; 0.946 vs L1, 200 pairs, 95% CI [0.932, 0.961]; see PROGRESS.md)

Running jobs (stop/resume any time, inside WSL): `bash tools/jobs.sh status`, `bash tools/jobs.sh stop gen|pack|train`, `bash tools/jobs.sh start gen|pack|train`. Details in `PROGRESS.md`.

1. **Data:** 25M positions from `data_gen --variants Classical --depth 12 --keep-noisy`. That is about 43 h on 18 threads, at 0.58M positions/h. Details in §6.
2. `encode.rs` and `net.py`: start with L = 8, d = 256, 8 heads (about 4M parameters, about 21k positions/s measured).
3. **Training:**
   - Policy: cross-entropy against Apeiron's best move. Optionally a soft MultiPV target (§6).
   - Value: Apeiron's score through a logistic, plus a small WDL term.
   - Moves left.
   - x-mirror augmentation.
   - About 10 epochs, roughly 10 h on the GPU.
4. **Offline gate:** top-1 agreement with depth-12 labels on fresh, held-out games ≥ 45%; value log-loss below that of the logistic of Apeiron's own static eval.
5. **Match:** argmax policy vs level 1. SPRT with logistic bounds [0, 70.4] (70.4 Elo = 60%), then confirmation (§5).

- You write: `encode.rs`, `net.py`, the loss functions.
- Tooling: data conversion and the loader.
- Compute: about 2 days of CPU for labels, about 0.5 day of GPU.

### M3 · MCTS beats Apeiron level 4 at > 55% (40–80 h) · **done 2026-10-19** (E7: 0.911 at 1 s, 0.940 at 5 s vs L4, 0 losses; see PROGRESS.md)

1. `mcts.rs`: PUCT, virtual-loss batching through the D10 loop, tree reuse, exact terminals, fixed time per move.
2. Tune cPUCT, FPU and root temperature by SPRT against our own previous version.
3. Match against level 4 at 1 s/move: SPRT [0, 34.9] (34.9 Elo = 55%), then confirmation.
4. Collect positions from our games against Apeiron (on-policy) and relabel them for the next net (§6).

- You write: all of `mcts.rs`.
- Tooling: the match glue.
- Compute: about 1000 games × about 200 plies × 1 s, with 8 games sharing the GPU, is about 4 h per SPRT.

### M4 · Self-play RL beats Apeiron v7.10.0 at full strength at > 55%, 5 s per move (120–250 h) · **in progress since 2026-10-19** (PROGRESS E12; jobs `rl`, `sp`, `ladder`)

1. **KataGo-style loop:**
   - 512 parallel games;
   - playout cap randomization: p = 0.25 for a full search (about 800 visits), otherwise about 150 visits; only full searches are recorded;
   - Gumbel root;
   - policy target pruning;
   - auxiliary targets (D8);
   - policy-surprise weighting;
   - a replay window;
   - a new net every N samples, mixing in 20–30% Apeiron-labelled data early on, then decaying it.
2. **Ladder:** level 6 → level 7 → v7.1.0 at full strength → v7.10.0 at full strength (§5). Grow the net to 8–12M parameters if the 4M net plateaus.

- **Throughput assumptions:**
  - 4M-parameter net at about 50k evaluations/s (measured on an idle GPU in WSL; fused attention and CUDA graphs may add more, unverified);
  - about 250 visits per move on average, giving about 200 moves/s, about 720k moves/h, and about 180k recorded samples/h (about 4.3M/day).
- **Estimate:** 20–60M samples to pass the teacher, so **1–2 weeks locally** with the 4M net (proportionally longer for a bigger net, see R7).
- **Cloud burst option:** 4× RTX 4090 at about $0.34–0.74/h each on Runpod (H100: $2–4/h; [Oct 2026 aggregator](https://deploybase.ai/articles/rtx-4090-cloud-price), [Hostinger comparison](https://www.hostinger.com/tutorials/?p=158956)). Two weeks would cost about **$500–1000**.
- You write: the RL loop logic in `selfplay.rs` and `train.py`.
- Tooling: orchestration and plots.

### M5 · "Unbeatable by humans" (open-ended research, 60–150 h plus compute)

Elo alone cannot show this. KataGo, at superhuman settings, lost more than 97% of games to an adversarial policy that is itself weak, and human amateurs then reproduced the trick ([Wang et al., ICML 2023, arXiv 2211.00241](https://arxiv.org/abs/2211.00241); [goattack.far.ai](https://goattack.far.ai)). So "unbeatable" is defined operationally by four tests, all at 5 s per move:

- **a) Margin over the strongest engine.** ≥ 75% (about +190 Elo) against Apeiron v7.10.0 and against the newest release.
- **b) Adversarial robustness (§9 R4).** An adversary trained against our frozen engine with a fixed budget (start with 2 GPU-days) wins < 1%. Every exploit found is fixed by adversarial training and re-tested.
- **c) Out-of-distribution stress.** No losses in 1000 games against Apeiron levels 4–8 from unusual starts: pieces stretched far apart, early back-rank escapes, and positions mined from human games (§9 R2).
- **d) Human games.** **0 losses in ≥ 100 games** against the strongest available players on infinitechess.org. By the rule of three, this bounds the loss rate below 3% at 95% confidence. It needs the community's cooperation (§8).

---

## 5. Benchmark protocol

**Frozen opponent.**
- Apeiron **v7.10.0** (`4e1aba5`), built in WSL2 Ubuntu 24.04 with `nightly-2025-11-15` and `--features sprt`.
- Record the binary's sha256 in the results.
- Keep v7.1.0 (`be5a76f`) as an extra rung.
- Never upgrade mid-milestone.

**Interface.** `sprt serve` (§1.6), one warm process per game.
- Apeiron is charged its own `elapsed_ms`.
- Our engine is charged wall-clock time including net inference.

**Time.**
- **Milestone confirmations: 5 s per move**, which is the target playing condition.
- Development SPRTs: 1 s per move, so an SPRT finishes in hours instead of days.
- Overrunning 2× the budget loses the game.
- Cost of one confirmation: a game is about 200 plies × 5 s, roughly 17 min. 200 pairs (400 games) with 4 games in parallel take about **1.2 days**.

**Hardware and parity.**
- Same machine. Apeiron gets 16 threads in total; our engine gets the GPU plus ≤ 4 CPU threads.
- Running 4 games in parallel: Apeiron gets `APEIRON_THREADS = 4` per game, and our 4 engines share the GPU. Report our evaluations per second per game alongside the result.
- One extra run plays a single game at a time, with Apeiron at 16 threads, so the result under the fairest conditions is also known.
- Also report results against Apeiron on 1 thread.

**Openings.**
- 500 positions:
  1. random MultiPV picks from Apeiron at depth 7 for 8 plies (Apeiron's own book method);
  2. keep only positions with |eval| ≤ 60 cp at depth 14, deduplicated.
- Each opening is played twice with colours swapped, giving pentanomial pairs.
- The file is frozen and its hash recorded.

**Rules and adjudication.**
- Site rules as in §1.1.
- Every move is checked by our exact movegen, and a sample of games goes through the site's [ICN validator](https://infinitechess.org/icnvalidator).
- An illegal move, a crash or a time forfeit loses.
- Cap of 1000 plies, which is a draw.
- **No adjudication in confirmation runs.** In SPRT runs, resign only when Apeiron reports |score| ≥ 1000 cp *and* our WDL ≥ 0.97, both for 6 plies.

**Statistics.**
- Pentanomial GSPRT with logistic Elo, α = β = 0.05 (LLR ±2.94), H0: Elo = 0.
- H1 equals the target score: 400·log10(p/(1−p)), which is 34.9 for 55% and 70.4 for 60%.
- At least 200 pairs.

**A milestone counts only if both of these hold:**
- (i) the SPRT accepts H1;
- (ii) a fresh **200-pair (400-game) confirmation match at 5 s per move** has a point estimate ≥ the target score and a 95% confidence interval (pentanomial variance) whose lower bound is above 50%.

**Latest Apeiron.** Development stays frozen at v7.10.0. When M4 passes, rebuild the newest Apeiron release and run the same confirmation match against it. That result is the "beats the latest" claim.

Publish the ICN games, the binary hashes, the configs and the LLR trace.

**Draw-rate note.** Apeiron's Classical self-play draws 75–82% of games. A 55% score corresponds to only about +35 Elo, but it needs real wins. A small contempt (draw value −0.05 against weaker opponents) is allowed and must be declared.

---

## 6. Data pipeline

**Phase 1 (M2).**
- `data_gen --variants Classical --depth 12 --keep-noisy` with randomized openings (defaults: 12 plies, MultiPV 5, 120 cp window). Target **25M positions, about 43 h** (0.58M positions/h, 18 threads).
- Optional soft policy: a small **private patch** to Apeiron that dumps MultiPV top-8 root scores at depth 10 for about 5M positions. The patch lives in a GPL fork outside our repo (§7).
- Filtering:
  - drop positions after the first twofold repetition and after adjudication;
  - downweight short repetition games, since 44 of 72 games ended that way at depth 9;
  - shuffle across games with a buffer of at least 2M positions;
  - deduplicate by hash.

**Phase 1b (M3).**
- Positions from our games against Apeiron, relabelled by Apeiron at depth 12.
- Needs either a relabel tool in the fork, or `serve` driven with `fixed_time`, not `max_depth` (see the trap in §1.6).
- This is the on-policy data Apeiron's own experiments say is needed for positional inputs.

**Phase 2 (M4).** Our self-play: visit distributions over candidates, plus value, auxiliary targets and the result.

**Storage.**
- Raw: game records (Apeiron's JSONL, or our equivalent: start ICN, moves, per-ply labels).
- Training shards: `.npz` written by a Rust replay tool. Replaying with *our* movegen doubles as test M1-c.
- Per position:
  - pieces: int64 [32, 3] (x, y, packed code);
  - side, en passant, clock, repetition count;
  - candidates: variable length (from-index u8, to x/y i64, promotion u8) through an offsets array;
  - target index or soft policy (f16);
  - score f32, WDL i8, moves left i16;
  - game id, ply, depth.
- Size: about 1 KB per position, so 25M positions is about 25 GB raw and less with zstd. There is 548 GB free.
- Features are not stored; Rust computes them per batch (D11).

---

## 7. Risks and mitigations

| risk | mitigation |
|---|---|
| Long-range moves outside the candidate set (a move that can never be played) | Wider root set; coverage metric against Apeiron's played moves; every miss becomes a test; far representatives are relative, not tied to an absolute shell |
| Tactical blindness: about 10^4 nodes vs Apeiron's depth 24 | Rules-derived attack, pin and check features; search-score value targets; boosted priors for checks and captures; scale evaluations per second (smaller net, CUDA graphs, later TensorRT) before scaling net size |
| Compute | Measured budgets (§0.1); cloud burst estimate in M4; playout cap randomization and Gumbel to stretch simulations |
| Data/overfitting (InfNNUE, sparse inputs) | Eval targets during distillation, on-policy relabels from M3, a fresh-game holdout as referee (Apeiron's own rule), exact symmetry augmentation |
| Moving target: v7.1 → v7.10 took 7 days (tag dates 2026-10-04 → 2026-10-11) | Freeze v7.10.0; re-test the latest release only after M4 as a stretch goal |
| Rule mismatch (repetition window, en passant, insufficient material, border) | Site oracle diff (M1-b), Apeiron replay (M1-c), ICN validator on match games |
| High draw rate (75–82%) | Opening suite filtered for balance and diversity; declared contempt; enough pairs |
| Windows build and antivirus problems | Everything that spawns engines runs in WSL2 |
| GPU shared with other applications (seen during these measurements) | Pin benchmarks to an idle GPU; report evaluations/s alongside every match |
| Systematic blind spots that a human can learn to exploit (the KataGo lesson) | M5 adversary training (R4), out-of-distribution stress games, human test games |
| No public source of human games (the site has no export API; Apeiron's data was an export from the site) | Ask the site admins for an export (§8); until then, use our own stress positions |

**Kill and pivot gates.**
- **After M2:** if top-1 agreement is below 45% or policy alone loses to level 1, fix inputs and relations before writing MCTS.
- **After M3 + one RL generation:** if 1 s/move scores below 30% against level 6, try MCTS leaves with a short capture-only resolution, and a larger evaluations-per-second budget, before any alpha-beta pivot.

**GPL-3.0 boundary (Apeiron) and AGPL-3.0 (site).**
- **Never in our repo:** Apeiron or site source code, translations of it, its eval-net weights (`src/eval_net/*.bin`), or any crate linking `apeiron`.
- **Allowed:** running Apeiron as a separate program over stdin/stdout or command line, which the FSF treats as communication between separate programs ([GPL FAQ #MereAggregation](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation)).
- **Data we generate with Apeiron** (positions, scores, moves) is output, not program code. Per the FSF, the GPL does not cover the output of a program's use ([#GPLOutput](https://www.gnu.org/licenses/gpl-faq.html#GPLOutput)), so nets trained on it are ours.
- **Private Apeiron patches** (perft dump, MultiPV dump, relabel tool) live in a separate GPL fork under `ref/`. If we ever distribute them, they are distributed under the GPL.
- **The site's code** runs only as a local test oracle and is never copied. Rules are implemented from the rule descriptions in §1.1. Our candidate rule (D4) is designed independently, and §1.3 documents Apeiron's rule only so we can measure against it.
- Not legal advice. If the engine will be published, take a quick look with someone who knows copyleft.

---

## 8. Open questions for you

1. **Variant:** Classical (assumed), not CoaIP?
2. ~~**Target version**~~ **Decided:** the latest Apeiron. Frozen at v7.10.0 for development; the newest release is re-tested after M4 (§5).
3. ~~**Changing M1**~~ **Decided:** our own engine and candidate rule, no Apeiron-compatible mode. M1 = exact match with the site + 8×8 perft + coverage.
4. **Fair hardware:** 5 s per move is decided. Still to confirm: Apeiron gets 16 threads in total against our GPU plus ≤ 4 threads (proposal in §5).
5. ~~**Deployment**~~ **Decided:** the engine runs locally only, never in the browser. The GPU + MCTS design (D9, D10) stands, with no WASM or CPU-only constraint on net size.
6. **Budget:** is any cloud budget available for M4 and M5, and if so how much?
7. **Apeiron fork:** may we keep private patches to Apeiron (MultiPV and relabel output) in a GPL fork under `ref/`?
8. ~~**Platform**~~ **Decided:** WSL2.
9. **Ownership split:** your weekly hours, and whether the split marked `[YOU]` in §3 matches what you want to write yourself.
10. **Human data and opponents (M5-d, §9 R2/R4):** can we ask the infinitechess.org admins for an export of human games, and find strong players willing to play test games? Apeiron's developers got such an export (`apeiron:src/bin/export_eval_features.rs:147-150`); the site itself has no public API for it.
11. **Publishing:** should the findings (§9) go out as a paper or blog post, and the engine as open source? That decides how strictly the GPL boundary in §7 must be documented.

---

## 9. Research agenda

The aim is findings that are new for unbounded-board chess, not just a stronger engine. A quick literature search found **no published neural engine** (policy + value + search) for chess on an unbounded board. Apeiron's net only corrects its hand-written evaluation (§1.5). The search was limited, so treat "new" as "not found yet" and re-check before publishing.

Each item lists the question, the experiment, the metric, and when it fits. Every experiment is logged in `PROGRESS.md` with its date, setup, numbers and verdict, the way Apeiron logs its net experiments (`apeiron:docs/eval-net-plan.md`, Log).

**R1 · A complete, finite move abstraction on an unbounded board** (M1–M3)
- Question: can the policy cover *every* legal move without a hand-tuned candidate list?
- Idea: on each ray, the *breakpoints* are the distances where the destination's relation to some piece changes: it lands on that piece's row, column or diagonal, ±1 or ±2 from it, on a knight offset, or adjacent to a king. Between two neighbouring breakpoints, all squares relate to every piece in the same way and differ only in distance. That splits each ray into O(pieces) **segments**.
- The policy then picks (piece, direction, segment), a finite choice that still covers everything, plus a distance inside the segment: the near end, the far end, or a log-scale choice.
- Experiment: compare against the D4 heuristic on four numbers: coverage of Apeiron's moves, policy top-1, Elo at 1 s, and the number of candidates per node.
- Sub-question: is the exact distance inside a segment ever decisive? Measure by having Apeiron evaluate several squares from the same segment.

**R2 · Do translation-invariant relational nets generalize to extreme distances?** (M2, M5-c)
- Question: does a net trained on compact positions keep its accuracy when pieces are 10^3–10^12 apart?
- Experiment: "stretched" test sets. Move groups of pieces apart along free lines, keeping every relation that matters (Apeiron's compression keeps only order beyond a fixed span, §1.3/`src/compression`). Measure policy and value agreement with Apeiron as a function of the stretch.
- Expected finding: a span S beyond which only the order of pieces matters. If the net degrades past S, clip the distance buckets at S.

**R3 · Does distillation from Apeiron help or put a ceiling on us?** (M2–M4)
- Experiment: two small-scale RL runs with the same compute, one starting from the distilled net and one from random weights. Compare the curves of Elo against compute.
- This also answers whether our own engine needs Apeiron at all.

**R4 · Adversarial robustness and exploits a human could learn** (M5)
- Question: does our engine have systematic blind spots, the kind the cyclic-group exploit revealed in KataGo?
- Experiment: train an adversary policy against our frozen engine (victim-play, as in Wang et al. 2023). Classify every exploit it finds. Fix it with adversarial training and measure the exploit win rate before and after.
- Expected infinite-board candidates: far escapes behind the lines, mating nets with no edges, insufficient-material borderlines (K+Q vs K is a draw), and repetition traps.

**R5 · Dense per-piece auxiliary targets** (M2, M4)
- Question: does a per-token "survival" head speed up training? It predicts whether each piece is captured within k plies, a chess analogue of KataGo's per-point ownership target, which sped up its training ([arXiv 1902.10565](https://arxiv.org/abs/1902.10565)).
- Experiment: ablation on the same data, comparing the policy and value learning curves.

**R6 · Search under unbounded branching** (M3–M4)
- Question: at a fixed 5 s, which search plays best: PUCT, Gumbel with sequential halving, or progressive widening over R1's segments?
- Experiment: SPRT matches between the variants, plus the rate at which a root move is missing from the search.

**R7 · Net size against evaluations per second at 5 s** (M4)
- Question: which net size gives the best play at 5 s per move? Measured on an idle GPU: a 4M-parameter net gives about 250k evaluations per move, a 12M net about 125k, a 25M net about 70k.
- Experiment: a scaling curve of Elo against parameters at a fixed 5 s, using an idle GPU. It sets the final net size for "highest quality of play".
