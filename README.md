# Sokoban Solver — make.fun bounty entry

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/cryptocollectorid/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/cryptocollectorid/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

Entry for the make.fun bounty **"Build the best Sokoban solver, 21 November
2026"** (pot $100, 1st–10th place $10 each). One program, `main.ts` (Node 22,
standard library only), answers all 5 instances of a set in a single run of at
most 60 seconds.

## How it works

The generator builds every level as a 1-wide hallway with 1–4 rooms off it
(6×6 slots, one door each, 3–4 boxes and goals per room), and a box pushed
through a door into the hallway can never come back — so every box stays in
its room and the push-optimal count P\* is the sum of the rooms' optima.

The solver exploits that structure:

1. **Decompose** the level into its rooms (floor components above/below the
   hallway row, each opening through a single door).
2. **Solve each room independently** with a push-optimal A\* over pushes:
   - State = player square + sorted box squares.
   - Admissible heuristic = minimum-cost bipartite matching (Hungarian) of
     boxes to goals over wall-only BFS distances — each push moves one box
     one cell, so the matching cost lower-bounds the pushes left.
   - Pruning = dead squares (reverse pull-reachability from the goals) and
     2×2 freeze blocks (a box caught in a full block of walls/boxes while
     off-goal can never move).
   - The player walks between pushes by BFS over the room floor.
   - A solution must end with the player able to walk back out to the door
     (except the last room, which need not be exited).
3. **Stitch** the rooms together with hallway transit moves. Transit costs
   zero pushes, so per-room push-optimality is exactly global
   push-optimality.
4. **Fallbacks**: if a room's A\* exhausts its node budget, a greedy
   best-first search (same heuristic, same pruning) still finds a solving
   — even suboptimal — sequence, because an unsolved instance scores 0
   while a solved-but-suboptimal one still scores 500×P\*/pushes.

**Determinism**: budgets are counted in node expansions, never wall-clock
time, iteration orders are fixed, and no clock, environment, or randomness
is read — two runs on one set write byte-identical outputs. A global
expansion cap across the whole run guarantees a single hard room can never
eat the 60 s wall and zero the entry.

## Local results

Scored with the kit's own scorer (`program/scorer/score.ts`) on the kit's
practice set, and on 6 further sets drawn from the kit's real generator
(`program/generator/draw.ts`) with fresh random seeds:

| Set | This solver | House baseline (plain-astar 1.0.0) |
| --- | --- | --- |
| Practice set | **1,000,000** (5/5 at exactly P\*) | 200,000 (1/5 solved) |
| 6 random sets | **1,000,000 each** (30/30 at exactly P\*) | — |

Whole-set runtime is ~0.5 s locally (limit: 60 s), worst room ~5,100
expansions (budget: 60,000). Outputs verified byte-identical across
repeated runs.

## Run it

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads `/input/<id>/level.txt` (XSB) for `i001`–`i005` and writes
`/output/<id>/solution.txt` (moves in `u`/`d`/`l`/`r`) for each.
