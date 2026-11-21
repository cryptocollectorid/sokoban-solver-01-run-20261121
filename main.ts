// Sokoban solver for make.fun — sokoban-solver-01-run-20261121.
//
// Strategy: the generator builds every level as a 1-wide hallway with 1-4 rooms
// off it (6x6 slots, one door each, 3-4 boxes/goals per room), and a box pushed
// through a door into the hallway can never come back, so every box stays in its
// room and the push-optimal count P* is the sum of the rooms' optima. We solve
// each room independently with a push-optimal A* (push-space search, Hungarian
// matching heuristic over BFS distances, dead-square + 2x2-freeze pruning), then
// stitch the rooms together with hallway transit moves. Pushes are what the
// checker scores, so per-room push-optimality is exactly global push-optimality.
//
// Determinism: budgets are counted in node expansions, never wall-clock time;
// iteration orders are fixed; the grader runs the entry twice and the outputs
// must be byte-identical. No clock, no environment, no randomness is read.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Per-room A* expansion budget, per-room greedy-fallback budget, and a global
// budget across the whole run. The global cap is the safety valve: a room far
// harder than anything in the practice set can never eat the 60 s wall and
// zero the whole entry. Observed worst room on 35 generated levels: ~5,100
// expansions; typical whole-set total is under 40,000.
const ASTAR_BUDGET = 60_000;
const GREEDY_BUDGET = 120_000;
const GLOBAL_BUDGET = 1_000_000;

const DIRS: { dx: number; dy: number; ch: string }[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

interface Grid {
  w: number;
  h: number;
  n: number;
  wall: Uint8Array;
  goals: Set<number>;
  boxes: number[];
  player: number;
}

interface Room {
  cells: number[];
  cellSet: Set<number>;
  doors: number[];
  boxes: number[];
  goals: number[];
}

interface RoomResult {
  ok: boolean;
  moves: string;
  pushes: number;
  endPlayer: number;
}

function parse(text: string): Grid {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const n = w * h;
  const wall = new Uint8Array(n);
  const goals = new Set<number>();
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#' ? 1 : 0;
      if ('.*+'.includes(ch)) goals.add(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, n, wall, goals, boxes, player };
}

// All-pairs shortest-path distances ignoring boxes (walls only).
function allPairsDist(g: Grid): Int32Array {
  const dist = new Int32Array(g.n * g.n).fill(-1);
  const q = new Int32Array(g.n);
  for (let s = 0; s < g.n; s++) {
    if (g.wall[s]) continue;
    let qh = 0;
    let qt = 0;
    q[qt++] = s;
    dist[s * g.n + s] = 0;
    while (qh < qt) {
      const c = q[qh++];
      const d = dist[s * g.n + c];
      const x = c % g.w;
      const y = (c / g.w) | 0;
      for (const { dx, dy } of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
        const nn = ny * g.w + nx;
        if (g.wall[nn] || dist[s * g.n + nn] !== -1) continue;
        dist[s * g.n + nn] = d + 1;
        q[qt++] = nn;
      }
    }
  }
  return dist;
}

// Split the level into its rooms: the hallway is the player's row, rooms are
// the floor components above/below it, each opening through a single door.
function decompose(g: Grid): { hallRow: number; rooms: Room[] } {
  const hallRow = (g.player / g.w) | 0;
  const seen = new Uint8Array(g.n);
  const rooms: Room[] = [];
  const stack: number[] = [];
  for (let s = 0; s < g.n; s++) {
    if (g.wall[s] || ((s / g.w) | 0) === hallRow || seen[s]) continue;
    const cells: number[] = [];
    stack.length = 0;
    stack.push(s);
    seen[s] = 1;
    while (stack.length > 0) {
      const c = stack.pop()!;
      cells.push(c);
      const x = c % g.w;
      const y = (c / g.w) | 0;
      for (const { dx, dy } of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
        const nn = ny * g.w + nx;
        if (g.wall[nn] || ((nn / g.w) | 0) === hallRow || seen[nn]) continue;
        seen[nn] = 1;
        stack.push(nn);
      }
    }
    const cellSet = new Set(cells);
    const doors = cells.filter((c) => {
      const x = c % g.w;
      const y = (c / g.w) | 0;
      for (const { dx, dy } of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (ny === hallRow && nx >= 0 && nx < g.w && !g.wall[ny * g.w + nx]) return true;
      }
      return false;
    });
    rooms.push({
      cells,
      cellSet,
      doors,
      boxes: g.boxes.filter((b) => cellSet.has(b)),
      goals: [...g.goals].filter((x) => cellSet.has(x)),
    });
  }
  return { hallRow, rooms };
}

// Squares from which a box can still be pulled to a goal (reverse pull BFS).
// A square that is not live and not a goal can never hold a box in a solvable
// state, so states with a box there are pruned.
function liveSquares(g: Grid, roomCells: Set<number>, goals: number[]): Set<number> {
  const live = new Set<number>(goals);
  let changed = true;
  while (changed) {
    changed = false;
    for (const c of roomCells) {
      if (live.has(c)) continue;
      const x = c % g.w;
      const y = (c / g.w) | 0;
      for (const { dx, dy } of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        const bx = x - dx;
        const by = y - dy;
        if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
        if (bx < 0 || by < 0 || bx >= g.w || by >= g.h) continue;
        const nn = ny * g.w + nx;
        const b = by * g.w + bx;
        if (live.has(nn) && !g.wall[b] && roomCells.has(b)) {
          live.add(c);
          changed = true;
          break;
        }
      }
    }
  }
  return live;
}

function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a = [...Array(n).keys()];
  const gen = (k: number): void => {
    if (k === n) {
      out.push(a.slice());
      return;
    }
    for (let i = k; i < n; i++) {
      const t = a[k];
      a[k] = a[i];
      a[i] = t;
      gen(k + 1);
      const t2 = a[k];
      a[k] = a[i];
      a[i] = t2;
    }
  };
  gen(0);
  return out;
}

// Shared scratch buffers for the search (reused, never reallocated per state).
class Scratch {
  q = new Int32Array(1024);
  seen = new Int32Array(1024);
  stamp = 0;
  occ = new Int32Array(1024);
  occStamp = 0;
  prev = new Int32Array(1024);
  prevCh = new Int32Array(1024);

  ensure(n: number): void {
    if (this.q.length < n) {
      this.q = new Int32Array(n);
      this.seen = new Int32Array(n);
      this.occ = new Int32Array(n);
      this.prev = new Int32Array(n);
      this.prevCh = new Int32Array(n);
    }
  }
}

// Player reachability over roomCells avoiding boxes. Returns the seen stamp.
function reachability(
  g: Grid,
  roomCells: Set<number>,
  boxes: number[],
  from: number,
  sc: Scratch,
): number {
  sc.ensure(g.n);
  sc.stamp++;
  const stamp = sc.stamp;
  const occStamp = ++sc.occStamp;
  for (const b of boxes) sc.occ[b] = occStamp;
  let qh = 0;
  let qt = 0;
  sc.q[qt++] = from;
  sc.seen[from] = stamp;
  while (qh < qt) {
    const c = sc.q[qh++];
    const x = c % g.w;
    const y = (c / g.w) | 0;
    for (const { dx, dy } of DIRS) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
      const nn = ny * g.w + nx;
      if (!roomCells.has(nn) || g.wall[nn] || sc.occ[nn] === occStamp || sc.seen[nn] === stamp) continue;
      sc.seen[nn] = stamp;
      sc.q[qt++] = nn;
    }
  }
  return stamp;
}

// Shortest player path (move letters) from `from` to `to` over roomCells
// avoiding boxes; '' when already there, null when unreachable.
function bfsPath(
  g: Grid,
  roomCells: Set<number>,
  boxes: number[],
  from: number,
  to: number,
  sc: Scratch,
): string | null {
  if (from === to) return '';
  sc.ensure(g.n);
  const stamp = ++sc.stamp;
  const occStamp = ++sc.occStamp;
  for (const b of boxes) sc.occ[b] = occStamp;
  let qh = 0;
  let qt = 0;
  sc.q[qt++] = from;
  sc.seen[from] = stamp;
  sc.prev[from] = from;
  while (qh < qt) {
    const c = sc.q[qh++];
    const x = c % g.w;
    const y = (c / g.w) | 0;
    for (let d = 0; d < 4; d++) {
      const { dx, dy } = DIRS[d];
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
      const nn = ny * g.w + nx;
      if (!roomCells.has(nn) || g.wall[nn] || sc.occ[nn] === occStamp || sc.seen[nn] === stamp) continue;
      sc.seen[nn] = stamp;
      sc.prev[nn] = c;
      sc.prevCh[nn] = d;
      if (nn === to) {
        let s = '';
        let cur = nn;
        while (cur !== from) {
          s = DIRS[sc.prevCh[cur]]!.ch + s;
          cur = sc.prev[cur];
        }
        return s;
      }
      sc.q[qt++] = nn;
    }
  }
  return null;
}

interface SearchOpts {
  budget: number;
  requireExit: boolean;
  greedy: boolean;
  global: { used: number; cap: number };
}

// Push-space search over one room. The player starts on `entry` (the hallway
// cell by the door); boxes must stay inside the room; the search ends with
// every box on a goal and, when requireExit is set, the player able to walk
// back out to the door.
function searchRoom(
  g: Grid,
  room: Room,
  dist: Int32Array,
  entry: number,
  sc: Scratch,
  opts: SearchOpts,
): RoomResult | null {
  const roomCells = new Set<number>(room.cells);
  roomCells.add(entry);
  const live = liveSquares(g, roomCells, room.goals);
  const n = room.boxes.length;
  const goalArr = room.goals.slice().sort((a, b) => a - b);
  const perms = n === goalArr.length && n <= 4 ? permutations(n) : null;
  const door = room.doors[0] ?? entry;

  // Admissible heuristic: minimum-cost matching of boxes to goals over
  // wall-only BFS distances (each push moves one box one cell). Falls back to
  // the sum of per-box nearest-goal distances when the counts differ.
  const heuristic = (bs: number[]): number => {
    if (perms !== null) {
      let best = Infinity;
      for (const p of perms) {
        let s = 0;
        for (let i = 0; i < n; i++) {
          const d = dist[bs[i]! * g.n + goalArr[p[i]!]!];
          if (d < 0) {
            s = Infinity;
            break;
          }
          s += d;
        }
        if (s < best) best = s;
      }
      return best;
    }
    let s = 0;
    for (const b of bs) {
      let m = Infinity;
      for (const gl of goalArr) {
        const d = dist[b * g.n + gl];
        if (d >= 0 && d < m) m = d;
      }
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  };

  // Only the box just moved can newly deadlock: a box on a non-live square,
  // or a box caught in a full 2x2 block of walls/boxes while off-goal.
  const movedDead = (bs: number[], moved: number): boolean => {
    if (!g.goals.has(moved) && !live.has(moved)) return true;
    const x = moved % g.w;
    const y = (moved / g.w) | 0;
    const boxSet = new Set(bs);
    for (const [ox, oy] of [[0, 0], [-1, 0], [0, -1], [-1, -1]] as const) {
      const cs: number[] = [];
      let ok = true;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const cx = x + ox + dx;
          const cy = y + oy + dy;
          if (cx < 0 || cy < 0 || cx >= g.w || cy >= g.h) {
            ok = false;
            break;
          }
          cs.push(cy * g.w + cx);
        }
        if (!ok) break;
      }
      if (!ok) continue;
      if (!cs.every((c) => g.wall[c] || boxSet.has(c))) continue;
      for (const c of cs) {
        if (boxSet.has(c) && !g.goals.has(c)) return true;
      }
    }
    return false;
  };

  const packKey = (p: number, bs: number[]): number => {
    let k = p;
    let m = 256;
    for (const b of bs) {
      k += b * m;
      m *= 256;
    }
    return k;
  };

  // Binary heap on parallel arrays, ordered by f, then h, then insertion id
  // (greedy mode orders by h alone). Fixed tie-breaks keep runs deterministic.
  const hf: number[] = [];
  const hh: number[] = [];
  const hid: number[] = [];
  const hless = (i: number, j: number): boolean =>
    opts.greedy
      ? hh[i]! < hh[j]! || (hh[i] === hh[j] && hid[i]! < hid[j]!)
      : hf[i]! < hf[j]! || (hf[i] === hf[j] && (hh[i]! < hh[j]! || (hh[i] === hh[j] && hid[i]! < hid[j]!)));
  const hpush = (f: number, h: number, id: number): void => {
    let i = hf.length;
    hf.push(f);
    hh.push(h);
    hid.push(id);
    while (i > 0) {
      const par = (i - 1) >> 1;
      if (!hless(i, par)) break;
      const tf = hf[i]!;
      hf[i] = hf[par]!;
      hf[par] = tf;
      const th = hh[i]!;
      hh[i] = hh[par]!;
      hh[par] = th;
      const ti = hid[i]!;
      hid[i] = hid[par]!;
      hid[par] = ti;
      i = par;
    }
  };
  const hpop = (): number => {
    const top = hid[0]!;
    const lf = hf.pop()!;
    const lh = hh.pop()!;
    const li = hid.pop()!;
    if (hf.length > 0) {
      hf[0] = lf;
      hh[0] = lh;
      hid[0] = li;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= hf.length) break;
        if (c + 1 < hf.length && hless(c + 1, c)) c++;
        if (!hless(c, i)) break;
        const tf = hf[i]!;
        hf[i] = hf[c]!;
        hf[c] = tf;
        const th = hh[i]!;
        hh[i] = hh[c]!;
        hh[c] = th;
        const ti = hid[i]!;
        hid[i] = hid[c]!;
        hid[c] = ti;
        i = c;
      }
    }
    return top;
  };

  const startBoxes = room.boxes.slice().sort((a, b) => a - b);
  const h0 = heuristic(startBoxes);
  if (h0 === Infinity) return null;
  for (const b of startBoxes) {
    if (!g.goals.has(b) && !live.has(b)) return null;
  }
  const players: number[] = [entry];
  const boxStates: number[][] = [startBoxes];
  const gs: number[] = [0];
  const parents: number[] = [-1];
  const pushDir: number[] = [-1];
  const pushBox: number[] = [-1];
  const best = new Map<number, number>([[packKey(entry, startBoxes), 0]]);
  const seenSet = opts.greedy ? new Set<number>([packKey(entry, startBoxes)]) : null;
  hpush(h0, h0, 0);
  let expanded = 0;
  let goalId = -1;
  while (hf.length > 0) {
    if (opts.global.used >= opts.global.cap) return null;
    const id = hpop();
    if (!opts.greedy && best.get(packKey(players[id]!, boxStates[id]!)) !== id) continue;
    const bs = boxStates[id]!;
    const p = players[id]!;
    if (bs.every((b) => g.goals.has(b))) {
      if (!opts.requireExit) {
        goalId = id;
        break;
      }
      const stamp = reachability(g, roomCells, bs, p, sc);
      if (sc.seen[door] === stamp) {
        goalId = id;
        break;
      }
      continue; // solved but the player cannot walk out: prune, keep searching
    }
    if (++expanded > opts.budget) return null;
    opts.global.used++;
    const stamp = reachability(g, roomCells, bs, p, sc);
    const boxSet = new Set(bs);
    for (let bi = 0; bi < n; bi++) {
      const b = bs[bi]!;
      const bx = b % g.w;
      const by = (b / g.w) | 0;
      for (let d = 0; d < 4; d++) {
        const { dx, dy } = DIRS[d]!;
        const tx = bx + dx;
        const ty = by + dy;
        const sx = bx - dx;
        const sy = by - dy;
        if (tx < 0 || ty < 0 || tx >= g.w || ty >= g.h) continue;
        if (sx < 0 || sy < 0 || sx >= g.w || sy >= g.h) continue;
        const t = ty * g.w + tx;
        const s = sy * g.w + sx;
        if (!roomCells.has(t) || g.wall[t] || boxSet.has(t)) continue;
        if (!roomCells.has(s) || g.wall[s] || sc.seen[s] !== stamp) continue;
        const nb = bs.slice();
        nb[bi] = t;
        nb.sort((a, b) => a - b);
        if (movedDead(nb, t)) continue;
        const hn = heuristic(nb);
        if (hn === Infinity) continue;
        const ng = gs[id]! + 1;
        const k = packKey(b, nb);
        if (opts.greedy) {
          if (seenSet!.has(k)) continue;
          seenSet!.add(k);
        } else {
          const ex = best.get(k);
          if (ex !== undefined && gs[ex]! <= ng) continue;
        }
        const nid = players.length;
        players.push(b);
        boxStates.push(nb);
        gs.push(ng);
        parents.push(id);
        pushDir.push(d);
        pushBox.push(b);
        if (!opts.greedy) best.set(k, nid);
        hpush(opts.greedy ? hn : ng + hn, hn, nid);
      }
    }
  }
  if (goalId < 0) return null;

  // Rebuild the move string: player walks to each push, then pushes.
  const pushes: { box: number; dir: number }[] = [];
  for (let t = goalId; t > 0; t = parents[t]!) pushes.push({ box: pushBox[t]!, dir: pushDir[t]! });
  pushes.reverse();
  let moves = '';
  let playerPos = entry;
  let curBoxes = startBoxes.slice();
  for (const { box, dir } of pushes) {
    const { dx, dy, ch } = DIRS[dir]!;
    const behind = box - (dy * g.w + dx);
    const path = bfsPath(g, roomCells, curBoxes, playerPos, behind, sc);
    if (path === null) return null;
    moves += path + ch;
    const bi = curBoxes.indexOf(box);
    curBoxes[bi] = box + (dy * g.w + dx);
    curBoxes.sort((a, b) => a - b);
    playerPos = box;
  }
  let endPlayer = playerPos;
  if (opts.requireExit) {
    const path = bfsPath(g, roomCells, curBoxes, playerPos, door, sc);
    if (path === null) return null;
    moves += path;
    endPlayer = door;
  }
  return { ok: true, moves, pushes: pushes.length, endPlayer };
}

interface SolvedRoom {
  door: number;
  entry: number;
  res: RoomResult;
  exitOptional: boolean;
}

// Solve one level: decompose into rooms, solve each, stitch with hallway
// transit moves. A room solved without an exit must go last.
function solveLevel(text: string, global: { used: number; cap: number }): string {
  const g = parse(text);
  const dist = allPairsDist(g);
  const sc = new Scratch();
  const { hallRow, rooms } = decompose(g);
  // Fallback for a level with no hallway/room structure at all: treat the
  // whole floor as one room, entered where the player stands. The generated
  // sets always decompose, but a malformed level must still get an answer.
  const infos =
    rooms.length === 0
      ? [
          {
            room: {
              cells: [...Array(g.n).keys()].filter((c) => !g.wall[c]),
              cellSet: new Set([...Array(g.n).keys()].filter((c) => !g.wall[c])),
              doors: [g.player],
              boxes: g.boxes.slice(),
              goals: [...g.goals],
            } as Room,
            door: g.player,
            entry: g.player,
          },
        ]
      : rooms
          .map((room) => {
            const door = room.doors[0] ?? g.player;
            const dx = door % g.w;
            const dy = (door / g.w) | 0;
            const entry = (dy < hallRow ? dy + 1 : dy - 1) * g.w + dx;
            return { room, door, entry };
          })
          .filter((x) => x.room.boxes.length > 0)
          .sort((a, b) => a.door % g.w - (b.door % g.w));
  if (infos.length === 0) return '';

  const solved: SolvedRoom[] = [];
  for (const { room, door, entry } of infos) {
    let res = searchRoom(g, room, dist, entry, sc, {
      budget: ASTAR_BUDGET,
      requireExit: true,
      greedy: false,
      global,
    });
    let exitOptional = false;
    if (res === null) {
      res = searchRoom(g, room, dist, entry, sc, {
        budget: ASTAR_BUDGET,
        requireExit: false,
        greedy: false,
        global,
      });
      exitOptional = true;
    }
    if (res === null) {
      res = searchRoom(g, room, dist, entry, sc, {
        budget: GREEDY_BUDGET,
        requireExit: true,
        greedy: true,
        global,
      });
      exitOptional = false;
    }
    if (res === null) {
      res = searchRoom(g, room, dist, entry, sc, {
        budget: GREEDY_BUDGET,
        requireExit: false,
        greedy: true,
        global,
      });
      exitOptional = true;
    }
    if (res === null) return ''; // this level is lost; keep the other levels
    solved.push({ door, entry, res, exitOptional });
  }
  // An exit-optional room goes last so the player never has to leave it.
  const optIdx = solved.findIndex((s) => s.exitOptional);
  if (optIdx >= 0) {
    const [last] = solved.splice(optIdx, 1);
    solved.push(last!);
  }

  const hallCells = new Set<number>();
  for (let x = 0; x < g.w; x++) {
    const c = hallRow * g.w + x;
    if (!g.wall[c]) hallCells.add(c);
  }
  let moves = '';
  let playerPos = g.player;
  for (const { entry, res } of solved) {
    hallCells.add(playerPos);
    const path = bfsPath(g, hallCells, [], playerPos, entry, sc);
    if (path === null) return '';
    moves += path;
    moves += res.moves;
    playerPos = res.endPlayer;
  }
  return moves;
}

const globalBudget = { used: 0, cap: GLOBAL_BUDGET };
for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    const text = readFileSync(join(I, id, 'level.txt'), 'utf8');
    moves = solveLevel(text, globalBudget);
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
