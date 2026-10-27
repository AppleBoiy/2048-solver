# 2048 best-move solver (4×4 and 5×5)

Expectimax solver in a single C file. Give it a board, it tells you the best move
(and the expected value of each of the four moves).

## Build

```sh
cc -O3 -DSIZE=4 -o solve4 solver.c -lm -lpthread
cc -O3 -DSIZE=5 -o solve5 solver.c -lm -lpthread
./solve4 test && ./solve5 test   # checks the table moves against a naive implementation
```

No dependencies beyond a C compiler (gcc or clang; 5×5 uses `unsigned __int128`).

## Use

Tiles are given row by row, top-left first. `0`, `.`, `_` or `-` mean empty.
Any other non-digit characters (`/`, `,`, `|`, newlines…) are just separators.

```sh
./solve4 best "0 0 2 2 / 0 4 8 16 / 2 2 32 64 / 4 8 128 256"
```
```
     .     .     2     2
     .     4     8    16
     2     2    32    64
     4     8   128   256
  up    :      1527822.9   <== best
  down  :        illegal
  left  :      1506212.8
  right :      1526606.2
Best move: up   (depth 6, 3117348 leaves, 242.4 ms)
```

5×5 works the same with 25 tiles:

```sh
./solve5 best --json "2 . . . . / . 4 . . . / . . 8 . 2 / . . . . . / 16 32 64 128 256"
# {"size":5,"best":"left","depth":6,"ms":146.3,"scores":{"up":...,"down":...,"left":...,"right":...}}
```

With no tiles on the command line, `best` reads boards from stdin (16 or 25 tiles each,
any layout) and answers each one — handy for piping from a screen reader / bot:

```sh
printf '2 2 0 0\n0 0 0 0\n0 0 0 0\n0 0 0 4\n' | ./solve4 best -j
```

Self-play benchmark:

```sh
./solve4 play                 # one full game, prints final board + score
./solve4 play -g 10 -s 42     # 10 games, seeded, prints max-tile distribution
./solve5 play -v              # print every move
```

Options: `-d N` fixed depth (default adaptive), `-j` JSON, `-g` games, `-s` seed,
`-v` verbose, `--single-thread`.

## How it works

- **Bitboard.** Each cell is a 4-bit exponent (0 = empty, 1 = 2, … 15 = 32768).
  4×4 fits in a `uint64_t`, 5×5 (100 bits) in an `unsigned __int128`.
- **Row tables.** All 16^N possible rows (65,536 for 4×4, 1,048,576 for 5×5) are
  precomputed once: result of sliding left/right, points scored, heuristic value.
  Up/down = transpose → row move → transpose.
- **Expectimax.** Max nodes are your 4 moves; chance nodes average over every
  empty cell × {2 with p = 0.9, 4 with p = 0.1}. Depth adapts to the number of
  distinct tiles (3 up to 8 on 4×4, up to 6 on 5×5); branches whose probability
  falls below 1e-4 are cut early. A transposition table caches chance nodes, and the
  four root moves are searched in parallel threads.
- **Heuristic** (per row and per column, summed): reward empty cells and adjacent
  equal tiles, penalise non-monotonic rows and large tile sums — the well-tested
  weights from nneonneo's 2048-AI, generalised to N-cell rows.

## Limits

- Max tile is 32768 (4-bit cells). Two 32768s won't merge.
- 5×5 uses the 4×4 heuristic weights; they play well but aren't tuned for 5×5.
