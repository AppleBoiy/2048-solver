/*
 * solver.c — 2048 best-move solver (expectimax) for 4x4 and 5x5 boards.
 *
 * Build:  cc -O3 -DSIZE=4 -o solve4 solver.c -lm -lpthread
 *         cc -O3 -DSIZE=5 -o solve5 solver.c -lm -lpthread
 *
 * Board encoding: one 4-bit nibble per cell holding the tile exponent
 * (0 = empty, 1 = 2, 2 = 4, ... 15 = 32768). Cell (r,c) lives at nibble
 * r*N + c. 4x4 fits in a uint64, 5x5 (100 bits) in an unsigned __int128.
 * Every possible row (16^N of them) is precomputed into lookup tables for
 * the left/right move result, the merge score, and the heuristic value.
 *
 * Search: expectimax. Max nodes = the player's 4 moves, chance nodes =
 * every empty cell x {2 (p=0.9), 4 (p=0.1)}. Branches whose cumulative
 * probability drops below CPROB_MIN are cut and scored heuristically,
 * and a transposition table caches chance-node values. The four
 * top-level moves are searched in parallel threads.
 */
#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef SIZE
#define SIZE 4
#endif
#define N SIZE
#define CELLS (N * N)
#define ROWBITS (4 * N)
#define ROWCOUNT (1u << ROWBITS)

#if N == 4
typedef uint64_t board_t;
#define MAX_AUTO_DEPTH 8
#elif N == 5
typedef unsigned __int128 board_t;
#define MAX_AUTO_DEPTH 6
#else
#error "SIZE must be 4 or 5"
#endif

#define ROWMASK ((board_t)(ROWCOUNT - 1))

/* ---- heuristic weights (nneonneo-style, generalised to N cells/row) ---- */
static const float SCORE_LOST_PENALTY = 200000.0f;
static const float SCORE_MONOTONICITY_POWER = 4.0f;
static const float SCORE_MONOTONICITY_WEIGHT = 47.0f;
static const float SCORE_SUM_POWER = 3.5f;
static const float SCORE_SUM_WEIGHT = 11.0f;
static const float SCORE_MERGES_WEIGHT = 700.0f;
static const float SCORE_EMPTY_WEIGHT = 270.0f;

static const float CPROB_MIN = 0.0001f;
static const int CACHE_DEPTH_LIMIT = 15;

enum { UP, DOWN, LEFT, RIGHT };
static const char *DIR_NAME[4] = {"up", "down", "left", "right"};

/* ---- lookup tables ---- */
static uint32_t *row_left_tab, *row_right_tab, *merge_pts_tab;
static float *heur_tab;

static inline uint32_t get_row(board_t b, int r) { return (uint32_t)((b >> (r * ROWBITS)) & ROWMASK); }
static inline int get_cell(board_t b, int i) { return (int)((b >> (4 * i)) & 0xF); }
static inline board_t set_cell(board_t b, int i, int v) {
    b &= ~((board_t)0xF << (4 * i));
    return b | ((board_t)v << (4 * i));
}

static uint32_t reverse_row(uint32_t row) {
    uint32_t r = 0;
    for (int i = 0; i < N; i++) r |= ((row >> (4 * i)) & 0xF) << (4 * (N - 1 - i));
    return r;
}

static void init_tables(void) {
    row_left_tab = malloc(sizeof(uint32_t) * ROWCOUNT);
    row_right_tab = malloc(sizeof(uint32_t) * ROWCOUNT);
    merge_pts_tab = malloc(sizeof(uint32_t) * ROWCOUNT);
    heur_tab = malloc(sizeof(float) * ROWCOUNT);
    if (!row_left_tab || !row_right_tab || !merge_pts_tab || !heur_tab) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (uint32_t row = 0; row < ROWCOUNT; row++) {
        int line[N];
        for (int i = 0; i < N; i++) line[i] = (row >> (4 * i)) & 0xF;

        /* heuristic */
        float sum = 0;
        int empty = 0, merges = 0, prev = 0, counter = 0;
        for (int i = 0; i < N; i++) {
            int rank = line[i];
            sum += powf((float)rank, SCORE_SUM_POWER);
            if (rank == 0) {
                empty++;
            } else {
                if (prev == rank) counter++;
                else if (counter > 0) { merges += 1 + counter; counter = 0; }
                prev = rank;
            }
        }
        if (counter > 0) merges += 1 + counter;
        float mono_l = 0, mono_r = 0;
        for (int i = 1; i < N; i++) {
            float a = powf((float)line[i - 1], SCORE_MONOTONICITY_POWER);
            float b = powf((float)line[i], SCORE_MONOTONICITY_POWER);
            if (line[i - 1] > line[i]) mono_l += a - b;
            else mono_r += b - a;
        }
        heur_tab[row] = SCORE_LOST_PENALTY + SCORE_EMPTY_WEIGHT * empty + SCORE_MERGES_WEIGHT * merges -
                        SCORE_MONOTONICITY_WEIGHT * (mono_l < mono_r ? mono_l : mono_r) - SCORE_SUM_WEIGHT * sum;

        /* move left: slide toward index 0, merge each pair once */
        int out[N] = {0}, k = 0, last = 0;
        uint32_t pts = 0;
        for (int i = 0; i < N; i++) {
            int v = line[i];
            if (!v) continue;
            if (last && v == last && v < 15) { /* 32768+32768 can't be represented: no merge */
                out[k - 1] = v + 1;
                pts += 1u << (v + 1);
                last = 0;
            } else {
                out[k++] = v;
                last = v;
            }
        }
        uint32_t res = 0;
        for (int i = 0; i < N; i++) res |= (uint32_t)out[i] << (4 * i);
        row_left_tab[row] = res;
        merge_pts_tab[row] = pts;
    }
    /* right = reverse(left(reverse(row))) */
    for (uint32_t row = 0; row < ROWCOUNT; row++)
        row_right_tab[row] = reverse_row(row_left_tab[reverse_row(row)]);
}

/* ---- board primitives ---- */
#if N == 4
static inline board_t transpose(board_t x) {
    board_t a1 = x & 0xF0F00F0FF0F00F0FULL;
    board_t a2 = x & 0x0000F0F00000F0F0ULL;
    board_t a3 = x & 0x0F0F00000F0F0000ULL;
    board_t a = a1 | (a2 << 12) | (a3 >> 12);
    board_t b1 = a & 0xFF00FF0000FF00FFULL;
    board_t b2 = a & 0x00FF00FF00000000ULL;
    board_t b3 = a & 0x00000000FF00FF00ULL;
    return b1 | (b2 >> 24) | (b3 << 24);
}
#else
static inline board_t transpose(board_t b) {
    board_t t = 0;
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) t |= ((b >> (4 * (r * N + c))) & 0xF) << (4 * (c * N + r));
    return t;
}
#endif

static inline board_t apply_rows(board_t b, const uint32_t *tab) {
    board_t r = 0;
    for (int i = 0; i < N; i++) r |= (board_t)tab[get_row(b, i)] << (i * ROWBITS);
    return r;
}

static inline board_t execute_move(int dir, board_t b) {
    switch (dir) {
    case LEFT: return apply_rows(b, row_left_tab);
    case RIGHT: return apply_rows(b, row_right_tab);
    case UP: return transpose(apply_rows(transpose(b), row_left_tab));
    default: return transpose(apply_rows(transpose(b), row_right_tab));
    }
}

static uint32_t move_points(int dir, board_t b) {
    board_t src = (dir == UP || dir == DOWN) ? transpose(b) : b;
    uint32_t p = 0;
    for (int i = 0; i < N; i++) p += merge_pts_tab[get_row(src, i)];
    return p;
}

static inline int count_empty(board_t b) {
    int n = 0;
    for (int i = 0; i < CELLS; i++) n += ((b >> (4 * i)) & 0xF) == 0;
    return n;
}

static int count_distinct_tiles(board_t b) {
    uint32_t seen = 0;
    for (int i = 0; i < CELLS; i++) seen |= 1u << get_cell(b, i);
    seen >>= 1;
    int n = 0;
    while (seen) { seen &= seen - 1; n++; }
    return n;
}

static int max_rank(board_t b) {
    int m = 0;
    for (int i = 0; i < CELLS; i++) { int v = get_cell(b, i); if (v > m) m = v; }
    return m;
}

static inline float score_heur_board(board_t b) {
    float s = 0;
    board_t t = transpose(b);
    for (int i = 0; i < N; i++) s += heur_tab[get_row(b, i)] + heur_tab[get_row(t, i)];
    return s;
}

/* ---- transposition table ---- */
#define TT_BITS 20
#define TT_SIZE (1u << TT_BITS)
typedef struct {
    board_t key;
    float score;
    uint32_t gen;
    uint8_t depth;
} tt_entry;

typedef struct {
    tt_entry *tt;
    uint32_t gen;
    int depth_limit, curdepth, maxdepth;
    unsigned long evals, hits;
} search_t;

static inline uint32_t hash_board(board_t b) {
    uint64_t x = (uint64_t)b;
#if N == 5
    x ^= (uint64_t)(b >> 64) * 0x9E3779B97F4A7C15ULL;
#endif
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x & (TT_SIZE - 1);
}

/* ---- expectimax ---- */
static float score_max_node(search_t *s, board_t b, float cprob);

static float score_chance_node(search_t *s, board_t b, float cprob) {
    if (cprob < CPROB_MIN || s->curdepth >= s->depth_limit) {
        if (s->curdepth > s->maxdepth) s->maxdepth = s->curdepth;
        s->evals++;
        return score_heur_board(b);
    }
    tt_entry *e = NULL;
    if (s->curdepth < CACHE_DEPTH_LIMIT) {
        e = &s->tt[hash_board(b)];
        if (e->gen == s->gen && e->key == b && e->depth <= s->curdepth) {
            s->hits++;
            return e->score;
        }
    }
    int empty = count_empty(b);
    cprob /= empty;
    float res = 0;
    for (int i = 0; i < CELLS; i++) {
        if (get_cell(b, i)) continue;
        res += score_max_node(s, set_cell(b, i, 1), cprob * 0.9f) * 0.9f;
        res += score_max_node(s, set_cell(b, i, 2), cprob * 0.1f) * 0.1f;
    }
    res /= empty;
    if (e) {
        e->key = b; e->score = res; e->gen = s->gen; e->depth = (uint8_t)s->curdepth;
    }
    return res;
}

static float score_max_node(search_t *s, board_t b, float cprob) {
    float best = 0;
    s->curdepth++;
    for (int d = 0; d < 4; d++) {
        board_t nb = execute_move(d, b);
        if (nb != b) {
            float v = score_chance_node(s, nb, cprob);
            if (v > best) best = v;
        }
    }
    s->curdepth--;
    return best;
}

/* ---- top level (one thread per move) ---- */
typedef struct {
    board_t board;
    int dir, depth_limit;
    int valid;
    float score;
    search_t s;
} job_t;

static tt_entry *thread_tt[4];
static uint32_t thread_gen[4];

static void *run_job(void *arg) {
    job_t *j = arg;
    board_t nb = execute_move(j->dir, j->board);
    j->valid = nb != j->board;
    j->score = 0;
    if (!j->valid) return NULL;
    memset(&j->s, 0, sizeof j->s);
    j->s.tt = thread_tt[j->dir];
    j->s.gen = ++thread_gen[j->dir];
    j->s.depth_limit = j->depth_limit;
    /* small bonus so equal-valued moves don't tie at 0 when everything loses */
    j->score = score_chance_node(&j->s, nb, 1.0f) + 1e-6f;
    return NULL;
}

typedef struct {
    int best;          /* -1 if no move possible */
    int valid[4];
    float score[4];
    int depth;
    unsigned long evals;
    double ms;
} result_t;

static int g_fixed_depth = 0;
static int g_threads = 1;

static result_t find_best_move(board_t b) {
    result_t r;
    memset(&r, 0, sizeof r);
    int depth = g_fixed_depth;
    if (!depth) {
        depth = count_distinct_tiles(b) - 2;
        if (depth < 3) depth = 3;
        if (depth > MAX_AUTO_DEPTH) depth = MAX_AUTO_DEPTH;
    }
    r.depth = depth;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    job_t jobs[4];
    pthread_t th[4];
    for (int d = 0; d < 4; d++) {
        jobs[d].board = b; jobs[d].dir = d; jobs[d].depth_limit = depth;
        if (g_threads) pthread_create(&th[d], NULL, run_job, &jobs[d]);
        else run_job(&jobs[d]);
    }
    r.best = -1;
    float best = -1;
    for (int d = 0; d < 4; d++) {
        if (g_threads) pthread_join(th[d], NULL);
        r.valid[d] = jobs[d].valid;
        r.score[d] = jobs[d].score;
        if (jobs[d].valid) r.evals += jobs[d].s.evals;
        if (jobs[d].valid && jobs[d].score > best) { best = jobs[d].score; r.best = d; }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    r.ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    return r;
}

/* ---- I/O ---- */
static void print_board(FILE *f, board_t b) {
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            int v = get_cell(b, r * N + c);
            if (v) fprintf(f, "%6u", 1u << v);
            else fprintf(f, "%6s", ".");
        }
        fputc('\n', f);
    }
}

/* Reads CELLS tiles from text. Empty = 0 . _ -  ; separators = anything else non-digit. */
static int parse_board(const char **pp, board_t *out) {
    const char *p = *pp;
    board_t b = 0;
    int n = 0;
    while (*p && n < CELLS) {
        if (isdigit((unsigned char)*p)) {
            unsigned long v = strtoul(p, (char **)&p, 10);
            int rank = 0;
            if (v != 0) {
                if (v & (v - 1) || v < 2 || v > 32768) {
                    fprintf(stderr, "invalid tile %lu (must be 0 or a power of 2 from 2 to 32768)\n", v);
                    return -1;
                }
                while ((1ul << rank) != v) rank++;
            }
            b = set_cell(b, n++, rank);
        } else if (*p == '.' || *p == '_' || *p == '-') {
            b = set_cell(b, n++, 0);
            p++;
        } else {
            p++;
        }
    }
    *pp = p;
    if (n == 0) return 0;
    if (n < CELLS) {
        fprintf(stderr, "need %d tiles for a %dx%d board, got %d\n", CELLS, N, N, n);
        return -1;
    }
    *out = b;
    return 1;
}

static void report(board_t b, result_t r, int json) {
    if (json) {
        printf("{\"size\":%d,\"best\":%s%s%s,\"depth\":%d,\"ms\":%.1f,\"scores\":{", N,
               r.best < 0 ? "" : "\"", r.best < 0 ? "null" : DIR_NAME[r.best], r.best < 0 ? "" : "\"", r.depth, r.ms);
        for (int d = 0; d < 4; d++) {
            if (r.valid[d]) printf("\"%s\":%.2f", DIR_NAME[d], r.score[d]);
            else printf("\"%s\":null", DIR_NAME[d]);
            if (d < 3) putchar(',');
        }
        printf("}}\n");
        fflush(stdout);
        return;
    }
    print_board(stdout, b);
    for (int d = 0; d < 4; d++) {
        if (r.valid[d]) printf("  %-5s : %14.1f%s\n", DIR_NAME[d], r.score[d], d == r.best ? "   <== best" : "");
        else printf("  %-5s : %14s\n", DIR_NAME[d], "illegal");
    }
    if (r.best < 0) printf("No legal moves - game over.\n");
    else printf("Best move: %s   (depth %d, %lu leaves, %.1f ms)\n\n", DIR_NAME[r.best], r.depth, r.evals, r.ms);
    fflush(stdout);
}

static char *read_all(FILE *f) {
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        if (len + 1 >= cap) buf = realloc(buf, cap *= 2);
        buf[len++] = (char)ch;
    }
    buf[len] = 0;
    return buf;
}

/* ---- self-play ---- */
static uint64_t rng_state = 88172645463325252ULL;
static inline uint64_t rng(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state;
}

static board_t spawn_tile(board_t b) {
    int empty = count_empty(b);
    if (!empty) return b;
    int idx = (int)(rng() % (uint64_t)empty);
    int rank = (rng() % 10) == 0 ? 2 : 1;
    for (int i = 0; i < CELLS; i++) {
        if (get_cell(b, i)) continue;
        if (idx-- == 0) return set_cell(b, i, rank);
    }
    return b;
}

static void play(int games, int verbose) {
    int hist[16] = {0};
    double total_score = 0;
    for (int g = 0; g < games; g++) {
        board_t b = spawn_tile(spawn_tile(0));
        unsigned long moves = 0, score = 0;
        double ms = 0;
        for (;;) {
            result_t r = find_best_move(b);
            if (r.best < 0) break;
            ms += r.ms;
            score += move_points(r.best, b);
            b = spawn_tile(execute_move(r.best, b));
            moves++;
            if (verbose) {
                printf("move %lu: %s  score %lu  (depth %d, %.1f ms)\n", moves, DIR_NAME[r.best], score, r.depth, r.ms);
                print_board(stdout, b);
            } else if (moves % 500 == 0) {
                fprintf(stderr, "  game %d: %lu moves, score %lu, max %u\r", g + 1, moves, score, 1u << max_rank(b));
            }
        }
        int mr = max_rank(b);
        hist[mr]++;
        total_score += score;
        printf("game %d: score %lu, max tile %u, %lu moves, %.2f moves/s\n", g + 1, score, 1u << mr, moves,
               ms > 0 ? moves / (ms / 1000.0) : 0);
        print_board(stdout, b);
        fflush(stdout);
    }
    if (games > 1) {
        printf("\n%d games, mean score %.0f\nmax tile reached (at least):\n", games, total_score / games);
        int cum = 0;
        for (int r = 15; r >= 1; r--) {
            cum += hist[r];
            if (cum) printf("  %6u : %5.1f%%\n", 1u << r, 100.0 * cum / games);
        }
    }
}

/* ---- self-test: table moves vs. naive grid moves ---- */
static board_t naive_move(int dir, board_t b) {
    int g[N][N], o[N][N];
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) g[r][c] = get_cell(b, r * N + c);
    for (int line = 0; line < N; line++) {
        int in[N], out[N] = {0}, k = 0, last = 0;
        for (int i = 0; i < N; i++) {
            switch (dir) {
            case LEFT: in[i] = g[line][i]; break;
            case RIGHT: in[i] = g[line][N - 1 - i]; break;
            case UP: in[i] = g[i][line]; break;
            default: in[i] = g[N - 1 - i][line]; break;
            }
        }
        for (int i = 0; i < N; i++) {
            if (!in[i]) continue;
            if (last && in[i] == last && last < 15) { out[k - 1]++; last = 0; }
            else { out[k++] = in[i]; last = in[i]; }
        }
        for (int i = 0; i < N; i++) {
            switch (dir) {
            case LEFT: o[line][i] = out[i]; break;
            case RIGHT: o[line][N - 1 - i] = out[i]; break;
            case UP: o[i][line] = out[i]; break;
            default: o[N - 1 - i][line] = out[i]; break;
            }
        }
    }
    board_t r = 0;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) r = set_cell(r, y * N + x, o[y][x]);
    return r;
}

static int self_test(void) {
    int fails = 0;
    for (int t = 0; t < 200000; t++) {
        board_t b = 0;
        for (int i = 0; i < CELLS; i++) {
            int v = (int)(rng() % 6);
            b = set_cell(b, i, v ? (int)(rng() % 16) * (v > 2) + (v <= 2 ? 0 : 0) : 0);
            if (rng() % 3 == 0) b = set_cell(b, i, (int)(rng() % 4)); /* encourage merges */
        }
        for (int d = 0; d < 4; d++) {
            if (execute_move(d, b) != naive_move(d, b)) {
                if (fails++ < 3) {
                    printf("mismatch dir %s on board:\n", DIR_NAME[d]);
                    print_board(stdout, b);
                }
            }
        }
        if (transpose(transpose(b)) != b) fails++;
    }
    printf("self-test %dx%d: %s (%d failures)\n", N, N, fails ? "FAILED" : "ok", fails);
    return fails ? 1 : 0;
}

/* ---- main ---- */
static void usage(const char *prog) {
    fprintf(stderr,
            "2048 best-move solver for %dx%d boards (expectimax)\n\n"
            "usage:\n"
            "  %s best [opts] <%d tiles...>   best move for one board (row by row, 0 or . = empty)\n"
            "  %s best [opts]                 read boards from stdin, %d tiles each, answer each one\n"
            "  %s play [opts]                 let the solver play full games (benchmark)\n"
            "  %s test                        verify move tables against a naive implementation\n\n"
            "options:\n"
            "  -d, --depth D    fixed search depth in moves (default: adaptive, 3..%d)\n"
            "  -j, --json       machine-readable output (best)\n"
            "  -g, --games G    number of games (play, default 1)\n"
            "  -s, --seed S     RNG seed (play)\n"
            "  -v, --verbose    print every move (play)\n"
            "  --single-thread  don't search the 4 moves in parallel\n\n"
            "example:\n  %s best \"0 0 2 2 / 0 4 8 16 / 2 2 32 64 / 4 8 128 256%s\"\n",
            N, N, prog, CELLS, prog, CELLS, prog, prog, MAX_AUTO_DEPTH, prog, N == 5 ? " / ... (5 per row)" : "");
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    const char *cmd = argv[1];
    int json = 0, games = 1, verbose = 0;
    uint64_t seed = (uint64_t)time(NULL);
    size_t tlen = 1;
    char *tiles = calloc(1, 1);
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if ((!strcmp(a, "-d") || !strcmp(a, "--depth")) && i + 1 < argc) g_fixed_depth = atoi(argv[++i]);
        else if (!strcmp(a, "-j") || !strcmp(a, "--json")) json = 1;
        else if ((!strcmp(a, "-g") || !strcmp(a, "--games")) && i + 1 < argc) games = atoi(argv[++i]);
        else if ((!strcmp(a, "-s") || !strcmp(a, "--seed")) && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) verbose = 1;
        else if (!strcmp(a, "--single-thread")) g_threads = 0;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else {
            tlen += strlen(a) + 1;
            tiles = realloc(tiles, tlen);
            strcat(tiles, a);
            strcat(tiles, " ");
        }
    }
    rng_state = seed * 2654435761ULL + 1;
    init_tables();
    for (int d = 0; d < 4; d++) {
        thread_tt[d] = calloc(TT_SIZE, sizeof(tt_entry));
        if (!thread_tt[d]) { fprintf(stderr, "out of memory\n"); return 1; }
    }

    if (!strcmp(cmd, "test")) return self_test();
    if (!strcmp(cmd, "play")) { play(games, verbose); return 0; }
    if (!strcmp(cmd, "best")) {
        const char *src = tiles;
        char *buf = NULL;
        if (strspn(tiles, " ") == strlen(tiles)) src = buf = read_all(stdin);
        int any = 0;
        for (;;) {
            board_t b;
            int rc = parse_board(&src, &b);
            if (rc < 0) return 1;
            if (rc == 0) break;
            any = 1;
            report(b, find_best_move(b), json);
        }
        free(buf);
        if (!any) { fprintf(stderr, "no board given\n"); usage(argv[0]); return 2; }
        return 0;
    }
    usage(argv[0]);
    return 2;
}
