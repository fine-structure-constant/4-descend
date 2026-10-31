// Build an exact best-move table for every non-terminal position with <= 12 discs.
//
// Generation-time tool only.  It links Pascal Pons' Connect 4 solver
// (AGPL v3, https://github.com/PascalPons/connect4) and its published 7x6.book.
// The resulting table is plain data: for each symmetric position, one optimal
// move and the exact score.  The runtime bot only reads that data.
//
// Build (Pons sources must be available; the local copy adds a fromBits()
// factory to Position.hpp and lives outside this repository):
//   g++ -O3 -DNDEBUG --std=c++17 -I/tmp/mbwork -pthread \
//       tools/movebook_build.cpp /tmp/mbwork/Solver.cpp -o /tmp/movebook_build
//
// Output format (little endian):
//   char[4] "MBK2", uint32 N,
//   uint32 keys[N] (grouped by disc count, sorted within each level),
//   uint8 moves[N], int8 scores[N] (Pons convention, side to move).
// Keys of different depths interleave, so run tools/sort_movebook.py before
// binary-searching: the runtime needs one strictly increasing key array.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "Solver.hpp"

using namespace GameSolver::Connect4;
using Clock = std::chrono::steady_clock;

static const uint64_t BOTTOM = [] { uint64_t b = 0; for (int c = 0; c < 7; c++) b |= 1ull << (7 * c); return b; }();
static const uint64_t BOARD = [] { uint64_t b = 0; for (int c = 0; c < 7; c++) b |= 63ull << (7 * c); return b; }();
static const uint64_t COLS[7] = {63ull, 63ull << 7, 63ull << 14, 63ull << 21,
                                 63ull << 28, 63ull << 35, 63ull << 42};

static inline uint64_t possible(uint64_t mask) { return (mask + BOTTOM) & BOARD; }

static inline bool alignment(uint64_t bits) {
  for (int s : {1, 7, 6, 8}) {
    uint64_t pairs = bits & (bits >> s);
    if (pairs & (pairs >> (2 * s))) return true;
  }
  return false;
}

static inline uint64_t key3_fwd(uint64_t own, uint64_t mask) {
  uint64_t key = 0;
  for (int c = 0; c < 7; c++) {
    uint64_t bits = (mask >> (7 * c)) & 0x7f;
    while (bits) {
      uint64_t low = bits & (~bits + 1);
      int r = __builtin_ctzll(low);
      key = key * 3 + (((own >> (7 * c + r)) & 1) ? 1 : 2);
      bits ^= low;
    }
    key *= 3;
  }
  return key;
}

static inline uint64_t key3_rev(uint64_t own, uint64_t mask) {
  uint64_t key = 0;
  for (int c = 6; c >= 0; c--) {
    uint64_t bits = (mask >> (7 * c)) & 0x7f;
    while (bits) {
      uint64_t low = bits & (~bits + 1);
      int r = __builtin_ctzll(low);
      key = key * 3 + (((own >> (7 * c + r)) & 1) ? 1 : 2);
      bits ^= low;
    }
    key *= 3;
  }
  return key;
}

static inline uint64_t mirror_bits(uint64_t x) {
  uint64_t y = 0;
  for (int c = 0; c < 7; c++) y |= ((x >> (7 * c)) & 0x7f) << (7 * (6 - c));
  return y;
}

struct Rec {
  uint32_t key;
  uint64_t own;
  uint64_t mask;
};

static std::vector<Rec> level[13];
static std::vector<uint32_t> level_keys[13];
static std::vector<uint8_t> level_mask[13];
static std::vector<int8_t> level_score[13];

static void dfs(uint64_t own, uint64_t mask, int d, std::vector<uint64_t>& seen) {
  uint64_t f = key3_fwd(own, mask), r = key3_rev(own, mask);
  uint32_t k = (uint32_t)((f < r ? f : r) / 3);
  uint64_t word = k >> 6, bit = 1ull << (k & 63);
  if (seen[word] & bit) return;
  seen[word] |= bit;
  uint64_t co = own, cm = mask;
  if (r < f) { co = mirror_bits(own); cm = mirror_bits(mask); }
  level[d].push_back({k, co, cm});
  if (d == 12) return;
  uint64_t moves = possible(mask);
  for (int c = 0; c < 7; c++) {
    uint64_t b = moves & COLS[c];
    if (!b || alignment(own | b)) continue;
    dfs(own ^ mask, mask | b, d + 1, seen);
  }
}

int main(int argc, char** argv) {
  std::string book = "/tmp/7x6.book";
  std::string out = "movebook12.bin";
  int threads = 28;
  long long limit = -1;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--book" && i + 1 < argc) book = argv[++i];
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--threads" && i + 1 < argc) threads = atoi(argv[++i]);
    else if (a == "--limit" && i + 1 < argc) limit = atoll(argv[++i]);
  }

  auto t_start = Clock::now();
  std::vector<uint64_t> seen((387420489ull + 63) / 64, 0);
  dfs(0, 0, 0, seen);
  std::printf("[enumerate] %.1fs\n", std::chrono::duration<double>(Clock::now() - t_start).count());
  uint64_t total = 0;
  for (int d = 0; d <= 12; d++) {
    std::sort(level[d].begin(), level[d].end(), [](const Rec& a, const Rec& b) { return a.key < b.key; });
    level_keys[d].resize(level[d].size());
    for (size_t i = 0; i < level[d].size(); i++) level_keys[d][i] = level[d][i].key;
    total += level[d].size();
    std::printf("[enumerate] depth %2d: %9zu\n", d, level[d].size());
  }
  std::printf("[enumerate] total %zu non-terminal positions <= 12 discs\n", (size_t)total);

  // Global uniqueness: key3 must identify a position regardless of depth.
  {
    std::vector<uint32_t> all;
    all.reserve(total);
    for (int d = 0; d <= 12; d++) all.insert(all.end(), level_keys[d].begin(), level_keys[d].end());
    std::sort(all.begin(), all.end());
    for (size_t i = 1; i < all.size(); i++)
      if (all[i] == all[i - 1]) { std::printf("[error] duplicate key %u\n", all[i]); return 1; }
    std::printf("[enumerate] keys unique across depths\n");
  }

  // Exact values and optimal moves for all 12-disc positions.
  size_t n12 = level[12].size();
  size_t analyze_count = (limit > 0 && (size_t)limit < n12) ? (size_t)limit : n12;
  level_mask[12].assign(n12, 0);
  level_score[12].assign(n12, 0);
  std::atomic<size_t> next{0};
  std::atomic<size_t> done{0};
  auto t_analyze = Clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; t++) {
    pool.emplace_back([&]() {
      Solver solver;
      solver.loadBook(book);
      size_t i;
      while ((i = next.fetch_add(1, std::memory_order_relaxed)) < analyze_count) {
        const Rec& r = level[12][i];
        uint64_t moves = possible(r.mask);
        uint8_t mask = 0;
        int best = 0;
        for (int c = 0; c < 7; c++) {
          uint64_t b = moves & COLS[c];
          if (b && alignment(r.own | b)) mask |= (uint8_t)(1 << c);
        }
        if (mask) {
          best = (43 - 12) / 2;  // only immediate wins can score this at 12 discs
        } else {
          Position P = Position::fromBits(r.own, r.mask);
          std::vector<int> scores = solver.analyze(P, false);
          best = -1000;
          for (int c = 0; c < 7; c++) {
            if (scores[c] == Solver::INVALID_MOVE) continue;
            if (scores[c] > best) { best = scores[c]; mask = (uint8_t)(1 << c); }
            else if (scores[c] == best) mask |= (uint8_t)(1 << c);
          }
        }
        level_mask[12][i] = mask;
        level_score[12][i] = (int8_t)best;
        size_t d = ++done;
        if (d % 100000 == 0) {
          double s = std::chrono::duration<double>(Clock::now() - t_analyze).count();
          std::printf("[analyze] %zu / %zu  %.1fs  eta %.0fs\n", d, analyze_count, s,
                      s / d * (analyze_count - d));
          std::fflush(stdout);
        }
      }
    });
  }
  for (auto& th : pool) th.join();
  std::printf("[analyze] done %.1fs\n", std::chrono::duration<double>(Clock::now() - t_analyze).count());

  // Bottom-up exact values for <= 11 discs.
  auto child_value = [](int d, uint64_t own, uint64_t mask) -> int {
    uint64_t f = key3_fwd(own, mask), r = key3_rev(own, mask);
    uint32_t k = (uint32_t)((f < r ? f : r) / 3);
    const auto& keys = level_keys[d];
    auto it = std::lower_bound(keys.begin(), keys.end(), k);
    if (it == keys.end() || *it != k) {
      std::printf("[error] missing child key %u at depth %d\n", k, d);
      std::exit(2);
    }
    return level_score[d][it - keys.begin()];
  };
  auto t_derive = Clock::now();
  for (int d = 11; d >= 0; d--) {
    size_t n = level[d].size();
    level_mask[d].assign(n, 0);
    level_score[d].assign(n, 0);
    for (size_t i = 0; i < n; i++) {
      const Rec& r = level[d][i];
      uint8_t mask = 0;
      uint64_t moves = possible(r.mask);
      for (int c = 0; c < 7; c++) {
        uint64_t b = moves & COLS[c];
        if (b && alignment(r.own | b)) mask |= (uint8_t)(1 << c);
      }
      int best;
      if (mask) {
        best = (43 - d) / 2;
      } else {
        best = -1000;
        for (int c = 0; c < 7; c++) {
          uint64_t b = moves & COLS[c];
          if (!b) continue;
          uint64_t cm = r.mask | b, co = r.own ^ r.mask;
          int v = (cm == BOARD) ? 0 : -child_value(d + 1, co, cm);
          if (v > best) { best = v; mask = (uint8_t)(1 << c); }
          else if (v == best) mask |= (uint8_t)(1 << c);
        }
      }
      if (!mask) { std::printf("[error] no move at depth %d index %zu\n", d, i); return 3; }
      level_mask[d][i] = mask;
      level_score[d][i] = (int8_t)best;
    }
    std::printf("[derive] depth %2d done\n", d);
    std::fflush(stdout);
  }
  std::printf("[derive] %.1fs\n", std::chrono::duration<double>(Clock::now() - t_derive).count());

  // Cross-check derived exact scores against the solver (book covers <= 11).
  if (limit < 0) {
    int checked = 0, bad = 0;
    std::mt19937_64 rng(20261030);
    Solver solver;
    solver.loadBook(book);
    for (int d = 0; d <= 11 && checked < 20000; d++) {
      std::uniform_int_distribution<size_t> pick(0, level[d].size() - 1);
      for (int i = 0; i < 20000 / 12; i++) {
        const Rec& r = level[d][pick(rng)];
        Position P = Position::fromBits(r.own, r.mask);
        int exact = solver.solve(P, false);
        size_t idx = std::lower_bound(level_keys[d].begin(), level_keys[d].end(), r.key) - level_keys[d].begin();
        if (level_score[d][idx] != exact) {
          if (bad < 10) std::printf("[check] depth %d key %u derived %d solver %d\n",
                                    d, r.key, (int)level_score[d][idx], exact);
          bad++;
        }
        checked++;
      }
    }
    std::printf("[check] derived vs solver: %d checked, %d mismatches\n", checked, bad);
  }

  // Write table.
  std::vector<uint32_t> keys;
  std::vector<uint8_t> moves;
  std::vector<int8_t> scores;
  keys.reserve(total); moves.reserve(total); scores.reserve(total);
  for (int d = 0; d <= 12; d++) {
    keys.insert(keys.end(), level_keys[d].begin(), level_keys[d].end());
    moves.insert(moves.end(), level_mask[d].begin(), level_mask[d].end());
    scores.insert(scores.end(), level_score[d].begin(), level_score[d].end());
  }
  FILE* fp = std::fopen(out.c_str(), "wb");
  if (!fp) { std::printf("[error] cannot open %s\n", out.c_str()); return 4; }
  uint32_t n32 = (uint32_t)keys.size();
  std::fwrite("MBK2", 1, 4, fp);
  std::fwrite(&n32, 4, 1, fp);
  std::fwrite(keys.data(), 4, keys.size(), fp);
  std::fwrite(moves.data(), 1, moves.size(), fp);
  std::fwrite(scores.data(), 1, scores.size(), fp);
  std::fclose(fp);
  std::printf("[write] %s: %zu records, %.1f MiB, total %.1fs\n", out.c_str(), keys.size(),
              (8.0 + 6.0 * keys.size()) / (1024 * 1024),
              std::chrono::duration<double>(Clock::now() - t_start).count());
  return 0;
}
