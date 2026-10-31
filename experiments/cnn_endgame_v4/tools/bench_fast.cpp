// Benchmark: enumerate all non-terminal positions up to 12 plies,
// then time Solver::analyze on a sample of 12-ply positions.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "Solver.hpp"

using namespace GameSolver::Connect4;

static const uint64_t BOTTOM = []{ uint64_t b=0; for(int c=0;c<7;c++) b |= 1ull<<(7*c); return b; }();
static const uint64_t BOARD = []{ uint64_t b=0; for(int c=0;c<7;c++) b |= 63ull<<(7*c); return b; }();

static inline uint64_t possible(uint64_t mask) { return (mask + BOTTOM) & BOARD; }

static inline bool alignment(uint64_t bits) {
  for (int s : {1,7,6,8}) {
    uint64_t pairs = bits & (bits >> s);
    if (pairs & (pairs >> (2*s))) return true;
  }
  return false;
}

static inline uint64_t key3_fwd(uint64_t own, uint64_t mask) {
  uint64_t key = 0;
  for (int c = 0; c < 7; c++) {
    uint64_t bits = (mask >> (7*c)) & 0x7f;
    while (bits) {
      uint64_t low = bits & (~bits + 1);
      int r = __builtin_ctzll(low);
      key = key*3 + (((own >> (7*c+r)) & 1) ? 1 : 2);
      bits ^= low;
    }
    key *= 3;
  }
  return key;
}

static inline uint64_t key3_rev(uint64_t own, uint64_t mask) {
  uint64_t key = 0;
  for (int c = 6; c >= 0; c--) {
    uint64_t bits = (mask >> (7*c)) & 0x7f;
    while (bits) {
      uint64_t low = bits & (~bits + 1);
      int r = __builtin_ctzll(low);
      key = key*3 + (((own >> (7*c+r)) & 1) ? 1 : 2);
      bits ^= low;
    }
    key *= 3;
  }
  return key;
}

static inline uint64_t mirror_bits(uint64_t x) {
  uint64_t y = 0;
  for (int c = 0; c < 7; c++) y |= ((x >> (7*c)) & 0x7f) << (7*(6-c));
  return y;
}

struct Pos { uint64_t own, mask; };
struct Depths {
  std::vector<std::vector<Pos>> pos;
};

static void dfs(uint64_t own, uint64_t mask, int d, Depths& out, std::vector<uint64_t>& seen) {
  uint64_t f = key3_fwd(own, mask), r = key3_rev(own, mask);
  uint64_t k = std::min(f, r) / 3;
  uint64_t word = k >> 6, bit = 1ull << (k & 63);
  if (seen[word] & bit) return;
  seen[word] |= bit;
  uint64_t co = own, cm = mask;
  if (r < f) { co = mirror_bits(own); cm = mirror_bits(mask); }
  out.pos[d].push_back({co, cm});
  if (d == 12) return;
  uint64_t moves = possible(mask);
  for (int c = 0; c < 7; c++) {
    uint64_t b = moves & (63ull << (7*c));
    if (!b) continue;
    if (alignment(own | b)) continue;
    dfs(own ^ mask, mask | b, d+1, out, seen);
  }
}

int main(int argc, char** argv) {
  const char* book = argc > 1 ? argv[1] : "/tmp/7x6.book";
  int samples = argc > 2 ? atoi(argv[2]) : 200;
  uint64_t seed = argc > 3 ? strtoull(argv[3], nullptr, 10) : 12345;

  Depths D;
  D.pos.resize(13);
  std::vector<uint64_t> seen((387420489ull + 63)/64, 0);
  auto t0 = std::chrono::steady_clock::now();
  dfs(0, 0, 0, D, seen);
  auto t1 = std::chrono::steady_clock::now();
  double enum_s = std::chrono::duration<double>(t1-t0).count();
  uint64_t total = 0;
  for (int d = 0; d <= 12; d++) { total += D.pos[d].size(); std::printf("depth %2d canonical %9zu\n", d, D.pos[d].size()); }
  std::printf("total %llu, enum %.2fs\n", (unsigned long long)total, enum_s);

  Solver solver;
  auto tb0 = std::chrono::steady_clock::now();
  solver.loadBook(book);
  auto tb1 = std::chrono::steady_clock::now();
  std::printf("book load %.2fs\n", std::chrono::duration<double>(tb1-tb0).count());

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> pick(0, D.pos[12].size()-1);
  double max_t = 0, sum_t = 0;
  unsigned long long nodes0 = solver.getNodeCount();
  int checked = 0;
  for (int i=0; i<samples; i++) {
    Pos p=D.pos[12][pick(rng)];
    Position P=Position::fromBits(p.own,p.mask);
    auto t=std::chrono::steady_clock::now();
    int best=solver.solve(P,false); unsigned mask=0;
    for(int c=0;c<7;c++) if(P.canPlay(c)) {
      if(P.isWinningMove(c)) { if(best==15) mask|=1u<<c; }
      else { Position child=P;child.playCol(c);
        if(solver.probe(child,-best,-best+1)<=-best) mask|=1u<<c;
      }
    }
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
    sum_t+=seconds;max_t=std::max(max_t,seconds);
    auto exact=solver.analyze(P,false);unsigned expected=0;
    for(int c=0;c<7;c++) if(exact[c]==best) expected|=1u<<c;
    if(mask!=expected) {std::printf("MISMATCH %d %u %u\n",best,mask,expected);return 2;}
    checked++;
  }
  std::printf("probe %d: mean %.3fms max %.3fms total %.2fs; all masks equal full analysis\n",checked,1000*sum_t/checked,1000*max_t,sum_t);
}
