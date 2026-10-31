// Debug: print bitboards for a replay history.
#include "Position.hpp"
#include <iostream>
#include <string>
using namespace GameSolver::Connect4;
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  Position p;
  std::string h = argv[1];
  unsigned int n = p.play(h);
  std::cout << "played=" << n << " nbMoves=" << p.nbMoves()
            << " own=" << (unsigned long long)p.ownBits()
            << " mask=" << (unsigned long long)p.maskBits()
            << " key3=" << p.key3() << std::endl;
  return 0;
}
