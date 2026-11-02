// Independent offline distance oracle, linked with the existing Pons teacher.
// g++ -O3 -std=c++17 -I/tmp/mbwork tools/score_distance.cpp \
//     /tmp/mbwork/Solver.cpp -o /tmp/c4-score-distance
#include "Solver.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
using namespace GameSolver::Connect4;
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    Solver solver;
    solver.loadBook(argv[1]);
    const double budget = std::stod(argv[2]);
    Position root;
    for (char ch : std::string(argv[3])) {
        int col = ch - '1';
        if (col < 0 || col >= 7 || !root.canPlay(col) || root.isWinningMove(col)) return 3;
        root.playCol(col);
    }
    auto start = std::chrono::steady_clock::now();
    std::vector<int> values(7, -1000);
    for (int col = 0; col < 7; ++col) {
        if (!root.canPlay(col)) continue;
        if (root.isWinningMove(col)) {
            values[col] = (43 - root.nbMoves()) / 2;
            continue;
        }
        double left = budget - std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (left <= 0) { std::cout << "TIMEOUT\n"; return 0; }
        // The Python audit also enforces a process timeout for this teacher.
        Position child = root;
        child.playCol(col);
        try { values[col] = -solver.solve(child, false); }
        catch (...) { std::cout << "TIMEOUT\n"; return 0; }
    }
    for (int value : values) std::cout << value << ' ';
    std::cout << '\n';
}
