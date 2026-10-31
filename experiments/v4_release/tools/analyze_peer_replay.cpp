// Offline replay review with the retained Pascal Pons reference solver (AGPL-3.0).
#include "Solver.hpp"
#include <iostream>
#include <string>
using namespace GameSolver::Connect4;
int main(int argc,char**argv){
 if(argc!=2)return 2;
 Solver solver;solver.loadBook(argv[1]);
 std::string history;
 while(std::cin>>history){
  Position p;
  if(p.play(history)!=history.size())return 3;
  const auto values=solver.analyze(p,true);
  std::cout<<history;
  for(int v:values)std::cout<<" "<<v;
  std::cout<<std::endl;
 }
}
