// Offline labels only. The platform package contains Python inference code.
// Retained reference solver is AGPL-3.0; see tools/AGPL-3.0.txt.
#include "Solver.hpp"
#include <iostream>
#include <string>
#include <chrono>
using namespace GameSolver::Connect4;
int main(int argc,char**argv){
 if(argc!=2)return 2;
 Solver solver;solver.loadBook(argv[1]);
 std::string history;double seconds;
 while(std::cin>>history>>seconds){
  Position p;
  if(history=="-")history="";
  if(p.play(history)!=history.size())return 3;
  int scores[7];for(int &v:scores)v=-999;
  auto start=std::chrono::steady_clock::now();
  for(int c: {3,2,4,1,5,0,6}){
   if(!p.canPlay(c)){scores[c]=-1000;continue;}
   if(p.isWinningMove(c)){scores[c]=1;continue;}
   Position child=p;child.playCol(c);
   double spent=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
   if(spent>=seconds)continue;
   solver.setBudget(std::min(.15,seconds-spent));
   try {int v=-solver.solve(child,true);scores[c]=(v>0)-(v<0);}
   catch(const std::runtime_error&){}
  }
  std::cout<<(history.empty()?"-":history);
  for(int v:scores)std::cout<<' '<<v;
  std::cout<<std::endl;
 }
}
