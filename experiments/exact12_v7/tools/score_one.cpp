// Offline analysis only: per-column exact weak scores of ONE position, big budget.
#include "Solver.hpp"
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
using namespace GameSolver::Connect4;
int main(int argc,char**argv){
 if(argc!=4)return 2;
 Solver solver;solver.loadBook(argv[1]);
 double seconds=std::stod(argv[2]);
 std::string history=argv[3];
 Position p;
 if(history!="-"){
  for(size_t i=0;i<history.size();i++){
   int col=history[i]-'1';
   if(col<0||col>=Position::WIDTH||!p.canPlay(col)||p.isWinningMove(col)){std::cout<<"BAD\n";return 3;}
   p.playCol(col);
  }
 }
 std::vector<int> scores(7,-999);
 auto start=std::chrono::steady_clock::now();
 for(int c: {3,2,4,1,5,0,6}){
  if(!p.canPlay(c)){scores[c]=-1000;continue;}
  if(p.isWinningMove(c)){scores[c]=1;continue;}
  Position child=p;child.playCol(c);
  double spent=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  if(spent>=seconds){std::cout<<"TIMEOUT\n";return 0;}
  solver.setBudget(std::min(seconds,seconds-spent));
  try {int v=-solver.solve(child,true);scores[c]=(v>0)-(v<0);}
  catch(const std::runtime_error&){}
 }
 for(int v:scores)std::cout<<v<<' ';
 std::cout<<std::endl;
}
