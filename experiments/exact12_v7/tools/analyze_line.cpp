// Offline analysis only: exact weak-mode scores at every ply of a replay line.
#include "Solver.hpp"
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
using namespace GameSolver::Connect4;

static int replay(Position &p,const std::string &seq,size_t upTo){
 p=Position();
 for(size_t i=0;i<seq.size() && i<upTo;i++){
  int col=seq[i]-'1';
  if(col<0||col>=Position::WIDTH||!p.canPlay(col)||p.isWinningMove(col))return -1;
  p.playCol(col);
 }
 return 0;
}

int main(int argc,char**argv){
 if(argc!=3)return 2;
 Solver solver;solver.loadBook(argv[1]);
 double seconds=std::stod(argv[2]);
 std::string history;int startPly;
 while(std::cin>>history>>startPly){
  if(history=="-")history="";
  int moves=(int)history.size();
  for(int ply=startPly;ply<moves;ply++){
   Position q;
   if(replay(q,history,ply))return 3;
   int next=history[ply]-'1';
   if(q.isWinningMove(next))break; // line ends with a winning move; stop there
   std::vector<int> scores(7,-999);
   auto start=std::chrono::steady_clock::now();
   for(int c: {3,2,4,1,5,0,6}){
    if(!q.canPlay(c)){scores[c]=-1000;continue;}
    if(q.isWinningMove(c)){scores[c]=1;continue;}
    Position child=q;child.playCol(c);
    double spent=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if(spent>=seconds)continue;
    solver.setBudget(std::min(seconds,seconds-spent));
    try {int v=-solver.solve(child,true);scores[c]=(v>0)-(v<0);}
    catch(const std::runtime_error&){}
   }
   std::cout<<(history.empty()?"-":history)<<' '<<ply;
   for(int v:scores)std::cout<<' '<<v;
   std::cout<<std::endl;
  }
 }
}
