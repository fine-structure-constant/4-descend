// Offline data generator. Runtime/exported Bot uses Python MCTS + CNN only.
// Bitboard/null-window techniques: Pascal Pons, blog.gamesolver.org.
// This local implementation uses a fixed-size, collision-checked bound table.
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
using U=uint64_t;
constexpr U BOTTOM=0x40810204081ULL;
constexpr U BOARD=0xfdfbf7efdfbfULL;
constexpr int ORDER[7]={3,2,4,1,5,0,6};
constexpr U COL(int c){return U(63)<<(7*c);}
inline int count(U v){return __builtin_popcountll(v);}
inline U playable(U mask){return (mask+BOTTOM)&BOARD;}
inline U wins(U p,U mask){
 U r=(p<<1)&(p<<2)&(p<<3);
 for(int d:{7,6,8}){
  U a=(p<<d)&(p<<(2*d));r|=a&(p<<(3*d));r|=a&(p>>d);
  a=(p>>d)&(p>>(2*d));r|=a&(p<<d);r|=a&(p>>(3*d));
 }
 return r&(BOARD^mask);
}
struct Timeout{};
struct Entry{U key=0;int8_t lower=0,upper=0,best=0;};
struct Engine{
 std::vector<Entry> table=std::vector<Entry>(1<<21);
 uint64_t nodes=0;
 std::chrono::steady_clock::time_point deadline;
 void check(){if((++nodes&4095)==0 && std::chrono::steady_clock::now()>=deadline)throw Timeout{};}
 static U hash(U key){key^=key>>30;key*=0xbf58476d1ce4e5b9ULL;key^=key>>27;key*=0x94d049bb133111ebULL;return key^(key>>31);}
 int search(U own,U mask,int n,int alpha,int beta){
  check();if(n==42)return 0;
  U moves=playable(mask);
  if(wins(own,mask)&moves)return (43-n)/2;
  U opp=own^mask,threats=wins(opp,mask),forced=moves&threats;
  if(forced){if(forced&(forced-1))return -(42-n)/2;moves=forced;}
  moves&=~(threats>>1);
  if(!moves)return -(42-n)/2;
  if(n>=40)return 0;
  alpha=std::max(alpha,-(42-n)/2);beta=std::min(beta,(41-n)/2);
  if(alpha>=beta)return alpha;
  U key=own+mask,tag=key+1;
  size_t index=hash(key)&(table.size()-1);
  Entry old=table[index];
  int lo=-22,hi=22,preferred=-1;
  if(old.key==tag){lo=old.lower;hi=old.upper;preferred=old.best;
   if(lo>=beta)return lo;if(hi<=alpha)return hi;
   alpha=std::max(alpha,lo);beta=std::min(beta,hi);if(alpha>=beta)return alpha;
  }
  const int initialA=alpha,initialB=beta;
  struct Move{int rank,col;U bit;};std::array<Move,7> a;int len=0;
  for(int c:ORDER){U bit=moves&COL(c);if(bit)a[len++]={count(wins(own|bit,mask|bit))+(c==preferred?100:0),c,bit};}
  for(int i=1;i<len;i++){Move item=a[i];int j=i;while(j>0&&a[j-1].rank<item.rank){a[j]=a[j-1];j--;}a[j]=item;}
  int best=-22,bestCol=a[0].col;
  for(int i=0;i<len;i++){
   int v=-search(opp,mask|a[i].bit,n+1,-beta,-alpha);
   if(v>best){best=v;bestCol=a[i].col;}
   alpha=std::max(alpha,best);if(alpha>=beta)break;
  }
  if(best<=initialA)hi=std::min(hi,best);
  else if(best>=initialB)lo=std::max(lo,best);
  else lo=hi=best;
  table[index]=Entry{tag,(int8_t)lo,(int8_t)hi,(int8_t)bestCol};return best;
 }
 std::array<int,7> outcomes(U own,U mask,int known,double seconds,int preferred){
  nodes=0;int n=count(mask);std::array<int,7> out;out.fill(-2);
  U moves=playable(mask),immediate=wins(own,mask)&moves;
  for(int c=0;c<7;c++){
   U bit=moves&COL(c);if(!bit){out[c]=-3;continue;}
   if(immediate&bit)out[c]=1;
   else if((mask|bit)==BOARD)out[c]=0;
   else if(wins(own^mask,mask|bit)&playable(mask|bit))out[c]=-1;
  }
  auto start=std::chrono::steady_clock::now();
  auto end=start+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
  std::array<int,7> order={3,2,4,1,5,0,6};
  if(preferred>=0&&preferred<7){auto it=std::find(order.begin(),order.end(),preferred);std::rotate(order.begin(),it,it+1);}
  // Round-robin proof slices with reused transpositions.
  for(int sweep=0;sweep<30;sweep++){
   for(int c:order){
    if(out[c]!=-2)continue;
    if(std::chrono::steady_clock::now()>=end)return out;
    auto slice=std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(.15*(1+sweep)));
    deadline=std::min(end,std::chrono::steady_clock::now()+slice);
    U bit=moves&COL(c);
    try{
     if(known==1){
      int v=search(own^mask,mask|bit,n+1,-1,0);
      if(v<0){out[c]=1;return out;}
      // -4 is a proved non-winning alternative, draw or loss.
      out[c]=-4;
     }else{
      int v=search(own^mask,mask|bit,n+1,0,1);
      if(v>0){out[c]=-1;continue;}
      v=search(own^mask,mask|bit,n+1,-1,0);out[c]=v<0?1:0;
      if(out[c]==1)return out;
     }
    }catch(Timeout&){}
    if(known==1){
     int candidates=0,last=-1;for(int d=0;d<7;d++)if(out[d]==-2||out[d]==1){candidates++;last=d;}
     if(candidates==1){out[last]=1;return out;}
    }
   }
   bool any=false;for(int v:out)if(v==-2)any=true;if(!any)break;
  }
  return out;
 }
};
int main(){Engine engine;U own,mask;int known,preferred;double seconds;
 while(std::cin>>own>>mask>>known>>seconds>>preferred){
  auto result=engine.outcomes(own,mask,known,seconds,preferred);
  for(int v:result)std::cout<<v<<' ';std::cout<<engine.nodes<<std::endl;
 }
}
