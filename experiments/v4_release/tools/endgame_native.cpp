#define Py_LIMITED_API 0x03080000
#include <Python.h>
#include <time.h>
// Native exact-search accelerator with a CPython stable-ABI bridge.
// Original implementation; algorithm references are documented in REFERENCES.md.
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
inline double cpu_time(){timespec t;clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);return t.tv_sec+t.tv_nsec*1e-9;}
struct Timeout{};
struct Entry{U key=0;int8_t lower=0,upper=0,best=0;};
struct Engine{
 std::vector<Entry> table=std::vector<Entry>(1<<21);
 std::vector<unsigned char> book_keys,book_values;
 int book_score(U own,U mask,int n)const {
  if(n>14 || book_values.empty())return 99;
  auto partial=[&](bool reverse){U key=0;for(int c0=0;c0<7;c0++){
   int c=reverse?6-c0:c0;U bits=(mask>>(7*c))&63;
   while(bits){U low=bits&(~bits+1);int r=__builtin_ctzll(low);key=key*3+(((own>>(7*c+r))&1)?1:2);bits^=low;}
   key*=3;
  }return key;};
  U key=std::min(partial(false),partial(true))/3;size_t index=key%16777259;
  if(book_keys[index]!=(key&255) || !book_values[index])return 99;
  return int(book_values[index])-19;
 }
 uint64_t nodes=0;
 std::chrono::steady_clock::time_point deadline;
 double cpu_deadline=1e100;
 void check(){if((++nodes&4095)==0 && (std::chrono::steady_clock::now()>=deadline || cpu_time()>=cpu_deadline))throw Timeout{};}
 static U hash(U key){key^=key>>30;key*=0xbf58476d1ce4e5b9ULL;key^=key>>27;key*=0x94d049bb133111ebULL;return key^(key>>31);}
 int search(U own,U mask,int n,int alpha,int beta){
  check();if(n==42)return 0;
  U moves=playable(mask);
  if(wins(own,mask)&moves)return (43-n)/2;
  int book=book_score(own,mask,n);if(book!=99)return book;
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
 std::array<int,7> outcomes(U own,U mask,int known,int minimum,double seconds,int preferred){
  nodes=0;int n=count(mask);std::array<int,7> out;out.fill(-2);
  U moves=playable(mask),immediate=wins(own,mask)&moves;
  for(int c=0;c<7;c++){
   U bit=moves&COL(c);if(!bit){out[c]=-3;continue;}
   if(immediate&bit)out[c]=1;
   else if((mask|bit)==BOARD)out[c]=0;
   else if(wins(own^mask,mask|bit)&playable(mask|bit))out[c]=-1;
  }
  if(known==-1){for(int c=0;c<7;c++)if(out[c]!=-3)out[c]=-1;return out;}
  auto start=std::chrono::steady_clock::now();
  double cpu_end=cpu_time()+seconds;
  auto end=start+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.35));
  std::array<int,7> order={3,2,4,1,5,0,6};
  if(preferred>=0&&preferred<7){auto it=std::find(order.begin(),order.end(),preferred);std::rotate(order.begin(),it,it+1);}
  // Round-robin proof slices with reused transpositions.
  for(int sweep=0;sweep<30;sweep++){
   for(int c:order){
    if(out[c]!=-2)continue;
    if(std::chrono::steady_clock::now()>=end || cpu_time()>=cpu_end)return out;
    auto slice=std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(.15*(1+sweep)));
    deadline=std::min(end,std::chrono::steady_clock::now()+slice);
    cpu_deadline=std::min(cpu_end,cpu_time()+.10*(1+sweep));
    U bit=moves&COL(c);
    try{
     if(known==1){
      int v=search(own^mask,mask|bit,n+1,-1,0);
      if(v<0){out[c]=1;return out;}
      // -4 is a proved non-winning alternative, draw or loss.
      out[c]=-4;
     }else{
      int v=search(own^mask,mask|bit,n+1,0,1);
      if(v>0){out[c]=-1;}
      else {
        out[c]=2; // proved non-losing: draw or win
        v=search(own^mask,mask|bit,n+1,-1,0);out[c]=v<0?1:0;
      if(out[c]==1 || (minimum>=0 && out[c]>=0))return out;
      }
     }
    }catch(Timeout&){}
    if(known==1 || known==0 || minimum>=0){
     int target=(known==1 || minimum==1)?1:0;
     int candidates=0,last=-1;
     for(int d=0;d<7;d++){
       bool eligible=target==1?(out[d]==-2||out[d]==1||out[d]==2):(out[d]==-2||out[d]==-4||out[d]>=0);
       if(eligible){candidates++;last=d;}
     }
     if(candidates==1){out[last]=target==1?1:(known==0?0:2);return out;}
    }
   }
   bool any=false;for(int v:out)if(v==-2)any=true;if(!any)break;
  }
  return out;
 }
};

static Engine* ENGINE=nullptr;
static PyObject* analyze(PyObject*,PyObject* args){
 unsigned long long own,mask;int known,minimum,preferred;double seconds;
 if(!PyArg_ParseTuple(args,"KKiidi",&own,&mask,&known,&minimum,&seconds,&preferred))return nullptr;
 if((own&~mask) || (mask&~BOARD) || seconds<=0 || seconds>.72){PyErr_SetString(PyExc_ValueError,"Invalid position or CPU budget");return nullptr;}
 if(!ENGINE)ENGINE=new Engine();
 std::array<int,7> result;
 Py_BEGIN_ALLOW_THREADS
 result=ENGINE->outcomes(own,mask,known,minimum,seconds,preferred);
 Py_END_ALLOW_THREADS
 PyObject* list=PyList_New(7);
 for(int i=0;i<7;i++)PyList_SetItem(list,i,PyLong_FromLong(result[i]));
 return Py_BuildValue("NK",list,(unsigned long long)ENGINE->nodes);
}
static PyObject* load_book(PyObject*,PyObject* args){
 PyObject* data;if(!PyArg_ParseTuple(args,"O",&data))return nullptr;
 char* raw;Py_ssize_t size;
 if(PyBytes_AsStringAndSize(data,&raw,&size)<0)return nullptr;
 const size_t count=16777259;
 if(size!=6+2*count || raw[0]!=7 || raw[1]!=6 || raw[2]!=14 || raw[3]!=1 || raw[4]!=1 || raw[5]!=24){
  PyErr_SetString(PyExc_ValueError,"Invalid exact score book");return nullptr;
 }
 ENGINE->book_keys.assign(raw+6,raw+6+count);
 ENGINE->book_values.assign(raw+6+count,raw+6+2*count);
 Py_RETURN_NONE;
}
static PyObject* score_book(PyObject*,PyObject* args){
 unsigned long long own,mask;
 if(!PyArg_ParseTuple(args,"KK",&own,&mask))return nullptr;
 int score=ENGINE->book_score(own,mask,count(mask));
 if(score==99)Py_RETURN_NONE;
 return PyLong_FromLong(score);
}
static PyMethodDef methods[]={{"load_book",load_book,METH_VARARGS,"Load exact score leaf data."},{"book_score",score_book,METH_VARARGS,"Probe exact leaf score."},{"analyze",analyze,METH_VARARGS,"CPU-bounded exact outcome proofs."},{nullptr,nullptr,0,nullptr}};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_endgame_native","Native proof search; CNN still evaluates positions.",-1,methods};
PyMODINIT_FUNC PyInit__endgame_native(){ENGINE=new Engine();return PyModule_Create(&module);}
