// Independently verify coverage, legality and Bellman-optimality of every
// pre-move position in plies 1..12. No external solver is used by this checker.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
using U=uint64_t;
const U BOTTOM=[](){U v=0;for(int c=0;c<7;c++)v|=U(1)<<(7*c);return v;}();
const U BOARD=[](){U v=0;for(int c=0;c<7;c++)v|=U(63)<<(7*c);return v;}();
U moves(U mask){return(mask+BOTTOM)&BOARD;}
bool four(U p){for(int d:{1,6,7,8}){U a=p&(p>>d);if(a&(a>>(2*d)))return true;}return false;}
U key(U own,U mask,bool reverse){U v=0;for(int c0=0;c0<7;c0++){int c=reverse?6-c0:c0;for(int r=0;r<6;r++){
 U bit=U(1)<<(7*c+r);if(!(mask&bit))break;v=v*3+((own&bit)?1:2);}v*=3;}return v;}
unsigned reflect(unsigned x){unsigned v=0;for(int c=0;c<7;c++)if(x&(1u<<c))v|=1u<<(6-c);return v;}
std::vector<uint32_t> keys;std::vector<uint8_t> actions;std::vector<int8_t> scores;std::vector<int32_t> frontier;
std::vector<uint32_t> refkeys;std::vector<int8_t> refvalues;
std::vector<U> seen((387420489ULL+63)/64);U checked=0,depths[12]={};
size_t locate(U own,U mask){U a=key(own,mask,false),b=key(own,mask,true);uint32_t k=std::min(a,b)/3;
 auto it=std::lower_bound(keys.begin(),keys.end(),k);if(it==keys.end()||*it!=k)throw std::runtime_error("missing opening");return it-keys.begin();}
int32_t huff(U own,U mask,bool reverse){uint32_t v=0;for(int c0=0;c0<7;c0++){int c=reverse?6-c0:c0;
 for(int r=0;r<6;r++){U bit=U(1)<<(7*c+r);if(!(mask&bit))break;v=(v<<2)|((own&bit)?2:3);}v<<=1;}return int32_t(v<<1);}
int reference_value(U own,U mask){U f=key(own,mask,false),r=key(own,mask,true);uint32_t k=std::min(f,r)/3;
 auto it=std::lower_bound(refkeys.begin(),refkeys.end(),k);if(it==refkeys.end()||*it!=k)throw std::runtime_error("missing reference child");
 int value=refvalues[it-refkeys.begin()];return(value>0)-(value<0);
}
int value12(U own,U mask){for(bool reverse:{false,true}){int32_t k=huff(own,mask,reverse);auto it=std::lower_bound(frontier.begin(),frontier.end(),k);
 if(it!=frontier.end()&&(uint32_t(*it)&~3u)==uint32_t(k))return -int(uint32_t(*it)&3);}return 1;}
void visit(U own,U mask,int depth){
 U f=key(own,mask,false),r=key(own,mask,true);uint32_t k=std::min(f,r)/3;
 if(seen[k>>6]&(U(1)<<(k&63)))return;seen[k>>6]|=U(1)<<(k&63);
 size_t index=locate(own,mask);unsigned recorded=actions[index];if(r<f)recorded=reflect(recorded);
 U possible=moves(mask);int best=-2;unsigned optimal=0,legal=0;
 for(int c=0;c<7;c++){U bit=possible&(U(63)<<(7*c));if(!bit)continue;legal|=1u<<c;
  int v=four(own|bit)?1:-(!refkeys.empty()?reference_value(own^mask,mask|bit):(depth==11?value12(own^mask,mask|bit):int(scores[locate(own^mask,mask|bit)])));
  if(v>best){best=v;optimal=1u<<c;}else if(v==best)optimal|=1u<<c;
 }
 if(!recorded || (recorded&~legal) || (recorded&~optimal) || scores[index]!=best)throw std::runtime_error("illegal or suboptimal entry");
 checked++;depths[depth]++;
 if(depth==11)return;
 for(int c=0;c<7;c++){U bit=possible&(U(63)<<(7*c));if(bit&&!four(own|bit))visit(own^mask,mask|bit,depth+1);}
}
int main(int argc,char**argv){
 if(argc!=3&&argc!=4)return 1;
 std::ifstream table(argv[1],std::ios::binary);char magic[4];uint32_t count;
 table.read(magic,4);table.read((char*)&count,4);if(std::string(magic,4)!="MBK2"||count!=3356580)throw std::runtime_error("bad table");
 keys.resize(count);actions.resize(count);scores.resize(count);
 table.read((char*)keys.data(),4*count);table.read((char*)actions.data(),count);table.read((char*)scores.data(),count);
 if(!table || !std::is_sorted(keys.begin(),keys.end()) || std::adjacent_find(keys.begin(),keys.end())!=keys.end())throw std::runtime_error("invalid sorted keys");
 std::ifstream source(argv[2],std::ios::binary);unsigned char raw[4];
 while(source.read((char*)raw,4))frontier.push_back(int32_t((uint32_t(raw[0])<<24)|(uint32_t(raw[1])<<16)|(uint32_t(raw[2])<<8)|raw[3]));
 if(frontier.size()!=1735945)throw std::runtime_error("bad frontier");
 if(argc==4){
  std::ifstream ref(argv[3],std::ios::binary);char rmagic[4];uint32_t rn;ref.read(rmagic,4);ref.read((char*)&rn,4);
  if(std::string(rmagic,4)!="MBK2"||rn!=9188816)throw std::runtime_error("bad reference");
  refkeys.resize(rn);refvalues.resize(rn);ref.read((char*)refkeys.data(),4*rn);ref.seekg(rn,std::ios::cur);ref.read((char*)refvalues.data(),rn);
  if(!ref||!std::is_sorted(refkeys.begin(),refkeys.end()))throw std::runtime_error("reference not sorted");
 }
 visit(0,0,0);if(checked!=count)throw std::runtime_error("coverage mismatch");
 for(int d=0;d<12;d++)std::printf("ply %d: %llu verified\n",d+1,(unsigned long long)depths[d]);
 std::printf("PASS: %llu positions, complete first-twelve-ply coverage; every recorded action is WDL-optimal.\n",(unsigned long long)checked);
}
