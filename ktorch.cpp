#include "ktorch.h"

// --------------------------------------------------------------------------------------------------
// krrbuf - copy msg to a buffer for signalling error to k
// dictadd - add an entry in a dictionary mapping symbol -> k value
// xind - true if i is valid index of k list (type=0)
// kptr - given void *, return K value(type=0) containing a single long scalar = (intptr_t)void *
// xptr - given k value, set object pointer if possible and return true, else false
// xhelp - check for single argument: `help, or 2 symbols, e.g. `help`conv2d
// --------------------------------------------------------------------------------------------------
S krrbuf(const char *s) {
 ZC thread_local b[4096]; b[0]=0; 
 return strncat(b, s, sizeof(b)-1);
}

V dictadd(K x, S s,K v){K *k=kK(x); js(&k[0],cs(s)); jk(&k[1],v);}
V dictadd(K x,cS s,K v){K *k=kK(x); js(&k[0],cs(s)); jk(&k[1],v);}

B xind(K x,J i) {return !x->t && -1<i && i<x->n;}
K kptr(V *v){return knk(1,kj((intptr_t)v));}
B xptr(K x) {return !x->t && x->n==1 && kK(x)[0]->t==-KJ;}
B xptr(K x,Ptr &p) {return xptr(x) ? p=(Ptr)kK(x)[0]->j,true : false;}
B xptr(K x,J i,Ptr &p) {return xind(x,i) && xptr(kK(x)[i],p);}

B xhelp(K x) {return x->t == -KS && x->s == env().help;}
B xhelp(K x,S &s) {
 if(x->t==KS && x->n == 2 && kS(x)[0]==env().help)
  return s=kS(x)[1],true;
 else
  return false;
}

// ------------------------------------------------------------------------------------------
// match - return true if scalars match (check long/double value)
// kscalar - return k double/long from torch scalar
// xlen - 1 if scalar else x->n for lists, no. of table rows or dictionary elements
// kname - string from k data type
// ksizeof - given k type, return size of element, e.g. KF -> 8
// maptype - map k data type to/from torch type
// ------------------------------------------------------------------------------------------
B match(const Scalar &x,const Scalar &y) {
 if(x.isIntegral()) {
  if(y.isIntegral())
   return x.toLong() == y.toLong();
  else if(y.isFloatingPoint())
   return x.toDouble() == y.toDouble();
 } else if(x.isFloatingPoint()) {
  if(y.isFloatingPoint() || y.isIntegral())
   return x.toDouble() == y.toDouble();
 }
 AT_ERROR("Unexpected scalar type(s), neither integral or floating point, cannot compare");
}

K kscalar(const Scalar &s) {
 if(s.isIntegral())
  return kj(s.toLong());
 else if(s.isFloatingPoint())
  return kf(s.toDouble());
 AT_ERROR("Unexpected scalar type(s), neither integral or floating point, cannot convert");
}

J xlen(K x) {
 if(x->t < 0 || x->t > 99) return 1;
 else if(x->t < 98)        return x->n;
 else if(x->t == 98)       return xlen(kK(kK(x->k)[1])[0]);
 else                      return xlen(kK(x)[0]);
}

J xlen(K x,J i) {return xind(x,i) ? xlen(kK(x)[i]) : -1;}

cS kname(A k) {
 A t=abs(k); B b=k<0;
 switch(t) {
  case 0: return "list";
  case 1: return b ? "boolean scalar" : "boolean list";
  case 2: return b ? "guid scalar" : "guid list";
  case 4: return b ? "byte scalar" : "byte list";
  case 5: return b ? "short scalar" : "short list";
  case 6: return b ? "int scalar" : "int list";
  case 7: return b ? "long scalar" : "long list";
  case 8: return b ? "real scalar" : "real list";
  case 9: return b ? "float scalar" : "float list";
  case 10: return b ? "char scalar" : "char list";
  case 11: return b ? "symbol scalar" : "symbol list";
  case 12: return b ? "timestamp scalar" : "timestamp list";
  case 13: return b ? "month scalar" : "month list";
  case 14: return b ? "date scalar" : "date list";
  case 15: return b ? "datetime scalar" : "datetime list";
  case 16: return b ? "timespan scalar" : "timespan list";
  case 17: return b ? "minute scalar" : "minute list";
  case 18: return b ? "second scalar" : "second list";
  case 19: return b ? "time scalar" : "time list";
  case 97: return "nested sym enum";
  case 98: return "table";
  case 99: return "dictionary";
  case 100: return "lambda";
  case 101: return "null/unary primitive";
  case 102: return "operator";
  case 103: return "adverb";
  case 104: return "projection";
  case 105: return "composition";
  case 106: return "f'";
  case 107: return "f/";
  case 108: return "f\\";
  case 109: return "f':";
  case 110: return "f/:";
  case 111: return "f\\:";
  case 112: return "dynamic load";
  default:
    if(t>19 && t<77)
     return b ? "enum scalar" : "enum list";
    else if(t>76 && t<97)
     return "map";
    else
     return "value(unrecognized type)";
 }
}

J ksizeof(A k) {
 switch(k) {
  case KE: return sizeof(E);
  case KF: return sizeof(F);
  case KJ: return sizeof(J);
  case KI: return sizeof(I);
  case KH: return sizeof(H);
  case KC: return sizeof(C);
  case KB:
  case KG: return sizeof(G);
  default: AT_ERROR("No element size for k ",kname(k)); return -1;
 }
}

A maptype(TypeMeta s) {
 for(auto &m:env().dtype)
  if(s==std::get<1>(m)) return std::get<2>(m);
 AT_ERROR("No k data type found for torch type: ",s);
 return 0;
}

TypeMeta maptype(A k) {
 A t=(k<0) ? -k : k;
 for(auto &m:env().ktype)
  if(t==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("No torch type found for k: ",kname(k));
}

// --------------------------------------------------------------------------------------
// xnull  - true if null, i.e. (::)
// xempty - true if null or empty K list without type, i.e. :: or ()
// xmixed - true if up to m elements of k value has mixed types/lengths
// xsym - if arg is k symbol, return true and set sym, else false
// xsyms - if sym scalar or non-empty sym list, set 1st sym and return true
// xdev  - check sym for map to list of known devices, `cpu`cuda`cuda:0`cuda:1..
// xint64 - check for long scalar/list element and convert to int64_t
// xlong - check for long scalar/list, set value(s) and return true else false
// xdouble - check for scalar double from k, set value and return true, false else
// xdict - return true if k value is a dictionary
// xstate - check for dictionary/table defining module state
// xsize - check for long(s)/double(s), set array ref/expanding array used for sizing
// --------------------------------------------------------------------------------------
B xnull(K x) {return x->t==101 && x->g==0;}
B xnull(K x,J i) {return xind(x,i) && xnull(kK(x)[i]);}
B xempty(K x) {return xnull(x) ? true : (x->t ? false : x->n==0);}
B xempty(K x,J i) {return xind(x,i) && xempty(kK(x)[i]);}

B xmixed(K x,J m) {      // check up to m elements of k value for mixed types/lengths
 A t; J i,n;
 if(!x->t)                                              // if general list
  if(x->n > 1) {                                        // with more than 1 element
   t=kK(x)[0]->t;                                       // 1st type encountered
   if(t>19) return true;                                // enums,maps,etc.
   n=t<0 ? 1 : kK(x)[0]->n;                             // 1st size
   if(m>x->n) m=x->n;                                   // check up to m elements
   for(i=1;i<m;++i)
    if(t != kK(x)[i]->t) return true;                   // different data type or scalar vs list
    else if(n != (t<0 ? 1 : kK(x)[i]->n)) return true;  // different length
  }
 return false;
}

B xsym(K x,S &s) {return (x->t==-KS) ? s=x->s,true : false;}
B xsym(K x,J i,S &s) {return xind(x,i) && xsym(kK(x)[i],s);}
B xsyms(K x,S &s) {
 if(xsym(x,s)) return true;
 else if(x->t == KS && x->n) return s=kS(x)[0],true;
 else return false;
}

B xdev(K x,torch::Device &d) {
 if(x->t==-KS) {
  for(auto &m:env().device)
   if(x->s==std::get<0>(m)) return d=std::get<1>(m),true;
 }
 return false;
}

B xdev(K x,J i,torch::Device &d) {return xind(x,i) && xdev(kK(x)[i],d);}

B xint64(K x,int64_t &j) {return (x->t == -KJ) ? j=x->j,true : false;}  //convert J -> int64_t
B xint64(K x,J i,int64_t &j) {return xind(x,i) && xint64(kK(x)[i],j);}  //mac doesn't differentiate, linux does

B xlong(K x,J &j) {return (x->t == -KJ) ? j=x->j,true : false;}       //check k scalar
B xlong(K x,J i,J &j) {return xind(x,i) && xlong(kK(x)[i],j);}        //check k list element

B xlong(K x,J &n,J *&v){                                           //check for k list of longs
 if(x->t == KJ){          n=x->n; v=kJ(x); return true;            //list of long ints
 } else if(x->t == -KJ){  n=1;    v=&x->j; return true;            //scalar long ok too
 } else if(x->t == 0 && x->n == 0) { n=0;  return true;            //empty,no type also ok
 } else { return false;
 }
}

B xlong(K x,J i,J &n, J *&v) {return xind(x,i) && xlong(kK(x)[i],n,v);}  // check element of k list

B xdouble(K x,F &f) {return (x->t == -KF) ? f=x->f,true : false;}    //check k scalar
B xdouble(K x,J i,F &f) {return xind(x,i) && xdouble(kK(x)[i],f);}   //check k list element

B xdict(K x) {return x->t==99 && (kK(x)[0]->t==KS || (kK(x)[0]->t==0 && kK(x)[0]->n==0));}
B xdict(K x,J i) {return xind(x,i) && xdict(kK(x)[i]);}

B xstate(K x) {return xdict(x) || x->t==98;}
B xstate(K x,J i) {return xind(x,i) && xstate(kK(x)[i]);}

// retrieve long integers from x -> IntArrayRef (linux clang/gcc require int64_t* from J*)
B xsize(K x,JRef &s) {J n,*v; return (xlong(x,n,v)) ? s=JRef((int64_t*)v,n),true : false;}
B xsize(K x,J i,JRef &s) {return xind(x,i) && xsize(kK(x)[i],s);}  //check element of k list

// retrieve long integers/doubles from x -> ExpandingArray ptr of longs/floats
B xsize(K x,J d,int64_t *a) {
 B b=false;
 if((b=x->t == -KJ)) {
   for(J i=0;i<d;++i) a[i]=x->j;
 } else if(x->t == KJ) {
  if((b=d == x->n))
   for(J i=0;i<d;++i) a[i]=kJ(x)[i];
  else
   AT_ERROR(d,"-element list of long integers expected, ",x->n," supplied");
 }
 return b;
}

B xsize(K x,J d,F *a) {
 B b=false; 
 if((b=x->t == -KF)) {
  for(J i=0;i<d;++i) a[i]=x->f;
 } else if(x->t == KF) {
  if((b=d == x->n))
   for(J i=0;i<d;++i) a[i]=kF(x)[i];
  else
   AT_ERROR(d,"-element list of doubles expected, ",x->n," supplied");
 }
 return b;
}

B xsize(K x,J i,J d,int64_t *a) {return xind(x,i) && xsize(kK(x)[i],d,a);}
B xsize(K x,J i,J d,F       *a) {return xind(x,i) && xsize(kK(x)[i],d,a);}

// ------------------------------------------------------------------------------------------------------
// xten - check arg(s) for allocated ptr to tensor: set tensor & return true if found, else false
// xtenpair - check arg(s) for a pair of allocated tensor ptrs: if found, set & return true, else false
// xten3 - check arg(s) for a triplet of allocated tensors
// xtenarg - check arg(s) for a list of allocated tensors, or list of input arrays or mix of both
// ------------------------------------------------------------------------------------------------------
B xten(K x,Tensor &t) {
 Ptr p;
 return (xptr(x,p) && p->t==Class::tensor && p->c==Cast::tensor) ? t=*(Tensor*)p->v,true : false;
}

B xten(K x,J i,Tensor& t) {return xind(x,i) && xten(kK(x)[i],t);}
B xtenpair(K x,Tensor& y,Tensor& z) {return xten(x,0,y) && xten(x,1,z);}
B xtenpair(K x,J i,Tensor& y,Tensor& z) {return xind(x,i) && xtenpair(kK(x)[i],y,z);}
B xten3(K x,Tensor& t1,Tensor& t2,Tensor& t3) {return xten(x,0,t1) && xten(x,1,t2) && xten(x,2,t3);}
B xten3(K x,J i,Tensor& t1,Tensor& t2,Tensor& t3) {return xind(x,i) && xten3(kK(x)[i],t1,t2,t3);}
 
B xtenarg(K x,J i,Tensor& a,Tensor &b) {
 B p;
 p=xten(x,i,a)   ? true : (a=kput(x,i),false);
 p=xten(x,i+1,b) ? true : (b=kput(x,i+1),p);
 return p;
}

B xtenarg(K x,J i,Tensor& a,Tensor &b,Tensor &c) {
 B p;
 p=xten(x,i,a)   ? true : (a=kput(x,i),false);
 p=xten(x,i+1,b) ? true : (b=kput(x,i+1),p);
 p=xten(x,i+2,c) ? true : (c=kput(x,i+2),p);
 return p;
}

B xtenarg(K x,Tensor& a,Tensor &b)           {return xtenarg(x,0,a,b);}
B xtenarg(K x,Tensor& a,Tensor &b,Tensor &c) {return xtenarg(x,0,a,b,c);}
 
// ------------------------------------------------------------------------------------------------------
// xseq - check arg(s) for allocated sequential modules
// xloss - check arg(s) for allocated loss function
// xoptim - check arg(s) for allocated optimizer pointer
// ------------------------------------------------------------------------------------------------------
B xseq(K x,Sequential &s) {
 Ptr p;
 return (xptr(x,p) && p->t==Class::sequential) ? s=*(Sequential*)p->v,true : false;
}

B xseq(K x,J i,Sequential& s) {return xind(x,i) && xseq(kK(x)[i],s);}

B xloss(K x,Ptr &p) {return xptr(x,p) && p->t==Class::loss;}
B xloss(K x,J i,Ptr &p) {return xind(x,i) && xloss(kK(x)[i],p);}

B xoptim(K x,Ptr &p) {return xptr(x,p) && p->t==Class::optimizer;}
B xoptim(K x,J i,Ptr &p) {return xind(x,i) && xoptim(kK(x)[i],p);}

// ------------------------------------------------------------------------------------------------------
// xnum - check for double or long int k scalar, set double & return true, else false
// xnum - check for number(float,double,long,int,short), set torch scalar & return true, else false
// xnumn - similar to xnum, but with optional scalars which remain unset if null scalar from k
// xnumt - similar to xnum, but also attempts to convert tensor to scalar
// xnumlist - take single value from k numeric list -> torch scalar
// xbyte - convert k bool,char,byte -> torch scalar
// xscalar - convert k number or byte -> torch scalar
// ------------------------------------------------------------------------------------------------------
B xnum(K x,F f) {
 switch(x->t) {
  case -KF: return f=x->f,true;
  case -KJ: return f=x->j,true;
  default: return false;
 }
}
B xnum(K x,J i,F f) {return xind(x,i) && xnum(kK(x)[i],f);}

B xnum(K x,Scalar& s) {
 switch(x->t) {
  case -KF: return s=x->f, true;
  case -KE: return s=x->e, true;
  case -KJ: return s=(int64_t)x->j, true;
  case -KI: return s=x->i, true;
  case -KH: return s=x->h, true;
  default: return false;
 }
}
B xnum(K x,J i,Scalar& s) {return xind(x,i) && xnum(kK(x)[i],s);}

B xnumn(K x,c10::optional<Scalar>& s) {
 switch(x->t) {
  case -KF: if(x->f==x->f) s=x->f; return true;
  case -KE: if(x->e==x->e) s=x->e; return true;
  case -KJ: if(x->j!=nj) s=(int64_t)x->j; return true;
  case -KI: if(x->i!=ni) s=x->i; return true;
  case -KH: if(x->h!=nh) s=x->h; return true;
  default: return false;
 }
}
B xnumn(K x,J i,c10::optional<Scalar>& s) {return xind(x,i) && xnumn(kK(x)[i],s);}

B xnumt(K x,Scalar& s) {
 Tensor t;
 if(xnum(x,s))      return true;
 else if(xten(x,t)) return s=t.item(), true;
 else               return false;
}

B xnumt(K x,J i,Scalar& s) {return xind(x,i) && xnumt(kK(x)[i],s);}

B xnumlist(K x,J i,Scalar &a) {
 switch(x->t) {
  case KF: return a=kF(x)[i], true;
  case KE: return a=kE(x)[i], true;
  case KJ: return a=(int64_t)kJ(x)[i], true;
  case KI: return a=kI(x)[i], true;
  case KH: return a=kH(x)[i], true;
  case KB:
  case KC: return a=kG(x)[i], true;
  default: return false;
 }
}

B xbyte(K x,Scalar &s) { return (x->t==-KB || x->t==-KC || xt==-KG) ? s=x->g,true : false;}
B xbyte(K x,J i,Scalar &s) {return xind(x,i) && xbyte(kK(x)[i],s);}

B xscalar(K x,Scalar &s) { return xnum(x,s) || xbyte(x,s);}
B xscalar(K x,J i,Scalar &s) {return xind(x,i) && xscalar(kK(x)[i],s);}

// ------------------------------------------------------------------------------------------------------
// xbool - if value is boolean, set value and return true, else false
// xlevel - if value is short,int,long scalar, set int and return true else false
// mtype - match sym to/from TypeMeta(newer datatype from Caffe2)
// stype = match sym to/from ScalarType(older ATen datatypes)
// xtype - symbol to scalar type or type meta, return true if scalar type/type meta set, else false
// xopt - sym(s) -> tensor options, return true if ok, false if not sym(s) or error if unknown sym
// xto - device and datatype sym(s) -> tensor options, return true if ok, false if not sym(s)
// xmode - check if sym, if matches a known tensor creation mode, set mode and return true else false
// xbacksym - check if sym, if matches back prop graph setting, set retain/create graph flags else false
// ------------------------------------------------------------------------------------------------------
B xbool(K x,B &b) {return (x->t == -KB) ? b=x->g,true : false;}
B xbool(K x,J i,B &b) {return xind(x,i) && xbool(kK(x)[i],b);}

B xlevel(K x,I &n) {
 switch(-x->t) {
  case KH: return n=x->h, true;
  case KI: return n=x->i, true;
  case KJ: return n=x->j, true;
  default: return false;
 }
}

B xlevel(K x,J i,I &n) {return xind(x,i) && xlevel(kK(x)[i],n);}

TypeMeta mtype(S s) {
  for(auto &m:env().dtype) if(s==std::get<0>(m)) return std::get<1>(m);
  AT_ERROR("Unrecognized data type: ",s);
}

S mtype(TypeMeta t) {
  for(auto &m:env().dtype) if(t==std::get<1>(m)) return std::get<0>(m);
  AT_ERROR("Unrecognized data type: ",t);
}

ScalarType stype(S s) {return torch::typeMetaToScalarType(mtype(s));}
S stype(ScalarType t) {return mtype(torch::scalarTypeToTypeMeta(t));}
S stype(c10::optional<ScalarType> t) {return mtype(torch::scalarTypeToTypeMeta(t ? *t : ScalarType::Undefined));}

B xtype(K x,ScalarType &s)                {if(x->t == -KS) return s=stype(x->s), true; return false;}
B xtype(K x,c10::optional<ScalarType> &s) {if(x->t == -KS) return s=stype(x->s), true; return false;}
B xtype(K x,TypeMeta   &t) {if(x->t == -KS) return t=mtype(x->s), true; return false;}

B xtype(K x,J i,ScalarType &s)                {return xind(x,i) && xtype(kK(x)[i],s);}
B xtype(K x,J i,c10::optional<ScalarType> &s) {return xind(x,i) && xtype(kK(x)[i],s);}
B xtype(K x,J i, TypeMeta &t) {return xind(x,i) && xtype(kK(x)[i],t);}

B xopt(S s,TensorOptions &o) {
 auto &e=env();
 for(auto &m:e.device)
  if(s == std::get<0>(m)) return o=o.device(std::get<1>(m)), true;
 for(auto &m:e.dtype)
  if(s == std::get<0>(m)) return o=o.dtype(std::get<1>(m)), true;
 for(auto &m:e.layout)
  if(s == std::get<0>(m)) return o=o.layout(std::get<1>(m)), true;
 for(auto &m:e.gradient)
  if(s == std::get<0>(m)) return o=o.requires_grad(std::get<1>(m)), true;
 return false;
}

B xopt(K x,TensorOptions &o) {
 if (x->t == -KS || x->t == KS) {
  B a=x->t < 0; I i,n=a ? 1 : x->n;
  for(i=0; i<n; ++i) {
   S s=a ? x->s : kS(x)[i];
   if (!xopt(s,o))
    AT_ERROR("Unrecognized tensor option: `", s);
  }
  return true;
 } else {
  return false;
 }
}

B xopt(K x,J i,TensorOptions &o) { return !x->t && -1<x->n && i<x->n && xopt(kK(x)[i],o);}

B xto(S s,TensorOptions &o) {
 for(auto &m:env().device)
  if(s == std::get<0>(m)) return o=o.device(std::get<1>(m)), true;
 for(auto &m:env().dtype)
  if(s == std::get<0>(m)) return o=o.dtype(std::get<1>(m)), true;
 return false;
}

B xto(K x,TensorOptions &o) { 
 if (x->t == -KS || x->t == KS) {
  B a=x->t < 0; I i,n=a ? 1 : x->n;
  for(i=0; i<n; ++i) {
   S s=a ? x->s : kS(x)[i];
   if (!xto(s,o))
    AT_ERROR("Unrecognized option: `",s,", expecting device and/or datatype, e.g. `cuda or `cuda:0`float");
  }
  return true;
 } else {
  return false;
 }
}

B xmode(K x,Tensormode &m) {
 if(x->t == -KS) {
  for(auto &v:env().tensormode)
   if(x->s == std::get<0>(v)) return m=std::get<1>(v), true;
  AT_ERROR("Unrecognized tensor creation mode: ",x->s);
 }
 return false;
}

B xmode(K x,J i,Tensormode &m) {return xind(x,i) && xmode(kK(x)[i],m);}

B xbacksym(K x,B& a,B& b) {
 if(x->t == -KS) {
  for(auto &s:env().backsym)
   if(x->s == std::get<0>(s)) return a=std::get<1>(s),b=std::get<2>(s), true;
  AT_ERROR("Unrecognized setting for backprop: ",x->s,", expecting one of: free,retain,create or createfree");
 }
 return false;
}

B xbacksym(K x,J i,B& a,B& b) {return xind(x,i) && xbacksym(kK(x)[i],a,b);}

// ------------------------------------------------------------------------------------------
// xpairs - initialize a set of name-value pairs given as an argument from k
// xpair - evaluate the next name-value pair, set sym,numeric,list or general value
// xargc - return count of args to process given arg(s), offset, pairs structure to initiate
// xnone - return true if, given arg list and offset, no meaningful arg supplied
// ------------------------------------------------------------------------------------------
B xpairs(K x,Pairs &p) {   // initialize Pairs structure from k value
 p.a=0, p.i=0, p.n=0;      // sets a: 1-dict,2-pairs,3-list,4-syms
 if(x->t==99) {
  K y=kK(x)[0];
  if(y->t==KS || !(y->t || y->n))
   p.a=1, p.n=y->n;
  else
   AT_ERROR("Unexpected name,value dictionary with ",kname(kK(x)[0]->t)," as keys");
 } else if(x->t==KS) {
  if(x->n%2==0)
   p.a=4, p.n=x->n/2;
  else
   AT_ERROR("Uneven no. of symbols for name,value pairs: ",x->n);
 } else if(!x->t) {
  if(!x->n) {                      // empty list
   p.a=2, p.n=0;
  } else if(kK(x)[0]->t==-KS) {    // list of (sym1;val1;sym2;val2..)
   if(x->n%2==0)
    p.a=3, p.n=x->n/2;
   else
    AT_ERROR("Uneven no. of elements for name,value pairs in list: ",x->n);
  } else {                         // assume list of pairs if symbol in first pair
   K y=kK(x)[0];
   if(y->n==2 && (y->t==KS || (!y->t && kK(y)[0]->t==-KS)))
    p.a=2, p.n=x->n;
  }
 }
 return p.a ? (p.x=x,true) : false;
}

B xpairs(K x,J i,Pairs &p) {return xind(x,i) && xpairs(kK(x)[i],p);}

ZV xpair(Pairs& p,K x,J i) {
 if(x->t<0) {
  switch(x->t) {
   case -KS: p.s=x->s; p.t=-KS; break;
   case -KB: p.b=x->g; p.t=-KB; break;
   case -KH: p.j=x->h; p.t=-KJ; break;
   case -KI: p.j=x->i; p.t=-KJ; break;
   case -KJ: p.j=x->j; p.t=-KJ; break;
   case -KE: p.f=x->e; p.t=-KF; break;
   case -KF: p.f=x->f; p.t=-KF; break;
   default: AT_ERROR("name-value pairs not implemented for ",kname(x->t)," value"); break;
  }
 } else if (i>-1) {
  if(i>=x->n)
   AT_ERROR("name,value index[",i,"] invalid for ",kname(x->t)," with ",x->n," elements");
  switch(x->t) {
   case 0:  xpair(p,kK(x)[i],-1); break;
   case KS: p.s=kS(x)[i]; p.t=-KS; break;
   case KB: p.b=kG(x)[i]; p.t=-KB; break;
   case KH: p.j=kH(x)[i]; p.t=-KJ; break;
   case KI: p.j=kI(x)[i]; p.t=-KJ; break;
   case KJ: p.j=kJ(x)[i]; p.t=-KJ; break;
   case KE: p.f=kE(x)[i]; p.t=-KF; break;
   case KF: p.f=kF(x)[i]; p.t=-KF; break;
   default: AT_ERROR("name-value pairs not implemented for ",kname(x->t)," value"); break;
  }
 } else {
  p.v=x; p.t=x->t;
 }
}

B xpair(Pairs &p) {
 if(p.i<0 || p.i>=p.n) return false;
 I i=p.i; p.k=nullptr; K y;
 switch(p.a) {   
  case 1:  //dictionary
   p.k=kS(kK(p.x)[0])[i]; xpair(p,kK(p.x)[1],i); break;
  case 2:  //list of name-value pairs
   y=kK(p.x)[i];
   if(xlen(y)!= 2) {
    AT_ERROR("Name,value pair[",i,"] has ",xlen(y)," elements (expected 2)");
   } else if(y->t==KS) {
    p.k=kS(y)[0]; xpair(p,y,1);
   } else if(!y->t && kK(y)[0]->t==-KS) {
    p.k=kK(y)[0]->s; xpair(p,kK(y)[1],-1);
   } else {
    AT_ERROR("Name,value pair[",i,"] has no name symbol");
   }
   break;
  case 3:  //list of name,value,name,value..
   i*=2; y=kK(p.x)[i];
   if(y->t==-KS) {
    p.k=y->s; xpair(p,kK(p.x)[i+1],-1);
   } else {
    AT_ERROR("Unrecognized name for pair, element[",i,"], expected symbol, received: ",kname(y->t));
   }
   break;
  case 4:  // list of symbols
    i*=2; p.k=kS(p.x)[i]; xpair(p,p.x,i+1); break;
  default: AT_ERROR("Unrecognized name-value argument"); break;
 }
 return p.i++, true;
}

J xargc(K x,J i,Pairs &p) { // x:arg(s), i:offset, -1 if not applicable, p:pairs to initiate
 if(!x) {
  return 0;
 } else if(xdict(x)) {
  return xpairs(x,p), 0;             // dictionary of options, no other args to process
 } else if(x->t<0 || x->t>97) {
  return i>1 ? 0 : (i<0 ? 1 : 1-i);  // scalar arg, or table or different type of dictionary
 } else if(!x->n) {
  return 0;                          // return 0 for any empty list
 } else if(!(-1<i && i<=x->n)) {
  AT_ERROR("invalid offset: ",i,", for ",kname(x->t)," of length ",x->n);
 } else {
  return x->n-i-xpairs(x,x->n-1,p);  // subtract pairs from regular args to process
 }
}

B xnone(K x,J i) {Pairs p; return !(xargc(x,i,p) || p.n);}

// ------------------------------------------------------------------------------------------
// perr - signal error in type of value given w'name-value pair
// plen  - signal length mismatch of input list if non-negative length supplied
// psym - check for symbol value in name/value pair, return symbol, else error
// ptype - check for symbol value that matches defined data type, e.g. `long`float`double
// pbool - check for integral scalar with value 1 or 0, return true/false
// plong - check for integral scalar, return long int
// pdouble - check if numeric scalar, return double
// pnum - check for long/double, set torch scalar
// psize - check if integral scalar or list, set JRef or ExpandingArray, else error
// pten - attempt to define a tensor from provided scalar or array
// ------------------------------------------------------------------------------------------
V perr(const Pairs &p,cS s) {AT_ERROR("Option: ",p.k," is a ",kname(p.t),", expected a ",s);}

ZV plen(const Pairs &p,J n,J m) {
 if(n==0 && (p.t<0 || m)) {
   AT_ERROR("Option: ",p.k," requires zero elements, but single scalar value supplied");
 } else if(n>0 && (p.t>=0 && m!=n)) {
  AT_ERROR("Option: ",p.k," requires ",n," elements, but ",m," supplied");
 }
}

S psym(const Pairs &p) {if(p.t!=-KS) perr(p,"symbol"); return p.s;}
ScalarType ptype(const Pairs &p) {if(p.t!=-KS) perr(p,"symbol"); return torch::typeMetaToScalarType(mtype(p.s));}
B pbool(const Pairs &p) {if(p.t!=-KB) perr(p,"boolean"); return p.b;}
J plong(const Pairs &p) {if(p.t!=-KJ) perr(p,"long integer"); return p.j;}

F pdouble(const Pairs &p) {
 if(!(p.t==-KJ || p.t==-KF)) perr(p,"float, double or integer scalar");
 return p.t==-KJ ? p.j : p.f;
}

V pnum(const Pairs &p,torch::Scalar &s) {
 switch(p.t){
  case -KJ: s=(int64_t)p.j; break;
  case -KF: s=p.f; break;
  default: perr(p,"number"); break;
 }
}

V psize(const Pairs &p,JRef &s,J n) {
 if(p.t==-KJ)
  s=JRef((int64_t*)&p.j,1);  // recast for linux clang/gcc to go from J* -> int64_t*
 else if(!(p.t==KJ && xsize(p.v,s)))
  perr(p,"a long integer scalar or list");
 plen(p,n,s.size());
}

V psize(const Pairs &p,J d,int64_t *a) {
 if(p.t == -KJ) {
   for(J i=0;i<d;++i) a[i]=p.j;
 } else if(p.t == KJ) {
  if(d == xlen(p.v))
   for(J i=0;i<d;++i) a[i]=kJ(p.v)[i];
  else
   plen(p,d,xlen(p.v));
 } else {
  perr(p,"long integer scalar or list");
 }
}

V psize(const Pairs &p,J d,F *a) {
 if(p.t == -KF) {
   for(J i=0;i<d;++i) a[i]=p.f;
 } else if(p.t == KF) {
  if(d == xlen(p.v))
   for(J i=0;i<d;++i) a[i]=kF(p.v)[i];
  else
   plen(p,d,xlen(p.v));
 } else {
  perr(p,"double precision scalar or list");
 }
}

V pten(const Pairs &p,Tensor &t) {
 switch(p.t) {
  case 0: if(!(xten(p.v,t))) t=kput(p.v); break;
  case -KB: t=torch::full({},Scalar(p.b)).to(maptype(KB)); break;
  case -KJ: t=torch::full({},Scalar((int64_t)p.j)).to(maptype(KJ)); break;
  case -KF: t=torch::full({},Scalar(p.f)).to(maptype(KF)); break;
  case KB:
  case KH:
  case KI:
  case KJ:
  case KE:
  case KF: t=kput(p.v); break;
  default: perr(p,"numeric scalar/array or previously allocated tensor pointer");
 }
}

// -----------------------------------------------------------------------------------------
// kcast - given data type and array, cast and return, i.e. 1h$x
// kbool - cast k value to boolean
// kdict - tensor dictionary to k dictionary of names -> tensor values
// kfind - given list of symbols, find index of matching string, return -1 if not found
// klist - return k value from count and long/double pointer
// kex - true if given list is one unique value
// kexpand - given element count & data ptr from expanding array return scalar or list
// -----------------------------------------------------------------------------------------
K kcast(A t,K x) {return k(0,(S)"$",kh(t),r1(x),0);}
K kbool(K x) {return kcast(1,x);}

K kdict(const TensorDict &d) {
 K x=xD(ktn(KS,0),ktn(0,0));
 for(auto &a:d) dictadd(x,a.key().c_str(),kget(a.value()));
 return x;
}

J kfind(K k,const std::string &s) {
 if(k->t != KS) AT_ERROR("Unable to look up `",s," in ",kname(k->t),", expecting symbols");
 for(J i=0; i<k->n; ++i) if(!s.compare(kS(k)[i])) return i;
 return -1;
}

K klist(J n,const int64_t *j) {K x=ktn(KJ,n); memcpy(kG(x),j,n*sizeof(int64_t)); return x;}
K klist(J n,const F       *f) {K x=ktn(KF,n); memcpy(kG(x),f,n*sizeof(F));       return x;}

template<typename T>Z B kex(J n,const T *e) {
 B b=n>0; for(I i=1;i<n;++i) if(e[i-1]!=e[i]) return false; return b;
}

K kexpand(J n,const int64_t *e) {return kex<int64_t>(n,e) ? kj(e[0]) : klist(n,e);}
K kexpand(J n,const F       *e) {return kex<F>      (n,e) ? kf(e[0]) : klist(n,e);}

// -----------------------------------------------------------------------------------------
// kfree - free allocated object according to tag
// kto - convert tensor/module device and or data type, e.g. to[tensor;`cuda`float;0b]
// kdetail - return dictionary of attributes of given object and level of detail
// kzerograd - return dictionary of attributes of given object and level of detail
// -----------------------------------------------------------------------------------------
KAPI kfree(K x){
 KTRY
  Ptr p=nullptr;
  switch(xptr(x,p) ? p->t : Class::undefined) {
   case Class::tensor:     delete (Tensor*)p->v; break;
   case Class::sequential: delete (Sequential*)p->v; break;
   case Class::optimizer:  optfree(p->c,p->v); break;
   default: return KERR("Not a recognized pointer");
  }
  return delete p,(K)0;
 KCATCH("Unable to free object")
}

KAPI kto(K x,K y,K z) {
 KTRY
  B b=false; Ptr p; Tensor t; TensorOptions o;
  if(!(xptr(x,p))) {
   AT_ERROR("1st argument is a ",kname(x->t),", expected allocated tensor or module");
  } else if(!(xto(y,o) || xten(y,t))) {
   AT_ERROR("2nd argument is a ",kname(y->t),", expected tensor or tensor option(s), e.g. `cuda`float");
  } else if(!(xbool(z,b))) {
   AT_ERROR("3rd argument is a ",kname(z->t),", expected boolean flag for async");
  } else {
   if(t.defined())
    o=o.device(t.device()).dtype(t.dtype());
   if(!(o.has_device() || o.has_dtype()))
    AT_ERROR("No device or datatype specified");
   switch(p->t) {
    case Class::tensor:     ktento(p,o,b); break;
    case Class::sequential: kseqto(p,o,b); break;
    default: AT_ERROR("Unrecognized pointer from k, expecting allocated tensor or module");
   }
  }
  return (K)0;
 KCATCH("to");
}

KAPI kdetail(K x) {
 KTRY
  I n=0; Ptr p=nullptr;
  if(xptr(x,p) || (xptr(x,0,p) && xlevel(x,1,n) && x->n==2)) {
   if(n<0 || n>2)
    return KERR("Specify level of detail: 0,1,2");
   switch(p->t) {
    case Class::tensor:  return tensordetail(p,n);
    default:           return KERR("Unrecognized pointer");
   }
  } else {
   return KERR("Expected argument of ptr or (ptr;level)");
  }
 KCATCH("Unable to get detail")
}

KAPI kzerograd(K x) {
 KTRY
  Ptr p=nullptr;
  switch(xptr(x,p) ? p->t : Class::undefined) {
   case Class::tensor: {auto *t=(Tensor*)p->v; if(t->grad().defined()) t->grad().detach().zero_(); break;}
   case Class::sequential:(*(Sequential*)p->v)->zero_grad(); break;
   case Class::optimizer:  ((Optimizer*) p->v)->zero_grad(); break;
   default: AT_ERROR("Expecting pointer to tensor, module or optimizer");
  }
  return (K)0;
 KCATCH("zero gradients");
}


// ---------------------------------------------------------------------------------------------
// cudadevices - return number of CUDA devices enabled or available CUDA device symbols
// cudadevice - k interface to set/query current CUDA device, e.g. `cuda:0 
// ---------------------------------------------------------------------------------------------
KAPI cudadevices(K x) {
 if(xnull(x)) {
  return kj(env().cuda);
 } else if(xempty(x)) {
  K s=ktn(KS,0);
  for(auto& m:env().device) if((std::get<1>(m)).is_cuda()) js(&s,std::get<0>(m));
  return s;
 } else {
  return KERR("cudadevices[] returns count of available GPUs, cudadevices() returns CUDA syms");
 }
}

KAPI cudadevice(K x) {
 KTRY
  torch::Device d(torch::kCUDA);
  auto *g = c10::impl::getDeviceGuardImpl(d.type());
  if(!env().cuda) {
   return KERR("No CUDA device available");
  } else if(xempty(x)) {
   for(auto &m:env().device)
    if(g->getDevice()==std::get<1>(m)) return ks(std::get<0>(m));
   AT_ERROR("Unable to map CUDA device: ",g->getDevice().index()," to symbol");
  } else if(xdev(x,d) && d.is_cuda() && d.has_index()) {
   return g->setDevice(d), K(0);
  } else {
   return KERR("Unrecognized CUDA device, expecting cuda with valid device number, e.g. `cuda:0");
  }
 KCATCH("Unable to query/set CUDA device")
}

// ---------------------------------------------------------------------------------------------
// optsym - given tensor options, return underlying device,data type,layout & grad/nograd as sym
// optmap - given tensor options, return dictionary of attribute -> setting
// ---------------------------------------------------------------------------------------------
S& optsym(const torch::Device& d) {
 for(auto &m:env().device) if(d==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized device: ",d);
}

S& optsym(const TypeMeta& t) {
 for(auto &m:env().dtype) if(t==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized data type: ",t);
}

S& optsym(const torch::Layout& l) {
 for(auto &m:env().layout) if(l==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized layout: ",l);
}

S& optsym(const bool& g) {
 for(auto &m:env().gradient) if(g==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized gradient setting: ",g);
}

K optmap(const TensorOptions &o) {
 K a=ktn(KS,4),b=ktn(KS,4);
 kS(a)[0]=cs("device");   kS(b)[0]=optsym(o.device());
 kS(a)[1]=cs("dtype");    kS(b)[1]=optsym(o.dtype());
 kS(a)[2]=cs("layout");   kS(b)[2]=optsym(o.layout());
 kS(a)[3]=cs("gradient"); kS(b)[3]=optsym(o.requires_grad());
 return xD(a,b);
}

// ---------------------------------------------------------------------------------------------
// kdefault - k interface to query/set default tensor options
// ksetting - list/change configuration settings
// config - print or return strings of pytorch config (CUDA capability, build options, etc.)
// ---------------------------------------------------------------------------------------------
KAPI kdefault(K x) {
 torch::TensorOptions a,o;
 KTRY
  if(xempty(x)) {
   return optmap(o);
  } else if(xopt(x,o)) {
   if(a.device()!=o.device() || a.layout()!=o.layout() || a.requires_grad()!=o.requires_grad())
    AT_ERROR("Currently, only default data type can be reset");
   torch::set_default_dtype(o.dtype());
   return(K)0;
  } else {
   return KERR("Unrecognized argument for querying/setting default tensor options");
  }
 KCATCH("Unable to query/set default tensor option(s)");
}

KAPI ksetting(K x) {
 KTRY
  auto &e=env(); auto &c=at::globalContext(); B b,o=torch::hasOpenMP(); J n; S s;
  if(xempty(x)) {
   K r=xD(ktn(KS,0),ktn(0,0)),*s=&kK(r)[0],*v=&kK(r)[1];
   js(s,cs("mkl"));            jk(v,kb(torch::hasMKL()));
   js(s,cs("openmp"));         jk(v,kb(o));
   js(s,cs("threads"));        jk(v,kj(o ? torch::get_num_threads() : 1));
   js(s,cs("cuda"));           jk(v,kb(torch::cuda::is_available()));
   js(s,cs("magma"));          jk(v,kb(torch::hasMAGMA()));
   js(s,cs("cudnn"));          jk(v,kb(torch::cuda::cudnn_is_available()));
   js(s,cs("cudnnversion"));   jk(v,kj(torch::cuda::cudnn_is_available()
                                    ? at::detail::getCUDAHooks().versionCuDNN() : nj));
   js(s,cs("cudadevices"));    jk(v,kj(e.cuda));
   js(s,cs("benchmark"));      jk(v,kb(c.benchmarkCuDNN()));
   js(s,cs("deterministic"));  jk(v,kb(c.deterministicCuDNN()));
   js(s,cs("stackframe"));     jk(v,kb(e.frame));
   js(s,cs("alloptions"));     jk(v,kb(e.alloptions));
   return r;
  } else if (xsym(x,0,s) && xbool(x,1,b) && x->n==2) {
   if(s==cs("benchmark"))           c.setBenchmarkCuDNN(b);
   else if(s==cs("deterministic"))  c.setDeterministicCuDNN(b);
   else if(s==cs("stackframe"))     e.frame=b;
   else if(s==cs("alloptions"))     e.alloptions=b;
   else                             AT_ERROR("Unable to change setting: ",s);
   return(K)0;
  } else if (xsym(x,0,s) && s==cs("threads") && xlong(x,1,n) && x->n==2) {
   if(!o) AT_ERROR("Unable to set number of threads, OpenMP not available");
   torch::set_num_threads(n);
   return(K)0;
  } else {
   return KERR("Unrecognized arg(s) -- use empty arg to query, use (sym;bool) to set one of `benchmark`deterministic`stackframe`alloptions or (`threads;n) for threads");
  }
 KCATCH("Unable to query/change torch settings");
}

KAPI config(K x) {
 KTRY
  auto c=torch::show_config();
  if(xnull(x)) {
   std::cerr << c << "\n";
   return (K)0;
  } else if(xempty(x)) {
   std::stringstream s(c); std::string t; K z=ktn(0,0);
   while(std::getline(s,t,'\n')) jk(&z,kp((S)t.c_str()));
   return z;
  } else {
   return KERR("config expects empty argument: config[] prints to stderr, config() returns strings");
  }
 KCATCH("config");
}

// -----------------------------------------------------------------------------------
// deviceseed - query/set seed for given device, return initial seed in use for device
// seedmap - returns map of device sym -> seed
// kseed - k interface to query/set device seed or query/reset seed for all devices
// -----------------------------------------------------------------------------------
J deviceseed(torch::Device &d, B b=false,J s=0) { // d:device, b:set flag, s:seed to set
 torch::DeviceGuard dg(d);
 auto &g=at::globalContext().defaultGenerator(d.is_cuda() ? torch::kCUDA : torch::kCPU);
 if(b) {
  if (s==nj)
   g.seed();
  else
   g.set_current_seed(s);
 }
 return g.current_seed();
}

ZK seedmap(V) {
 auto a=env().device; auto n=a.size(); I i=0; K k=ktn(KS,n),v=ktn(KJ,n);
 for(auto& m:a)
  kS(k)[i]=std::get<0>(m),kJ(v)[i++]=deviceseed(std::get<1>(m));
 return xD(k,v);
}

KAPI kseed(K x) {
 KTRY
  torch::Device d(torch::DeviceType::CPU); J s;
  if(xempty(x)) {                 // if empty, report on seed for all devices
   return seedmap();
  } else if(xlong(x,s)) {         // set single random seed across all devices
   if(s==nj) s=at::detail::getNonDeterministicRandom();
   torch::manual_seed(s);
   return (K)0;
  } else if(xdev(x,d)) {          // query initial random seed for given device
   return kj(deviceseed(d));
  } else if(xdev(x,0,d) && xlong(x,1,s) && x->n==2) {  // set seed for given device
   deviceseed(d,true,s);
   return (K)0;
  } else {
   return KERR("Unrecognized arg(s) for seed, expected one of: device, seed or (device;seed)");
  }
 KCATCH("Unable to set/retrieve random seed(s)");
}

// -----------------------------------------------------------------------------------------
// initialize globals: device counts, device sym-int mapping, etc.
// kinit - called when shared library is first opened
// -----------------------------------------------------------------------------------------
Env& env() {static Env e; return e;}

ZV kinit() __attribute__((constructor));

ZV kinit() {
 C c[16]; auto &e=env(); auto &d=e.device;
 e.frame = false;                                                     //no stack frame on error msg
 e.cuda = torch::cuda::device_count();                                //count of available CUDA devices
 d.emplace_back(cs("cpu"),torch::Device(torch::DeviceType::CPU));     //build map from sym->device
 if(e.cuda) {
  d.emplace_back(cs("cuda"),torch::Device(torch::DeviceType::CUDA));  //current CUDA device, `cuda
  for(I i=0; i<e.cuda; ++i) {
   sprintf(c,"cuda:%d",i);                                            //device 0-n, e.g. `cuda:0
   d.emplace_back(ss(c),torch::Device(torch::DeviceType::CUDA,i));
  }
 }
}

// -----------------------------------------------------------------------------------------
// fn - given dictionary, along with name, fn & arg count, adds function to dictionary
// fns - returns K dictionary with function names and code
// -----------------------------------------------------------------------------------------
V fn(K x,cS s,V *f,I n){dictadd(x,s,dl(f,n));}

KAPI fns(K x){
 x=xD(ktn(KS,0),ktn(0,0));
 fn(x, "free",        KFN(kfree),       1);
 fn(x, "to",          KFN(kto),         3);
 fn(x, "detail",      KFN(kdetail),     1);
 fn(x, "zerograd",    KFN(kzerograd),   1);
 fn(x, "default",     KFN(kdefault),    1);
 fn(x, "setting",     KFN(ksetting),    1);
 fn(x, "config",      KFN(config),      1);
 fn(x, "cudadevice",  KFN(cudadevice),  1);
 fn(x, "cudadevices", KFN(cudadevices), 1);
 fn(x, "seed",        KFN(kseed),1);
 tensorfn(x);
 mathfn(x);
 modfn(x);
 lossfn(x);
 optfn(x);
 return x;
}
