#include "ktorch.h"
#include "kloss.h"

// append a loss option to a k dictionary given dict,name & value
#define OPTION(x,k,v) dictadd(x, lset(Setting::k), v)

using Lw  = Tensor (*)(const Tensor&, const Tensor&, const Tensor&, int64_t);           // loss w'wts
using Lwi = Tensor (*)(const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t);  // loss w'wts & ignore ind

// ------------------------------------------------------------------------------------------------------
// kloss - given loss type & shared pointer to newly created loss module, return kptr
// lmap - map to/from sym to loss function name, e.g. `mse <-> Cast::mse
// lset - map to/from sym to loss setting enum, e.g. `reduce <-> Setting::reduce
// ------------------------------------------------------------------------------------------------------
K kloss(Cast x,const Lossptr& y) {return kptr(new Kloss(x,y));}
K kloss(Cast c,const AnyModule& m) {return kptr(new Kmodule(Class::loss,c,m));}

static Cast lmap(S s) {
 for(auto&m:env().loss)
  if(std::get<0>(m)==s) return std::get<1>(m);
 AT_ERROR("Unrecognized loss function: ",s);
}

static S lmap(Cast c) {
 for(auto&m:env().loss)
  if(std::get<1>(m)==c) return std::get<0>(m);
 AT_ERROR("Unrecognized loss function: ",(I)c);
}

static S lset(Setting s) {
 for(auto&m:env().lset)
  if(std::get<1>(m)==s) return std::get<0>(m);
 AT_ERROR("Unrecognized loss setting: ",(I)s);
}

static Setting lset(S s) {
 for(auto&m:env().lset)
  if(std::get<0>(m)==s) return std::get<1>(m);
 AT_ERROR("Unrecognized loss setting: ",s);
}

// ----------------------------------------------------------------------------------------------------
// input checking fns with error msg specific to loss module name and setting
// check positional or name-value pairs for lbool->boolean, lsym->sym, int64-integer, ldouble..
// ----------------------------------------------------------------------------------------------------
static bool lbool(K x,J i,Cast c,Setting s) {
 bool b;
 TORCH_CHECK(xbool(x,i,b), lmap(c)," ",lset(s),": expected boolean scalar, given ",kname(x,i));
 return b;
}

static bool lbool(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KB, lmap(c)," ",p.k,": expected boolean scalar, given ",kname(p.t));
 return p.b;
}

static S lsym(K x,J i,Cast c,Setting s) {
 S sy;
 TORCH_CHECK(xsym(x,i,sy), lmap(c)," ",lset(s),": expected symbol, given ",kname(x,i));
 return sy;
}

static S lsym(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KS, lmap(c)," ",p.k,": expected symbol, given ",kname(p.t));
 return p.s;
}

static int64_t int64(K x,J i,Cast c,Setting s) {
 int64_t n;
 TORCH_CHECK(xint64(x,i,n), lmap(c)," ",lset(s),": expected long scalar, given ",kname(x,i));
 return n;
}

static int64_t int64(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ, lmap(c)," ",p.k,": expected long scalar, given ",kname(p.t));
 return p.j;
}

static double ldouble(K x,J i,Cast c,Setting s) {
 double f;
 TORCH_CHECK(xnum(x,i,f), lmap(c)," ",lset(s),": expected double, given ",kname(x,i));
 return f;
}

static double ldouble(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==-KF, lmap(c)," ",p.k,": expected double, given ",kname(p.t));
 return pdouble(p);
}

// ------------------------------------------------------------------------------------------------------
// rmsg,rmap - message and mapping for loss reduction to/from sym and enumeration
// xreduce - check if sym, if matches loss reduction, set int, e.g. `none -> 0, `mean -> 1, `sum -> 2
// ------------------------------------------------------------------------------------------------------
static std::string rmsg(bool b) {
 std::string s;
 for(auto&m:env().reduce)
  s += (b ? std::get<0>(m) : std::to_string(std::get<1>(m))) + ",";
 s.pop_back();
 return s;
}

static int64_t rmap(S s) {
 for(auto&m:env().reduce)
  if(std::get<0>(m)==s) return std::get<1>(m);
 AT_ERROR("Unrecognized setting for loss reduction: ",s,", expecting one of ",rmsg(true));
}

static S rmap(int64_t r) {
 for(auto&m:env().reduce)
  if(std::get<1>(m)==r) return std::get<0>(m);
 AT_ERROR("Unrecognized setting for loss reduction: ",r,", expecting one of ",rmsg(false));
}

static bool xreduce(K x,int64_t &r) {
 if(x->t == -KS) return r=rmap(x->s), true;
 return false;
}

static bool xreduce(K x,J i,int64_t &r) {
 if(x->t == KS && -1<x->n && x->n>i)
  return r=rmap(kS(x)[i]),true;
 else
  return xind(x,i) && xreduce(kK(x)[i],r);
}

// -----------------------------------------------------------------------------------------------
//  reduction arg uses variant, using functions below to translate sym -> variant value
// -----------------------------------------------------------------------------------------------
using Reduce1=c10::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum>;
using Reduce2=c10::variant<torch::enumtype::kNone, torch::enumtype::kBatchMean, torch::enumtype::kSum, torch::enumtype::kMean>;

static void reduce(Reduce1& r,Cast c,S s) {
 switch(emap(s)) {
  case Enum::none: r=torch::kNone; break;
  case Enum::mean: r=torch::kMean; break;
  case Enum::sum:  r=torch::kSum; break;
  default: AT_ERROR(lmap(c)," reduce:",s," is not one of none,mean,sum");
 }
}

static void reduce(Reduce2& r,Cast c,S s) {
 switch(emap(s)) {
  case Enum::none:      r=torch::kNone; break;
  case Enum::batchmean: r=torch::kBatchMean; break;
  case Enum::mean:      r=torch::kMean; break;
  case Enum::sum:       r=torch::kSum; break;
  default: AT_ERROR(lmap(c)," reduce:",s," is not one of none,batchmean,mean,sum");
 }
}

// ------------------------------------------------------------------------------------------------------
// reduce - return default reduction mode, or process given arg(s) & offset to return reduction mode
// lossfunc - call loss function with x,y tensors/arrays and optional reduction mode
// bce - binary cross entropy has option of batch weights, so function parses (x;y) or (x;y;wt)
// ------------------------------------------------------------------------------------------------------
template<typename O> static O reduce(K x,J i,Cast c) {
 O o; Pairs p; J n=xargc(x,i,p); S s=nullptr;
 TORCH_CHECK(n<2, lmap(c),": only 1 positional argument(reduce) expected, ",n," given");
 if(n==1) s=lsym(x,i,c,Setting::reduce);
 while(xpair(p)) {
  TORCH_CHECK(lset(p.k)==Setting::reduce, "Unrecognized option: ",p.k,", ",lmap(c)," loss expects single option: reduce");
  s=lsym(p,c);
 }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

template<typename O> static void reduce2(bool a,K x,const O& o,const O d=O());
template<typename O> static void reduce2(bool a,K x,const O& o,const O d) {
 if(a || d.reduction().index() != o.reduction().index())
  OPTION(x, reduce, ks(ESYM(o.reduction())));
}

int64_t reduce() {return Reduction::Mean;}

int64_t reduce(const char* s,K x,J i) { // check argument(s) for sym or named pair/dict, e.g. (`reduce;`mean))
 Pairs p; J n=xargc(x,i,p); auto r=reduce();
 if(!(n==0 || (n==1 && xreduce(x,i,r))))
  AT_ERROR("Unrecognized argument(s) for ",s," loss");
 while(xpair(p))
  if(lset(p.k) == Setting::reduce)
   r=rmap(psym(p));
  else
   AT_ERROR("Unrecognized option: ",p.k,", ",s," loss expects single option: reduce");
 return r;
}

static K lossfunc(K a,Cast c) {
 KTRY
  namespace nn=torch::nn; namespace f=nn::functional; bool b,p; Tensor r,x,y;
  TORCH_CHECK(a->t>=0, lmap(c),": not implemented for ",kname(a));
  b=a->n==2;
  if(a->t) {
   TORCH_CHECK(b, lmap(c),": loss expects 2-element arg of input & target, ",a->n," value(s) given");
   x=kput(a); y=x[1]; x=x[0]; p=false;
  } else {
   p=xtenarg(a,x,y);
  }
  switch(c) {
   case Cast::kl: r=b ? f::kl_div(x,y) : f::kl_div(x,y,reduce<nn::KLDivLossOptions>(a,2,c)); break;
   case Cast::l1: r=b ? f::l1_loss(x,y) : f::l1_loss(x,y,reduce<nn::L1LossOptions>(a,2,c)); break;
   case Cast::mse: r=b ? f::mse_loss(x,y) : f::mse_loss(x,y,reduce<nn::MSELossOptions>(a,2,c)); break;
   case Cast::multilabel:
    r=b ? f::multilabel_margin_loss(x,y)
        : f::multilabel_margin_loss(x,y,reduce<nn::MultiLabelMarginLossOptions>(a,2,c));
    break;
   case Cast::smoothl1:
    r=b ? f::smooth_l1_loss(x,y) : f::smooth_l1_loss(x,y,reduce<nn::SmoothL1LossOptions>(a,2,c));
    break;
   case Cast::softmargin:
    r=b ? f::soft_margin_loss(x,y) : f::soft_margin_loss(x,y,reduce<nn::SoftMarginLossOptions>(a,2,c));
    break;
   default: AT_ERROR("Unrecognized loss function"); break;
  }
 return kresult(p,r);
 KCATCH("loss");
}

KAPI kl(K x)          {return lossfunc(x, Cast::kl);}
KAPI l1(K x)          {return lossfunc(x, Cast::l1);}
KAPI mse(K x)         {return lossfunc(x, Cast::mse);}
KAPI multilabel(K x)  {return lossfunc(x, Cast::multilabel);}
KAPI smoothl1(K x)    {return lossfunc(x, Cast::smoothl1);}
KAPI softmargin(K x)  {return lossfunc(x, Cast::softmargin);}

static bool bcearg(K x) {return x->t==-KS || x->t==KS || xempty(x) || xdict(x);}  // true if arg is a setting (rather than wt tensor)

KAPI bce(K a) {
 KTRY
  bool p=false; Tensor l,x,y,w;
  if(!a->t && 1<a->n && a->n<5) {
   J n=(a->n==2) ? 2 : (a->n==4 ? 3 : 3-bcearg(kK(a)[2]));
   auto r=reduce("binary cross entropy",a,n);
   p=n==2 ? xtenarg(a,x,y) : xtenarg(a,x,y,w);
   l=torch::binary_cross_entropy(x,y,w,r);
  } else if(0 < a->t && a->t<20 && 1<a->n && a->n<4) {
   x=kput(a);
   l=a->n==2 ? torch::binary_cross_entropy(x[0],x[1]) : torch::binary_cross_entropy(x[0],x[1],x[2]);
  } else {
   AT_ERROR("binary cross entropy loss expects (input;target), (input;target;reduction), (input;target;weights) or (input;target;weights;reduction)");
  }
  return p ? kten(l) : kget(l);
 KCATCH("binary cross entropy")
}

// ------------------------------------------------------------------------------------------------------
// classwt - set optional class weights & reduction mode, also index to ignore for some losses
//           classes with optional index use the same elements, so a templated fn is used,
//           but otheres use "weight" vs "pos_weight", requiring class-specific overloads
// ------------------------------------------------------------------------------------------------------
static void classwt(K x,J i,Cast c,S& s,Tensor& w) {
 Pairs p; J n=xargc(x,i,p); s=nullptr;
 if(n && xsym(x,i+n-1,s)) n--;
 if(n) {n--; if(!xempty(x,i+n) && !xten(x,i+n,w)) w=kput(x,i+n);}
 TORCH_CHECK(!n, lmap(c),": unrecognized positional arg(s), expected weights, reduce mode or (weights;reduce mode)");
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::weight: if(!pempty(p)) pten(p,w); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: AT_ERROR("Unrecognized ",lmap(c)," option: ",p.k); break;
  }
}

static auto& classwt(K x,J i,Cast c,torch::nn::BCEWithLogitsLossOptions&& o) {
 S s; Tensor w; classwt(x,i,c,s,w);
 if(s) reduce(o.reduction(),c,s);
 if(w.defined()) o.pos_weight(w);
 return o;
}

static auto& classwt(K x,J i,Cast c,torch::nn::MultiLabelSoftMarginLossOptions&& o) {
 S s; Tensor w; classwt(x,i,c,s,w);
 if(s) reduce(o.reduction(),c,s);
 if(w.defined()) o.weight(w);
 return o;
}

static void wtargs(Cast c,const char* s,K x,J i,Tensor& w,J& j,int64_t &r) {
 bool b=c==Cast::ce || c==Cast::nll; Pairs p; J n=xargc(x,i,p); j=-100; r=reduce();
 if(n && xreduce(x,i+n-1,r)) n--;
 if(n && xlong(x,i+n-1,j)) {if(b) n--; else AT_ERROR("Index to ignore not expected for ",s," loss");}
 if(n==1) {n--; if(!xten(x,i+n,w) && xlen(kK(x)[i+n])) w=kput(x,i+n);}
 if(n)
  AT_ERROR("Unrecognized arg(s) for ",s," loss");
 while(xpair(p)) {
  switch(lset(p.k)) {
   case Setting::weight: pten(p,w); break;
   case Setting::ignore: if(b) j=plong(p); else AT_ERROR("Index to ignore not expected for ",s," loss"); break;
   case Setting::reduce: r=rmap(psym(p)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for ",s," loss"); break;
  }
 }
}

template<typename O> O classwt(K x,J i,Cast c) {
 O o; Pairs p; J n=xargc(x,i,p); int64_t j; S s=nullptr; Tensor w;
 if(n && xsym(x,i+n-1,s)) n--;
 if(n && xint64(x,i+n-1,j)) n--, o.ignore_index(j);
 if(n) {n--; if(!xempty(x,i+n) && !xten(x,i+n,w)) w=kput(x,i+n);}
 TORCH_CHECK(!n, lmap(c),": unrecognized positional arg(s), expected (weights;ignore index;reduce mode)");
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::weight: if(!pempty(p)) pten(p,w); break;
   case Setting::ignore: o.ignore_index(int64(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for ",lmap(c)," loss"); break;
  }
 if(s) reduce(o.reduction(),c,s);
 if(w.defined()) o.weight(w);
 return o;
}

// ------------------------------------------------------------------------------------------------------
// classwt - get optional class weights & reduction mode, also index to ignore for some losses
// ------------------------------------------------------------------------------------------------------
static void classwt(bool a,K x,const torch::nn::BCEWithLogitsLossOptions& o) {
 if(a || o.pos_weight().defined()) OPTION(x,weight,kget(o.pos_weight()));
 reduce2(a,x,o);
}

static void classwt(bool a,K x,const torch::nn::MultiLabelSoftMarginLossOptions& o) {
 if(a || o.weight().defined()) OPTION(x,weight,kget(o.weight()));
 reduce2(a,x,o);
}

template<typename O> static void classwt(bool a,K x,const O& o) {
 const O d;
 if(a || o.weight().defined()) OPTION(x, weight, kget(o.weight()));
 if(a || d.ignore_index() != o.ignore_index()) OPTION(x, ignore, kj(o.ignore_index()));
 reduce2(a,x,o,d);
}

// ------------------------------------------------------------------------------------------------------
// wtloss - functional form for nll,cross entropy, multi-label soft margin (no c++ version yet)
// ------------------------------------------------------------------------------------------------------
Tensor multilabel_soft_margin_loss(const Tensor& x,const Tensor& y,const Tensor& w,int64_t r) {
 auto l = -(y * torch::log_sigmoid(x) + (1 - y) * torch::log_sigmoid(-x));
 if(w.defined()) l *= w;
 l = l.sum(1) / x.size(1); // only return n=batch size loss values
 switch(r) {
  case Reduction::None: return l;
  case Reduction::Mean: return l.mean();
  case Reduction::Sum:  return l.sum();
  default: AT_ERROR("Unrecognized reduction: ",r);
 }
 // unable to use torch::apply_loss_reduction(l,r), in anonymous namespace in ATen/native/Loss.cpp
}

static K wtloss(K a,Cast c,const char* s) {
 KTRY
  bool p=false; J j; int64_t r; Tensor l,x,y,w;
  if(a->t) {
   AT_ERROR(s," loss not implemented for ",kname(a->t));
  } else if(a->n < 2) {
   AT_ERROR(s," loss expects at least 2 args, (input;target)");
  }
  wtargs(c,s,a,2,w,j,r);
  p=xtenarg(a,x,y);
  switch(c) {
   case Cast::ce:        l=torch::nll_loss(torch::log_softmax(x,1),y,w,r,j); break;
   case Cast::nll:       l=torch::nll_loss(x,y,w,r,j); break;
   case Cast::multisoft: l=multilabel_soft_margin_loss(x,y,w,r); break;
   default: AT_ERROR("Unrecognized loss function: ",(I)c); break;
  }
  return p ? kten(l) : kget(l);
 KCATCH(s)
}

KAPI ce(K x)        {return wtloss(x, Cast::ce,        "cross entropy");}
KAPI nll(K x)       {return wtloss(x, Cast::nll,       "negative log-likelihood");}
KAPI multisoft(K x) {return wtloss(x, Cast::multisoft, "multi-label soft margin");}

// ---------------------------------------------------------------------------------------
// bceloss - handle binary cross-entropy with logits, separate call if batch weights
// bcelogits - input & target, with options for reduction and class weights
// bcelogitw - input, target & batch weights, along with options for reduction & class wts
// ---------------------------------------------------------------------------------------
static K bceloss(K a,bool b,const char* s) {  //a:args, b:true if batch wts
 KTRY
  bool p=false; J i=2+b,j; int64_t r; Tensor x,y,bw,w;
  if(a->t) {
   AT_ERROR(s," loss not implemented for ",kname(a->t));
  } else if(a->n < i) {
   AT_ERROR(s,(b ? " loss expects at least 3 args, (input;target;batch weights)"
                 : " loss expects at least 2 args, (input;target)"));
  }
  wtargs(Cast::bcelogits,s,a,i,w,j,r);
  p=xtenarg(a,x,y);
  if(b && !xten(a,2,bw)) bw=kput(a,2);
  return kresult(p, torch::binary_cross_entropy_with_logits(x,y,bw,w,r));
 KCATCH(s)
}

KAPI bcelogits(K x) {return bceloss(x, false, "binary cross-entropy with logits");}
KAPI bcelogitw(K x) {return bceloss(x, true,  "binary cross-entropy with logits & batch weights");}

// ------------------------------------------------------------------------------------------------------
// marginval  - default margin value given loss type
// margin - get/set optional margin & reduction arguments
// marginloss - functional form of loss functions w'margin & reduction args
// ------------------------------------------------------------------------------------------------------
static double marginval(Cast c) {return c==Cast::hinge ? 1.0 : 0.0;}
static void marginargs(Cast c,const char* s,K x,J i,double &m,int64_t &r) {
 Pairs p; J n=xargc(x,i,p); r=reduce(); m=marginval(c);
 if(n && xreduce(x,i+n-1,r)) n--;
 if(n && xnum(x,i+n-1,m)) n--;
 if(n)
  AT_ERROR("Unrecognized arg(s) for ",s," loss");
 while(xpair(p)) {
  switch(lset(p.k)) {
   case Setting::margin: m=pdouble(p); break;
   case Setting::reduce: r=rmap(psym(p)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for ",s," loss"); break;
  }
 }
}

template<typename O> static O margin(K x,J i,Cast c) {
 O o; Pairs p; J n=xargc(x,i,p); S s=nullptr;
 if(n && xsym(x,i+n-1,s)) n--;
 if(n) n--, o.margin(ldouble(x,i+n,c,Setting::margin));
 TORCH_CHECK(!n, lmap(c),": unrecognized positional arg(s), expected margin,reduce or (margin;reduce), e.g. (1.0;`mean)");
 while(xpair(p))
  switch(lset(p.k)) {
   case Setting::margin: o.margin(ldouble(p,c)); break;
   case Setting::reduce: s=lsym(p,c); break;
   default: AT_ERROR("Unrecognized ",lmap(c)," option: ",p.k); break;
  }
 if(s) reduce(o.reduction(),c,s);
 return o;
}

template<typename O> static void margin(bool a,K x,const O& o) {
 const O d;
 if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
 reduce2(a,x,o,d);
}

static K marginloss(K a,Cast c) {
 KTRY
  namespace nn=torch::nn; namespace f=nn::functional;
  bool b,p=false,h=c==Cast::hinge; Tensor r,x1,x2,y;
  TORCH_CHECK(a->t>=0, lmap(c),": not implemented for ",kname(a));
  TORCH_CHECK(a->n>=3-h, lmap(c), " loss expects (input", (h ? "" : "1;input2"),";target;optional arg(s)..)");
  b=a->n==3-h;
  if(a->t) {
   r=kput(a);
   if(h) x1=r[0], y=r[1];
   else  x1=r[0], x2=r[1], y=r[2];
  } else {
   p=h ? xtenarg(a,x1,y) : xtenarg(a,x1,x2,y);
  }
  switch(c) {
   case Cast::hinge: 
    r=b ? f::hinge_embedding_loss(x1,y) : f::hinge_embedding_loss(x1,y,margin<nn::HingeEmbeddingLossOptions>(a,2,c));
    break;
   case Cast::cosineloss:
    r=b ? f::cosine_embedding_loss(x1,x2,y) : f::cosine_embedding_loss(x1,x2,y,margin<nn::CosineEmbeddingLossOptions>(a,3,c));
    break;
   case Cast::margin:
    r=b ? f::margin_ranking_loss(x1,x2,y) : f::margin_ranking_loss(x1,x2,y,margin<nn::MarginRankingLossOptions>(a,3,c));
    break;
   default: AT_ERROR("Unrecognized loss function"); break;
  }
  return kresult(p,r);
 KCATCH("loss")
}

KAPI hinge(K x)      {return marginloss(x, Cast::hinge);}
KAPI cosineloss(K x) {return marginloss(x, Cast::cosineloss);}
KAPI Margin(K x)     {return marginloss(x, Cast::margin);}

// ------------------------------------------------------------------------------------------------------
// multiargs - process optional power,margin,weight & reduction arguments in given k array
// multimargin - funcional form of multi margin loss function
// ------------------------------------------------------------------------------------------------------
static void multiargs(K x,J i,Scalar &pw,Scalar& m,Tensor& w,int64_t &r) {
 Pairs p; J n=xargc(x,i,p); MultiMarginLossOptions o;
 r=o.reduce(); pw=o.p(); m=o.margin();
 if(n && xnum(x,i,pw)) i++,n--;
 if(n && xnum(x,i,m))  i++,n--;
 if(n && xreduce(x,i+n-1,r)) n--;
 if(n==1) {n--; if(!xten(x,i+n,w) && xlen(kK(x)[i+n])) w=kput(x,i+n);}
 if(n)
  AT_ERROR("Unrecognized arg(s) for multi-margin loss");
 while(xpair(p)) {
  switch(lset(p.k)) {
   case Setting::p:      pnum(p,pw); break;
   case Setting::margin: pnum(p,m); break;
   case Setting::weight: pten(p,w); break;
   case Setting::reduce: r=rmap(psym(p)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for multi-margin loss"); break;
  }
 }
}

static void multi(bool a,K x,const torch::nn::MultiMarginLossOptions& o) {
 const torch::nn::MultiMarginLossOptions d;
 if(a || d.p()      != o.p())      OPTION(x, p,      kj(o.p()));
 if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
 if(a || o.weight().defined())     OPTION(x, weight, kget(o.weight()));
 reduce2(a,x,o,d);
}

KAPI multimargin(K a) {
 KTRY
  bool b; Tensor x,y,w; Scalar p,m; int64_t r; 
  if(a->t || a->n<2 || a->n>6)
   AT_ERROR("multi-margin loss expects 2-6 args: (input;target;p;margin;weight;reduction)");
  multiargs(a,2,p,m,w,r); b=xtenarg(a,x,y);
  return kresult(b, torch::multi_margin_loss(x,y,p,m,w,r));
 KCATCH("multi-margin");
}

// ------------------------------------------------------------------------------------------------------
// triargs - process optional margin,p,eps,swap flag & reduction args in k array for triplet loss
// triplet  - funcional form of triplet margin loss function
// ------------------------------------------------------------------------------------------------------
static void triargs(K x,J i,double& m,double& pw,double& e,bool& s,int64_t& r) {
 Pairs p; J n=xargc(x,i,p); TripletLossOptions o; 
 m=o.margin(); pw=o.p(); e=o.eps(); s=o.swap(); r=o.reduce();
 if(n && xnum(x,i,m))  i++,n--;
 if(n && xnum(x,i,pw)) i++,n--;
 if(n && xdouble(x,i,e))  i++,n--;
 if(n && xbool(x,i,s)) i++,n--;
 if(n && xreduce(x,i,r)) i++,n--;
 if(n)
  AT_ERROR("Unrecognized arg(s) for triplet margin loss");
 while(xpair(p)) {
  switch(lset(p.k)) {
   case Setting::margin: m=pdouble(p); break;
   case Setting::p:      pw=pdouble(p); break;
   case Setting::eps:    e=pdouble(p); break;
   case Setting::swap:   s=pbool(p); break;
   case Setting::reduce: r=rmap(psym(p)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for triplet margin loss"); break;
  }
 }
}

static void triplet(bool a,K x,const torch::nn::TripletMarginLossOptions& o) {
 const torch::nn::TripletMarginLossOptions d;
 if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
 if(a || d.p()      != o.p())      OPTION(x, p,      kf(o.p()));
 if(a || d.eps()    != o.eps())    OPTION(x, eps,    kf(o.eps()));
 if(a || d.swap()   != o.swap())   OPTION(x, swap,   kb(o.swap()));
 reduce2(a,x,o,d);
}

KAPI Triplet(K a) {
 KTRY
  bool b,s; double e,m,p; Tensor x,y,z; int64_t r; 
  if(a->t) {
   AT_ERROR("triplet margin loss not implemented for ",kname(a->t));
  } else if(a->n < 3) {
   AT_ERROR("triplet margin loss expects at least 3 args, (anchor;positive;negative)");
  }
  triargs(a,3,m,p,e,s,r);
  b=xtenarg(a,x,y,z);
  return kresult(b, torch::triplet_margin_loss(x,y,z,m,p,e,s,r));
 KCATCH("triplet margin");
}

// ------------------------------------------------------------------------------------------------------
// poissonargs - process optional margin,p,eps,swap flag & reduction args in k array for triplet loss
// poissonloss  - funcional form of poisson nll loss function
// ------------------------------------------------------------------------------------------------------
static void poissonargs(K x,J i,bool& l,bool& f,double& e,int64_t& r) {
 Pairs p; J n=xargc(x,i,p); PoissonLossOptions o; 
 l=o.log(); f=o.full(); e=o.eps(); r=o.reduce();
 if(n && xbool(x,i,l))   i++,n--;
 if(n && xbool(x,i,f))   i++,n--;
 if(n && xdouble(x,i,e)) i++,n--;
 if(n && xreduce(x,i,r)) i++,n--;
 if(n)
  AT_ERROR("Unrecognized arg(s) for poisson nll loss");
 while(xpair(p)) {
  switch(lset(p.k)) {
   case Setting::log:    l=pbool(p); break;
   case Setting::full:   f=pbool(p); break;
   case Setting::eps:    e=pdouble(p); break;
   case Setting::reduce: r=rmap(psym(p)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for poisson nll loss"); break;
  }
 }
}

static void poisson(bool a,K x,const torch::nn::PoissonNLLLossOptions& o) {
 const torch::nn::PoissonNLLLossOptions d;
 if(a || d.log_input() != o.log_input()) OPTION(x, log,  kb(o.log_input()));
 if(a || d.full()      != o.full())      OPTION(x, full, kb(o.full()));
 if(a || d.eps()       != o.eps())       OPTION(x, eps,  kf(o.eps()));
 reduce2(a,x,o,d);
}

KAPI poissonloss(K a) {
 KTRY
  bool p,ln,f; double e; Tensor x,y; int64_t r; 
  if(a->t) {
   AT_ERROR("poisson nll loss not implemented for ",kname(a->t));
  } else if(a->n < 2) {
   AT_ERROR("poisson nll loss expects at least 2 args, (input;target)");
  }
  poissonargs(a,2,ln,f,e,r); p=xtenarg(a,x,y);
  return kresult(p, torch::poisson_nll_loss(x,y,ln,f,e,r));
 KCATCH("poisson nll loss");
}

// -------------------------------------------------------------------------------------------------------------------
// ctc1 - process args for CTC loss, blank value, flag for setting infinities -> zero & reduction method
// ctc2 - check inputs for tensor or arrays for input/target lengths (different forward call)
// ctc - funcional form of connectionist temporal classification loss between continuous time series & target sequence
// -------------------------------------------------------------------------------------------------------------------
static void ctc1(K x,J i,int64_t& b,bool& z,int64_t& r) {
 Pairs p; J n=xargc(x,i,p); CTCLossOptions o; 
 b=o.blank(); z=o.zeroinf(); r=o.reduce();
 while(n) {
  if(xint64(x,i,b) || xbool(x,i,z) || xreduce(x,i,r))
   i++,n--;
  else
   AT_ERROR("Unrecognized argument(position ",i,") for CTC loss");
 }
 while(xpair(p)) {
  switch(lset(p.k)) {
   case Setting::blank:   b=plong(p); break;
   case Setting::zeroinf: z=pbool(p); break;
   case Setting::reduce:  r=rmap(psym(p)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for CTC loss"); break;
  }
 }
}

static bool ctc2(K a,J i,Tensor& x,Tensor& y,Tensor& nx,Tensor& ny,IntArrayRef& jx,IntArrayRef& jy) {
  bool p=xtenarg(a,i,x,y);
  if(!(xsize(a,i+2,jx) && xsize(a,i+3,jy))) { // unless both lenghts given as k arrays
    if(!xten(a,i+2,nx)) nx=kput(a,2);  // define input lengths as tensor
    if(!xten(a,i+3,ny)) ny=kput(a,3);  // define target lengths as tensor
  }
  return p;
}

static void ctc(bool a,K x,const torch::nn::CTCLossOptions& o) {
 const torch::nn::CTCLossOptions d;
 if(a || d.blank()         != o.blank())         OPTION(x, blank,   kj(o.blank()));
 if(a || d.zero_infinity() != o.zero_infinity()) OPTION(x, zeroinf, kb(o.zero_infinity()));
 reduce2(a,x,o,d);
}

KAPI Ctc(K a) {
 KTRY
  bool p,z; IntArrayRef jx,jy; Tensor x,y,nx,ny; int64_t b,r; 
  if(a->t) {
   AT_ERROR("CTC loss not implemented for ",kname(a->t));
  } else if(a->n < 4) {
   AT_ERROR("CTC loss expects at least 4 args, (input;target;input lengths;target lengths)");
  }
  ctc1(a,4,b,z,r); 
  p=ctc2(a,0,x,y,nx,ny,jx,jy);
  return kresult(p, nx.defined() ? torch::ctc_loss(x,y,nx,ny,b,r,z) : torch::ctc_loss(x,y,jx,jy,b,r,z));
 KCATCH("CTC loss");
}

// ---------------------------------------------------------------------------------------------------
// lossinit - initialize loss modules, return k pointer
// lossopt - retrieve loss module options, return k dictionary of symbol -> setting
// lossdict - dictionary of loss module & options or full state (w'class, empty name, parms & buffers)
// lossfwd - given loss object, calls forward function on remaining inputs and returns loss
// lossto - given loss object and device/data type, converts tensors in options (e.g. class weights)
// losswt - return class wts if tensor is defined (used to determine device/datatype)
// loss - main api function that creates/calls loss objects and queries their properties
// ---------------------------------------------------------------------------------------------------
static K lossinit(S s,K x,J i) {
 J j; double m; Cast c=lmap(s); Tensor w; int64_t r; Lossptr a;
 switch(c) {
  case Cast::bce:         a=std::make_shared<BCELoss>(reduce(s,x,i)); break;
  case Cast::kl:          a=std::make_shared<KLDivLoss>(reduce(s,x,i)); break;
  case Cast::l1:          a=std::make_shared<L1Loss>(reduce(s,x,i)); break;
  case Cast::mse:         a=std::make_shared<MSELoss>(reduce(s,x,i)); break;
  case Cast::multilabel:  a=std::make_shared<MultiLabelMarginLoss>(reduce(s,x,i)); break;
  case Cast::smoothl1:    a=std::make_shared<SmoothL1Loss>(reduce(s,x,i)); break;
  case Cast::softmargin:  a=std::make_shared<SoftMarginLoss>(reduce(s,x,i)); break;

  case Cast::bcelogits:   wtargs(c,s,x,i,w,j,r); a=std::make_shared<BCEWithLogitsLoss>(w,r); break;
  case Cast::multisoft:   wtargs(c,s,x,i,w,j,r); a=std::make_shared<MultiLabelSoftMarginLoss>(w,r); break;
  case Cast::ce:          wtargs(c,s,x,i,w,j,r); a=std::make_shared<CrossEntropyLoss>(w,j,r); break;
  case Cast::nll:         wtargs(c,s,x,i,w,j,r); a=std::make_shared<NLLLoss>(w,j,r); break;

  case Cast::hinge:       marginargs(c,s,x,i,m,r); a=std::make_shared<HingeEmbeddingLoss>(m,r); break;
  case Cast::cosineloss:  marginargs(c,s,x,i,m,r); a=std::make_shared<CosineEmbeddingLoss>(m,r); break;
  case Cast::margin:      marginargs(c,s,x,i,m,r); a=std::make_shared<MarginRankingLoss>(m,r); break;

  case Cast::multimargin: {Scalar p,m;        multiargs(x,i,p,m,w,r);   a=std::make_shared<MultiMarginLoss>(p,m,w,r); break;}
  case Cast::triplet:     {bool s;double e,p; triargs(x,i,m,p,e,s,r);   a=std::make_shared<TripletMarginLoss>(m,p,e,s,r); break;}
  case Cast::poissonloss: {bool l,f;double e; poissonargs(x,i,l,f,e,r); a=std::make_shared<PoissonNLLLoss>(l,f,e,r); break;}
  case Cast::ctc:         {bool z;int64_t b;  ctc1(x,i,b,z,r);          a=std::make_shared<CTCLoss>(b,z,r); break;}
  default: AT_ERROR("Unrecognized loss function: ",s); break;
 }
 return kloss(c,a);
}

static AnyModule lossinit2(S s,Cast c,K x,J i) {
 namespace nn=torch::nn;
 switch(c) {
  case Cast::bce:         return AnyModule(nn::BCELoss(             reduce<nn::BCELossOptions>(x,i,c)));
  case Cast::kl:          return AnyModule(nn::KLDivLoss(           reduce<nn::KLDivLossOptions>(x,i,c)));
  case Cast::l1:          return AnyModule(nn::L1Loss(              reduce<nn::L1LossOptions>(x,i,c)));
  case Cast::mse:         return AnyModule(nn::MSELoss(             reduce<nn::MSELossOptions>(x,i,c)));
  case Cast::multilabel:  return AnyModule(nn::MultiLabelMarginLoss(reduce<nn::MultiLabelMarginLossOptions>(x,i,c)));
  case Cast::smoothl1:    return AnyModule(nn::SmoothL1Loss(        reduce<nn::SmoothL1LossOptions>(x,i,c)));
  case Cast::softmargin:  return AnyModule(nn::SoftMarginLoss(      reduce<nn::SoftMarginLossOptions>(x,i,c)));

  case Cast::bcelogits:   return AnyModule(nn::BCEWithLogitsLoss(classwt(x,i,c,nn::BCEWithLogitsLossOptions())));
  case Cast::multisoft:   return AnyModule(nn::MultiLabelSoftMarginLoss(classwt(x,i,c,nn::MultiLabelSoftMarginLossOptions())));
  case Cast::ce:          return AnyModule(nn::CrossEntropyLoss(classwt<nn::CrossEntropyLossOptions>(x,i,c)));
  case Cast::nll:         return AnyModule(nn::NLLLoss(classwt<nn::NLLLossOptions>(x,i,c)));

  case Cast::hinge:       return AnyModule(nn::HingeEmbeddingLoss( margin<nn::HingeEmbeddingLossOptions>(x,i,c)));
  case Cast::cosineloss:  return AnyModule(nn::CosineEmbeddingLoss(margin<nn::CosineEmbeddingLossOptions>(x,i,c)));
  case Cast::margin:      return AnyModule(nn::MarginRankingLoss(  margin<nn::MarginRankingLossOptions>(x,i,c)));

/*
  case Cast::multimargin: {Scalar p,m;        multiargs(x,i,p,m,w,r);   a=std::make_shared<MultiMarginLoss>(p,m,w,r); break;}
  case Cast::triplet:     {bool s;double e,p; triargs(x,i,m,p,e,s,r);   a=std::make_shared<TripletMarginLoss>(m,p,e,s,r); break;}
  case Cast::poissonloss: {bool l,f;double e; poissonargs(x,i,l,f,e,r); a=std::make_shared<PoissonNLLLoss>(l,f,e,r); break;}
  case Cast::ctc:         {bool z;int64_t b;  ctc1(x,i,b,z,r);          a=std::make_shared<CTCLoss>(b,z,r); break;}
*/
  default: AT_ERROR("Unrecognized loss function: ",s);
 }
}

static K lossopt(bool a,Cast c,Loss *l) {
 K x=xD(ktn(KS,0),ktn(0,0));
 if (auto* p=dynamic_cast<BasicLoss*>(l)) {
  LossOptions d,o=p->options;
  if(a || d.reduce() != o.reduce()) OPTION(x, reduce, ks(rmap(o.reduce())));
 } else if(auto* p=dynamic_cast<WeightedLoss*>(l)) {
  WeightedLossOptions d,o=p->options;
  if(a || o.weight().defined())     OPTION(x, weight, kget(o.weight()));
  if(a || d.reduce() != o.reduce()) OPTION(x, reduce, ks(rmap(o.reduce())));
 } else if(auto* p=dynamic_cast<LogLoss*>(l)) {
  LogLossOptions d,o=p->options;
  if(a || o.weight().defined())     OPTION(x, weight, kget(o.weight()));
  if(a || d.ignore() != o.ignore()) OPTION(x, ignore, kj(o.ignore()));
  if(a || d.reduce() != o.reduce()) OPTION(x, reduce, ks(rmap(o.reduce())));
 } else if(auto* p=dynamic_cast<MarginLoss*>(l)) {
  MarginLossOptions d(marginval(c)),o=p->options;
  if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
  if(a || d.reduce() != o.reduce()) OPTION(x, reduce, ks(rmap(o.reduce())));
 } else if(auto* p=dynamic_cast<MultiMarginLoss*>(l)) {
  MultiMarginLossOptions d,o=p->options;
  if(a || !match(d.p(),o.p()))           OPTION(x, p,      kscalar(o.p()));
  if(a || !match(d.margin(),o.margin())) OPTION(x, margin, kscalar(o.margin()));
  if(a || o.weight().defined())          OPTION(x, weight, kget(o.weight()));
  if(a || d.reduce() != o.reduce())      OPTION(x, reduce, ks(rmap(o.reduce())));
 } else if(auto* p=dynamic_cast<TripletMarginLoss*>(l)) {
  TripletLossOptions d,o=p->options;
  if(a || d.margin() != o.margin()) OPTION(x, margin, kf(o.margin()));
  if(a || d.p()      != o.p())      OPTION(x, p,      kf(o.p()));
  if(a || d.eps()    != o.eps())    OPTION(x, eps,    kf(o.eps()));
  if(a || d.swap()   != o.swap())   OPTION(x, swap,   kb(o.swap()));
  if(a || d.reduce() != o.reduce()) OPTION(x, reduce, ks(rmap(o.reduce())));
 } else if(auto* p=dynamic_cast<PoissonNLLLoss*>(l)) {
  PoissonLossOptions d,o=p->options;
  if(a || d.log()    != o.log())    OPTION(x, log,    kb(o.log()));
  if(a || d.full()   != o.full())   OPTION(x, full,   kb(o.full()));
  if(a || d.eps()    != o.eps())    OPTION(x, eps,    kf(o.eps()));
  if(a || d.reduce() != o.reduce()) OPTION(x, reduce, ks(rmap(o.reduce())));
 } else if(auto* p=dynamic_cast<CTCLoss*>(l)) {
  CTCLossOptions d,o=p->options;
  if(a || d.blank()   != o.blank())   OPTION(x, blank,   kj(o.blank()));
  if(a || d.zeroinf() != o.zeroinf()) OPTION(x, zeroinf, kb(o.zeroinf()));
  if(a || d.reduce()  != o.reduce())  OPTION(x, reduce,  ks(rmap(o.reduce())));
 } else {
  AT_ERROR("Unrecognized loss pointer",(I)c);
 }
 return x;
}
 
static K lossopt2(bool a,Cast c,AnyModule& m) {
 namespace nn=torch::nn;
 K x=xD(ktn(KS,0),ktn(0,0));
 switch(c) {
  case Cast::bce:        reduce2(a, x, m.get<nn::BCELoss>()->options); break;
  case Cast::kl:         reduce2(a, x, m.get<nn::KLDivLoss>()->options); break;
  case Cast::l1:         reduce2(a, x, m.get<nn::L1Loss>()->options); break;
  case Cast::mse:        reduce2(a, x, m.get<nn::MSELoss>()->options); break;
  case Cast::multilabel: reduce2(a, x, m.get<nn::MultiLabelMarginLoss>()->options); break;
  case Cast::smoothl1:   reduce2(a, x, m.get<nn::SmoothL1Loss>()->options); break;
  case Cast::softmargin: reduce2(a, x, m.get<nn::SoftMarginLoss>()->options); break;

  case Cast::bcelogits:  classwt(a, x, m.get<nn::BCEWithLogitsLoss>()->options); break;
  case Cast::multisoft:  classwt(a, x, m.get<nn::MultiLabelSoftMarginLoss>()->options); break;
  case Cast::ce:         classwt(a, x, m.get<nn::CrossEntropyLoss>()->options); break;
  case Cast::nll:        classwt(a, x, m.get<nn::NLLLoss>()->options); break;

  case Cast::hinge:       margin(a, x, m.get<nn::HingeEmbeddingLoss>()->options); break;
  case Cast::cosineloss:  margin(a, x, m.get<nn::CosineEmbeddingLoss>()->options); break;
  case Cast::margin:      margin(a, x, m.get<nn::MarginRankingLoss>()->options); break;

  case Cast::multimargin: multi(a, x, m.get<nn::MultiMarginLoss>()->options); break;
  case Cast::triplet:     triplet(a, x, m.get<nn::TripletMarginLoss>()->options); break;
  case Cast::poissonloss: poisson(a, x, m.get<nn::PoissonNLLLoss>()->options); break;
  case Cast::ctc:         ctc(a, x, m.get<nn::CTCLoss>()->options); break;
  default: AT_ERROR("Unrecognized loss module"); break;
 }
 return x;
}

K lossdict(bool a,bool b,Cast c,Loss* l) {
 //a:true if all options, b:true if full state
 K k,v;
 if(b) {
  k=statekeys(); v=ktn(0,k->n);
  kK(v)[0]=kc('l');   //class="l" for loss
  kK(v)[2]=ks((S)""); //empty user-defined name
  kK(v)[4]=ktn(0,0);  //empty parms
  kK(v)[5]=ktn(0,0);  //and buffers
 } else {
  k=ktn(KS,2),v=ktn(0,2);
  kS(k)[0]=statekey(State::module);
  kS(k)[1]=statekey(State::options);
 }
 kK(v)[b ? 1 : 0]=ks(lmap(c));
 kK(v)[b ? 3 : 1]=lossopt(a,c,l);
 return xD(k,v);
}
 
// this version of lossdict() called from generic state() function in k-level api
K lossdict(Ktag *g,K x) {
 bool a=env().alloptions;
 if(x->n==1 || (x->n==2 && xbool(x,1,a)))
  return lossdict(a,true,g->c,((Kloss*)g)->l.get());
 else
  AT_ERROR("Loss state requires 1-2 args: previously allocated ptr or (ptr;options flag)");
}

K lossdict2(bool a,bool b,Cast c,AnyModule &m) {
 //a:true if all options, b:true if full state
 K k,v;
 if(b) {
  k=statekeys(); v=ktn(0,k->n);
  kK(v)[0]=kc('l');   //class="l" for loss
  kK(v)[2]=ks((S)""); //empty user-defined name
  kK(v)[4]=ktn(0,0);  //empty parms
  kK(v)[5]=ktn(0,0);  //and buffers
 } else {
  k=ktn(KS,2),v=ktn(0,2);
  kS(k)[0]=statekey(State::module);
  kS(k)[1]=statekey(State::options);
 }
 kK(v)[b ? 1 : 0]=ks(lmap(c));
 kK(v)[b ? 3 : 1]=lossopt2(a,c,m);
 return xD(k,v);
}

static K lossfwd(Cast c,Loss *l,K a) {
 bool p; Tensor r,x,y,z;
 if(a->n==3) {
  p=xtenarg(a,1,x,y);
  r=l->forward(x,y);
 } else if(a->n==4) {
  p=xtenarg(a,1,x,y,z);
  r=l->forward(x,y,z);
 } else if(c==Cast::ctc && a->n==5) {
  IntArrayRef jx,jy; Tensor nx,ny; p=ctc2(a,1,x,y,nx,ny,jx,jy);
  if(nx.defined())
   r=nx.defined() ? l->forward(x,y,nx,ny) : l->forward(x,y,jx,jy);
 }
 if(r.defined())
  return p ? kten(r) : kget(r);
 else
  AT_ERROR("Unrecognized arg(s) for ",lmap(c)," forward call");
}

K lossto(Kloss* l,const TensorOptions& o,bool a) {
 auto s=torch::typeMetaToScalarType(o.dtype());
 if(o.has_device() && o.has_dtype()) l->l->to(o.device(),s,a);
 else if(o.has_device())             l->l->to(o.device(),a);
 else                                l->l->to(s,a);
 return (K)0;
}

Tensor losswt(Loss *l) {
 if(auto* p=dynamic_cast<WeightedLoss*>(l))
  return p->options.weight();
 else if(auto* p=dynamic_cast<LogLoss*>(l))
  return p->options.weight();
 else if(auto* p=dynamic_cast<MultiMarginLoss*>(l))
  return p->options.weight();
 else
 return {};
}

KAPI loss(K x) {
 KTRY
  S s; bool a=env().alloptions; Kloss *l;
  if(xsyms(x,s) || xsym(x,0,s)) {
   return lossinit(s,x,1); //define loss from sym or (sym;option(s)..)
  } else if(xdict(x)) {    //define loss from state dictionary
   return lossinit(statesym(State::module,x),statedict(State::options,x),-1);
  } else if(((l=xloss(x))) || (xbool(x,1,a) && x->n==2 && ((l=xloss(x,0))))) {
   return lossdict(a,false,l->c,l->get()); //given allocated loss ptr or ptr w'boolean, return options
  } else if((l=xloss(x,0)) && x->n>1) {
   return lossfwd(l->c,l->get(),x); //else, run forward calculation w'loss and input,target,..
  } else {
   AT_ERROR("Unrecognized arg(s)");
  }
 KCATCH("Loss module");
}

KAPI loss2(K x) {
 KTRY
  S s; bool a=env().alloptions; Kmodule *l;
  if(xsyms(x,s) || xsym(x,0,s)) {
   Cast c=lmap(s);
   return kloss(c, lossinit2(s,c,x,1));
/*
  } else if(xdict(x)) {    //define loss from state dictionary
   return lossinit(statesym(State::module,x),statedict(State::options,x),-1);
*/
  } else if(((l=xLoss(x))) || (xbool(x,1,a) && x->n==2 && ((l=xLoss(x,0))))) {
   return lossdict2(a,false,l->c,l->m); //given allocated loss ptr or ptr w'boolean, return options
/*
  } else if((l=xloss(x,0)) && x->n>1) {
   return lossfwd(l->c,l->get(),x); //else, run forward calculation w'loss and input,target,..
*/
  } else {
   AT_ERROR("Unrecognized arg(s)");
  }
 KCATCH("Loss module");
}

K lossattr(const Lossptr& l,Ktype k,Attr a) {
 switch(a) {
  case Attr::ref:     return kj(l.use_count());
  default: AT_ERROR(mapattr(a),": not implemented for loss modules");
 }
}

// ----------------------------------
// loss fns defined in k namespace
// ----------------------------------
void lossfn(K x) {
 fn(x, "loss",        KFN(loss),1);
 fn(x, "bce",         KFN(bce),1);
 fn(x, "bcelogits",   KFN(bcelogits),1);
 fn(x, "bcelogitw",   KFN(bcelogitw),1);
 fn(x, "ce",          KFN(ce),1);
 fn(x, "cosineloss",  KFN(cosineloss),1);
 fn(x, "ctc",         KFN(Ctc),1);
 fn(x, "hinge",       KFN(hinge),1);
 fn(x, "kl",          KFN(kl),1);
 fn(x, "l1",          KFN(l1),1);
 fn(x, "margin",      KFN(Margin),1);
 fn(x, "mse",         KFN(mse),1);
 fn(x, "multilabel",  KFN(multilabel),1);
 fn(x, "multimargin", KFN(multimargin),1);
 fn(x, "multisoft",   KFN(multisoft),1);
 fn(x, "nll",         KFN(nll),1);
 fn(x, "poissonloss", KFN(poissonloss),1);
 fn(x, "smoothl1",    KFN(smoothl1),1);
 fn(x, "softmargin",  KFN(softmargin),1);
 fn(x, "triplet",     KFN(Triplet),1);
}
