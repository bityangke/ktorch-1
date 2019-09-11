#include "ktorch.h"
#include "kloss.h"

// append a loss option to a k dictionary given dict,name & value
#define OPTION(x,k,v) dictadd(x, lset(Setting::k), v)

using Lf  = Tensor (*)(const Tensor&, const Tensor&, int64_t);                          // loss w'reduction only
using Lw  = Tensor (*)(const Tensor&, const Tensor&, const Tensor&, int64_t);           // loss w'wts
using Lwi = Tensor (*)(const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t);  // loss w'wts & ignore ind

// ------------------------------------------------------------------------------------------------------
// lmap - map to/from sym to loss function name, e.g. `mse <-> Cast::mse
// lset - map to/from sym to loss setting enum, e.g. `reduce <-> Setting::reduce
// rmsg,rmap - message and mapping for loss reduction to/from sym and enumeration
// xreduce - check if sym, if matches loss reduction, set int, e.g. `none -> 0, `mean -> 1, `sum -> 2
// ------------------------------------------------------------------------------------------------------
Z Cast lmap(S s) {
 for(auto&m:env().loss)
  if(std::get<0>(m)==s) return std::get<1>(m);
 AT_ERROR("Unrecognized loss function: ",s);
}

ZS lmap(Cast c) {
 for(auto&m:env().loss)
  if(std::get<1>(m)==c) return std::get<0>(m);
 AT_ERROR("Unrecognized loss function: ",(I)c);
}

Z S lset(Setting s) {
 for(auto&m:env().lset)
  if(std::get<1>(m)==s) return std::get<0>(m);
 AT_ERROR("Unrecognized loss setting: ",(I)s);
}

Z Setting lset(S s) {
 for(auto&m:env().lset)
  if(std::get<0>(m)==s) return std::get<1>(m);
 AT_ERROR("Unrecognized loss setting: ",s);
}

Z std::string rmsg(B b) {
 std::string s;
 for(auto&m:env().reduce)
  s += (b ? std::get<0>(m) : std::to_string(std::get<1>(m))) + ",";
 s.pop_back();
 return s;
}

Z int64_t rmap(S s) {
 for(auto&m:env().reduce)
  if(std::get<0>(m)==s) return std::get<1>(m);
 AT_ERROR("Unrecognized setting for loss reduction: ",s,", expecting one of ",rmsg(true));
}

Z S rmap(int64_t r) {
 for(auto&m:env().reduce)
  if(std::get<1>(m)==r) return std::get<0>(m);
 AT_ERROR("Unrecognized setting for loss reduction: ",r,", expecting one of ",rmsg(false));
}

Z B xreduce(K x,int64_t &r) {
 if(x->t == -KS) return r=rmap(x->s), true;
 return false;
}

Z B xreduce(K x,J i,int64_t &r) {
 if(x->t == KS && -1<x->n && x->n>i)
  return r=rmap(kS(x)[i]),true;
 else
  return xind(x,i) && xreduce(kK(x)[i],r);
}

// ------------------------------------------------------------------------------------------------------
// reduce - return default reduction mode, or process given arg(s) & offset to return reduction mode
// kloss - call loss function with x,y tensors/arrays and optional reduction mode
// bce - binary cross entropy has option of batch weights, so function parses (x;y) or (x;y;wt)
// ------------------------------------------------------------------------------------------------------
int64_t reduce() {return Reduction::Mean;}

int64_t reduce(cS s,K x,J i) { // check argument(s) for sym or named pair/dict, e.g. (`reduce;`mean))
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

ZK kloss(K a,Lf f,cS s) {
 KTRY
  Tensor x,y;
  if(!a->t && (a->n==2 || a->n==3)) {
   auto r=reduce(s,a,2);
   return xtenarg(a,x,y) ? kten(f(x,y,r)) : kget(f(x,y,r));
  } else if(0 < a->t && a->t<20 && a->n==2) {
   return x=kput(a), kget(f(x[0],x[1],reduce()));
  } else {
   AT_ERROR(s," loss expects (input;target) or (input;target;reduction), e.g. (x;y;`mean)");
  }
 KCATCH(s)
}

KAPI kl(K x)          {return kloss(x, torch::kl_div,                 "KL divergence");}
KAPI l1(K x)          {return kloss(x, torch::l1_loss,                "l1");}
KAPI mse(K x)         {return kloss(x, torch::mse_loss,               "mse");}
KAPI multilabel(K x)  {return kloss(x, torch::multilabel_margin_loss, "multi-label margin");}
KAPI smoothl1(K x)    {return kloss(x, torch::smooth_l1_loss,         "smooth l1");}
KAPI softmargin(K x)  {return kloss(x, torch::soft_margin_loss,       "soft margin");}

Z B bcearg(K x) {return x->t==-KS || x->t==KS || xempty(x) || xdict(x);}  // true if arg is a setting (rather than wt tensor)

KAPI bce(K a) {
 KTRY
  B p=false; Tensor l,x,y,w;
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
// wtargs - process args from k for weight tensor, index to ignore & reduction method (or some subset)
// wtloss - functional form for nll,cross entropy, multi-label soft margin (no c++ version yet)
// ------------------------------------------------------------------------------------------------------
ZV wtargs(Cast c,cS s,K x,J i,Tensor& w,J& j,int64_t &r) {
 B b=c==Cast::ce || c==Cast::nll; Pairs p; J n=xargc(x,i,p); j=-100; r=reduce();
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

ZK wtloss(K a,Cast c,cS s) {
 KTRY
  B p=false; J j; int64_t r; Tensor l,x,y,w;
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
ZK bceloss(K a,B b,cS s) {  //a:args, b:true if batch wts
 KTRY
  B p=false; J i=2+b,j; int64_t r; Tensor l,x,y,bw,w;
  if(a->t) {
   AT_ERROR(s," loss not implemented for ",kname(a->t));
  } else if(a->n < i) {
   AT_ERROR(s,(b ? " loss expects at least 3 args, (input;target;batch weights)"
                 : " loss expects at least 2 args, (input;target)"));
  }
  wtargs(Cast::bcelogits,s,a,i,w,j,r);
  p=xtenarg(a,x,y);
  if(b && !xten(a,2,bw)) bw=kput(a,2);
  l=torch::binary_cross_entropy_with_logits(x,y,bw,w,r);
  return p ? kten(l) : kget(l);
 KCATCH(s)
}

KAPI bcelogits(K x) {return bceloss(x, false, "binary cross-entropy with logits");}
KAPI bcelogitw(K x) {return bceloss(x, true,  "binary cross-entropy with logits & batch weights");}

// ------------------------------------------------------------------------------------------------------
// marginval  - default margin value given loss type
// marginargs - process optional margin & reduction arguments in given k array
// marginloss - funcional form of loss functions w'margin & reduction args
// ------------------------------------------------------------------------------------------------------
ZF marginval(Cast c) {return c==Cast::hinge ? 1.0 : 0.0;}
ZV marginargs(Cast c,cS s,K x,J i,double &m,int64_t &r) {
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

ZK marginloss(K a,Cast c,cS s) {
 KTRY
  B p=false,h=c==Cast::hinge; F m; int64_t r; Tensor l,x1,x2,y;
  if(a->t || a->n<3-h || a->n>5-h)
   AT_ERROR(s,h ? " loss expects (input;target), (input;target;margin) or (input;target;margin;reduction)"
                : " loss expects (input1;input2;target), (input1;input2;target;margin) or (input1;input2;target;margin;reduction)");
  marginargs(c,s,a,3-h,m,r);
  p=h ? xtenarg(a,x1,y) : xtenarg(a,x1,x2,y);
  switch(c) {
   case Cast::hinge:      l=torch::hinge_embedding_loss(x1,y,m,r); break;
   case Cast::cosineloss: l=torch::cosine_embedding_loss(x1,x2,y,m,r); break;
   case Cast::margin:     l=torch::margin_ranking_loss(x1,x2,y,m,r); break;
   default: AT_ERROR("Unrecognized loss function: ",(I)c); break;
  }
  return p ? kten(l) : kget(l);
 KCATCH(s)
}

KAPI hinge(K x)      {return marginloss(x, Cast::hinge,      "hinge embedding");}
KAPI cosineloss(K x) {return marginloss(x, Cast::cosineloss, "cosine embedding");}
KAPI margin(K x)     {return marginloss(x, Cast::margin,     "margin ranking");}

// ------------------------------------------------------------------------------------------------------
// multiargs - process optional power,margin,weight & reduction arguments in given k array
// multimargin - funcional form of multi margin loss function
// ------------------------------------------------------------------------------------------------------
ZV multiargs(K x,J i,Scalar &pw,Scalar& m,Tensor& w,int64_t &r) {
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

KAPI multimargin(K a) {
 KTRY
  B b; Tensor l,x,y,w; Scalar p,m; int64_t r; 
  if(a->t || a->n<2 || a->n>6)
   AT_ERROR("multi-margin loss expects 2-6 args: at minimum (input;target) up to (input;target;p;margin;weight;reduction)");
  multiargs(a,2,p,m,w,r); b=xtenarg(a,x,y);
  l=torch::multi_margin_loss(x,y,p,m,w,r);
  return b ? kten(l) : kget(l);
 KCATCH("multi-margin");
}

// ------------------------------------------------------------------------------------------------------
// triargs - process optional margin,p,eps,swap flag & reduction args in k array for triplet loss
// triplet  - funcional form of triplet margin loss function
// ------------------------------------------------------------------------------------------------------
ZV triargs(K x,J i,double& m,double& pw,double& e,bool& s,int64_t& r) {
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

KAPI triplet(K a) {
 KTRY
  B b,s; F e,m,p; Tensor l,x,y,z; int64_t r; 
  if(a->t) {
   AT_ERROR("triplet margin loss not implemented for ",kname(a->t));
  } else if(a->n < 3) {
   AT_ERROR("triplet margin loss expects at least 3 args, (anchor;positive;negative)");
  }
  triargs(a,3,m,p,e,s,r); b=xtenarg(a,x,y,z);
  l=torch::triplet_margin_loss(x,y,z,m,p,e,s,r);
  return b ? kten(l) : kget(l);
 KCATCH("triplet margin");
}

// ------------------------------------------------------------------------------------------------------
// poissonargs - process optional margin,p,eps,swap flag & reduction args in k array for triplet loss
// poissonloss  - funcional form of poisson nll loss function
// ------------------------------------------------------------------------------------------------------
ZV poissonargs(K x,J i,bool& l,bool& f,double& e,int64_t& r) {
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

KAPI poissonloss(K a) {
 KTRY
  B p,ln,f; F e; Tensor l,x,y; int64_t r; 
  if(a->t) {
   AT_ERROR("poisson nll loss not implemented for ",kname(a->t));
  } else if(a->n < 2) {
   AT_ERROR("poisson nll loss expects at least 2 args, (input;target)");
  }
  poissonargs(a,2,ln,f,e,r); p=xtenarg(a,x,y);
  l=torch::poisson_nll_loss(x,y,ln,f,e,r);
  return p ? kten(l) : kget(l);
 KCATCH("poisson nll loss");
}

// -------------------------------------------------------------------------------------------------------------------
// ctc1 - process args for CTC loss, blank value, flag for setting infinities -> zero & reduction method
// ctc2 - check inputs for tensor or arrays for input/target lengths (different forward call)
// ctc - funcional form of connectionist temporal classification loss between continuous time series & target sequence
// -------------------------------------------------------------------------------------------------------------------
ZV ctc1(K x,J i,int64_t& b,bool& z,int64_t& r) {
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

Z B ctc2(K a,J i,Tensor& x,Tensor& y,Tensor& nx,Tensor& ny,JRef& jx,JRef& jy) {
  B p=xtenarg(a,i,x,y);
  if(!(xsize(a,i+2,jx) && xsize(a,i+3,jy))) { // unless both lenghts given as k arrays
    if(!xten(a,i+2,nx)) nx=kput(a,2);  // define input lengths as tensor
    if(!xten(a,i+3,ny)) ny=kput(a,3);  // define target lengths as tensor
  }
  return p;
}

KAPI ctc(K a) {
 KTRY
  B p,z; JRef jx,jy; Tensor l,x,y,nx,ny; int64_t b,r; 
  if(a->t) {
   AT_ERROR("CTC loss not implemented for ",kname(a->t));
  } else if(a->n < 4) {
   AT_ERROR("CTC loss expects at least 4 args, (input;target;input lengths;target lengths)");
  }
  ctc1(a,4,b,z,r); 
  p=ctc2(a,0,x,y,nx,ny,jx,jy);
  l=nx.defined() ? torch::ctc_loss(x,y,nx,ny,b,r,z) : torch::ctc_loss(x,y,jx,jy,b,r,z);
  return p ? kten(l) : kget(l);
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
ZK lossinit(S s,K x,J i) {
 J j; F m; Cast c=lmap(s); Tensor w; int64_t r; Lossptr a;
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

  case Cast::multimargin: {Scalar p,m; multiargs(x,i,p,m,w,r);  a=std::make_shared<MultiMarginLoss>(p,m,w,r); break;}
  case Cast::triplet:     {B s;F e,p; triargs(x,i,m,p,e,s,r);   a=std::make_shared<TripletMarginLoss>(m,p,e,s,r); break;}
  case Cast::poissonloss: {B l,f;F e; poissonargs(x,i,l,f,e,r); a=std::make_shared<PoissonNLLLoss>(l,f,e,r); break;}
  case Cast::ctc:         {B z;int64_t b; ctc1(x,i,b,z,r);      a=std::make_shared<CTCLoss>(b,z,r); break;}
  default: AT_ERROR("Unrecognized loss function: ",s); break;
 }
 return kptr(new Kloss(c,a));
}

ZK lossopt(B a,Cast c,Loss *l) {
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
 
K lossdict(B a,B b,Cast c,Loss* l) {
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
 B a=env().alloptions;
 if(x->n==1 || (x->n==2 && xbool(x,1,a)))
  return lossdict(a,true,g->c,((Kloss*)g)->l.get());
 else
  AT_ERROR("Loss state requires 1-2 args: previously allocated ptr or (ptr;options flag)");
}

ZK lossfwd(Cast c,Loss *l,K a) {
 B p; Tensor r,x,y,z;
 if(a->n==3) {
  p=xtenarg(a,1,x,y);
  r=l->forward(x,y);
 } else if(a->n==4) {
  p=xtenarg(a,1,x,y,z);
  r=l->forward(x,y,z);
 } else if(c==Cast::ctc && a->n==5) {
  JRef jx,jy; Tensor nx,ny; p=ctc2(a,1,x,y,nx,ny,jx,jy);
  if(nx.defined())
   r=nx.defined() ? l->forward(x,y,nx,ny) : l->forward(x,y,jx,jy);
 }
 if(r.defined())
  return p ? kten(r) : kget(r);
 else
  AT_ERROR("Unrecognized arg(s) for ",lmap(c)," forward call");
}

K lossto(Kloss* l,const TensorOptions& o,B a) {
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
  S s; B a=env().alloptions; Kloss *l;
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

// ----------------------------------
// loss fns defined in k namespace
// ----------------------------------
V lossfn(K x) {
 fn(x, "loss",        KFN(loss),1);
 fn(x, "bce",         KFN(bce),1);
 fn(x, "bcelogits",   KFN(bcelogits),1);
 fn(x, "bcelogitw",   KFN(bcelogitw),1);
 fn(x, "ce",          KFN(ce),1);
 fn(x, "cosineloss",  KFN(cosineloss),1);
 fn(x, "ctc",         KFN(ctc),1);
 fn(x, "hinge",       KFN(hinge),1);
 fn(x, "kl",          KFN(kl),1);
 fn(x, "l1",          KFN(l1),1);
 fn(x, "margin",      KFN(margin),1);
 fn(x, "mse",         KFN(mse),1);
 fn(x, "multilabel",  KFN(multilabel),1);
 fn(x, "multimargin", KFN(multimargin),1);
 fn(x, "multisoft",   KFN(multisoft),1);
 fn(x, "nll",         KFN(nll),1);
 fn(x, "poissonloss", KFN(poissonloss),1);
 fn(x, "smoothl1",    KFN(smoothl1),1);
 fn(x, "softmargin",  KFN(softmargin),1);
 fn(x, "triplet",     KFN(triplet),1);
}
