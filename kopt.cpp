#include "ktorch.h"

#define OPTBUFFER(x,o,k) dictadd(x, #k, kget(o->k))
#define OPTSET(x,k,v) dictadd(x, oset(Setting::k), v)

using Adagrad        = torch::optim::Adagrad;
using AdagradOptions = torch::optim::AdagradOptions;
using Adam           = torch::optim::Adam;
using AdamOptions    = torch::optim::AdamOptions;
using LBFGS          = torch::optim::LBFGS;
using LBFGSOptions   = torch::optim::LBFGSOptions;
using RMSprop        = torch::optim::RMSprop;
using RMSpropOptions = torch::optim::RMSpropOptions;
using SGD            = torch::optim::SGD;
using SGDOptions     = torch::optim::SGDOptions;

// --------------------------------------------------------------------------------------
// kopt - given optimizer type & shared pointer to newly created optimizer, return k ptr
// omap - map to/from optimizer symbol/enumeration and default learning rate
// oset - optimizer settings, map sym <-> enum
// osize - size of buffers required (not counting trailing parameters without gradients)
// --------------------------------------------------------------------------------------
K kopt(Cast x,const std::shared_ptr<OptimizerBase>& y) {return kptr(new Kopt(x,y));}

static void omap(S s,Cast &c,double &r) {
 for(auto& m:env().opt)
   if(s==std::get<0>(m)) {c=std::get<1>(m); r=std::get<2>(m); return;}
 AT_ERROR("Unrecognized optimizer: ",s);
}

static void omap(Cast c,S &s,double &r) {
 for(auto& m:env().opt)
   if(c==std::get<1>(m)) {s=std::get<0>(m); r=std::get<2>(m); return;}
 AT_ERROR("Unrecognized optimizer: ",(I)c);
}

static Setting oset(S s) {
 for(auto& m:env().oset)
  if(s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("Unrecognized optimizer setting: ",s);
}

static S oset(Setting e) {
 for(auto& m:env().oset) if(e==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized optimizer setting: ",(I)e);
}

static size_t osize(const TensorVector& p) {
 size_t i,n=0;
 for(i=0; i<p.size(); ++i) 
  if(p.at(i).grad().defined()) n=i+1;
 return n;
}

// --------------------------------------------------------------------------------------
// bget - find buffer (vector of longs/tensors) from dictionary given name
// bset - set optimization buffers from k dictionary
//      - vector of longs (e.g. step_buffers, one per parameter)
//      - vector of tensors
//      - deque of tensors (LBFGS)
//      - single tensor (LBFGS)
// --------------------------------------------------------------------------------------
static K bget(K x,cS s) {
 auto i=xdict(x) ? kfind(kK(x)[0],s) : -1;
 return (i<0 || kK(x)[1]->t) ? nullptr : kK(kK(x)[1])[i];
}

static void bset(size_t n,cS s,const TensorVector& p,LongVector& v,const K x) {
 // restore vector of longs (parameter vector not referenced, included for uniform call)
 K y=bget(x,s);
 if(!y || !y->n) return;
 if(y->t != KJ) AT_ERROR("type error: ",s,", expecting long list, input is ",kname(y->t));
 if(n != (unsigned)y->n) AT_ERROR("length error: ",s,", expecting ",n," longs, input has ",y->n);
 v.resize(n);
 for(size_t i=0; i<n; ++i) v.emplace_back(kJ(y)[i]);
}

static void bset(size_t n,cS s,const TensorVector& p,TensorVector& v,const K x) {
 K y=bget(x,s);
 if(!y || !y->n) return;
 if(y->t) AT_ERROR("type error: ",s,", expecting general list, input is ",kname(y->t));
 if(n != (unsigned)y->n) AT_ERROR("length error: ",s,", expecting ",n," arrays, input has ",y->n);
 v.resize(n);
 for(size_t i=0; i<n; ++i) {
  auto a=kput(kK(y)[i]);
  auto b=torch::zeros_like(p.at(i));
  if(a.dtype() != b.dtype())
   AT_ERROR("Type mismatch: ",s,"[",i,"] is ",b.dtype(),", input is ",a.dtype());
  if(!b.is_same_size(a))
   AT_ERROR("Size mismatch: ",s,"[",i,"] is ",b.sizes(),", input is ",a.sizes());
  if(a.device() != b.device())
   b.set_data(a);
  else
   b.set_(a);
  v[i]=std::move(b);
 }
}

static void bset(size_t n,cS s,const TensorVector& p,TensorDeque& v,const K x) {
 // used w'LBFGS, not sure if parameter count/type/device relevant
 K y=bget(x,s);
 if(!y || !y->n) return;
 if(y->t) AT_ERROR("type error: ",s,", expecting general list, input is ",kname(y->t));
 v.resize(n);
 for(size_t i=0; i<n; ++i)
  v[i]=kput(kK(y)[i]);
}

static void bset(size_t n,cS s,const TensorVector& p,Tensor& t,const K x) {
 // used w'LBFGS, not sure if parameter count/type/device relevant
 K y=bget(x,s);
 if(!y || !y->n) return;
 t=kput(y);
}

// ----------------------------------------------------------------------------------------
// adagrad - parse args (lr;lrdecay;wtdecay) or (..;name/val pairs/dict)
//         - if given options,buffers, allocate new optimizer and return ptr
//         - if given previously allocated ptr, return dictionary of options & buffers
// ----------------------------------------------------------------------------------------
static void adagrad(K x,J i,AdagradOptions& o) {
 Pairs p; J n=xargc(x,i,p); F f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.learning_rate(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.lr_decay(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n) AT_ERROR("Unrecognized arg(s) for Adagrad optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:      f=pdouble(p); if(f==f) o.learning_rate(f); break;
   case Setting::lrdecay: f=pdouble(p); if(f==f) o.lr_decay(f); break;
   case Setting::decay:   f=pdouble(p); if(f==f) o.weight_decay(f); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for Adagrad optimization"); break;
  }
}

static Optptr adagrad(const TensorVector& w,const AdagradOptions& a,K y) {
 auto o=std::make_shared<Adagrad>(w,a);
 auto n=osize(o->parameters());
 if(y && n) {
  bset(n, "step_buffers", o->parameters(), o->step_buffers, y);
  bset(n, "sum_buffers",  o->parameters(), o->sum_buffers,  y);
 }
 return o;
}

static K adagrad(B a,F r,Adagrad* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); AdagradOptions d(r),o=v->options;
 if(a || d.learning_rate() != o.learning_rate()) OPTSET(x, lr,      kf(o.learning_rate()));
 if(a || d.lr_decay()      != o.lr_decay())      OPTSET(x, lrdecay, kf(o.lr_decay()));
 if(a || d.weight_decay()  != o.weight_decay())  OPTSET(x, decay,   kf(o.weight_decay()));
 return x;
}

static K adagrad(Adagrad* v) {  //return internal buffer state as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0));
 OPTBUFFER(x,v,step_buffers);
 OPTBUFFER(x,v,sum_buffers);
 return x;
}

// ----------------------------------------------------------------------------------------
// adam - parse args (lr;beta1;beta2;eps;wtdecay;amsgrad) or (..;name-value pairs/dict)
// ----------------------------------------------------------------------------------------
static void adam(K x,J i,AdamOptions& o) {
 Pairs p; J n=xargc(x,i,p); B b; F f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.learning_rate(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.beta1(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.beta2(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.eps(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xbool(x,i,b)){i++; n--; o.amsgrad(b);}
 if(n) AT_ERROR("Unrecognized arg(s) for Adam optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:      f=pdouble(p); if(f==f) o.learning_rate(f); break;
   case Setting::beta1:   f=pdouble(p); if(f==f) o.beta1(f); break;
   case Setting::beta2:   f=pdouble(p); if(f==f) o.beta2(f); break;
   case Setting::eps:     f=pdouble(p); if(f==f) o.eps(f); break;
   case Setting::decay:   f=pdouble(p); if(f==f) o.weight_decay(f); break;
   case Setting::amsgrad: o.amsgrad(pbool(p)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for Adagrad optimization"); break;
  }
}

static Optptr adam(const TensorVector& w,const AdamOptions& a,K y) {
 auto o=std::make_shared<Adam>(w,a);
 auto n=osize(o->parameters());
 if(y && n) {
  bset(n, "step_buffers",                o->parameters(), o->step_buffers,               y);
  bset(n, "exp_average_buffers",         o->parameters(), o->exp_average_buffers,        y);
  bset(n, "exp_average_sq_buffers",      o->parameters(), o->exp_average_sq_buffers,     y);
  bset(n, "max_exp_average_sq_buffers",  o->parameters(), o->max_exp_average_sq_buffers, y);
 }
 return o;
}

static K adam(B a,F r,Adam* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); AdamOptions d(r),o=v->options;
 if(a || d.learning_rate() != o.learning_rate()) OPTSET(x, lr,      kf(o.learning_rate()));
 if(a || d.beta1()         != o.beta1())         OPTSET(x, beta1,   kf(o.beta1()));
 if(a || d.beta2()         != o.beta2())         OPTSET(x, beta2,   kf(o.beta2()));
 if(a || d.weight_decay()  != o.weight_decay())  OPTSET(x, decay,   kf(o.weight_decay()));
 if(a || d.eps()           != o.eps())           OPTSET(x, eps,     kf(o.eps()));
 if(a || d.amsgrad()       != o.amsgrad())       OPTSET(x, amsgrad, kb(o.amsgrad()));
 return x;
}

static K adam(Adam* v) {  //return internal buffer state as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0));
 OPTBUFFER(x,v,step_buffers);
 OPTBUFFER(x,v,exp_average_buffers);
 OPTBUFFER(x,v,exp_average_sq_buffers);
 OPTBUFFER(x,v,max_exp_average_sq_buffers);
 return x;
}

// ---------------------------------------------------------------------------------------
// lbfgs - (lr;max iter;max eval;tolerance grad;tolerance change;history size)
// ---------------------------------------------------------------------------------------
static void lbfgs(K x,J i,LBFGSOptions& o) {
 Pairs p; J j,n=xargc(x,i,p); F f;
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.learning_rate(f);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.max_iter(j);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.max_eval(j);}
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.tolerance_grad(f);}
 if(n && xnum(x,i,f))  {i++; n--; if(f==f)  o.tolerance_change(f);}
 if(n && xlong(x,i,j)) {i++; n--; if(j!=nj) o.history_size(j);}
 if(n) AT_ERROR("Unrecognized arg(s) for LBFGS optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        f=pdouble(p); if(f==f)  o.learning_rate(f); break;
   case Setting::iter:      j=plong(p);   if(j!=nj) o.max_iter(j); break;
   case Setting::eval:      j=plong(p);   if(j!=nj) o.max_eval(j); break;
   case Setting::gradtol:   f=pdouble(p); if(f==f)  o.tolerance_grad(f); break;
   case Setting::changetol: f=pdouble(p); if(f==f)  o.tolerance_change(f); break;
   case Setting::history:   j=plong(p);   if(j!=nj) o.history_size(j); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for LBFGS optimization"); break;
  }
}

static Optptr lbfgs(const TensorVector& w,const LBFGSOptions& a,K y) {
 auto o=std::make_shared<LBFGS>(w,a); auto n=osize(o->parameters());
 if(y) {
  bset(n, "d",              o->parameters(), o->d, y);
  bset(n, "t",              o->parameters(), o->t, y);
  bset(n, "H_diag",         o->parameters(), o->H_diag, y);
  bset(n, "prev_flat_grad", o->parameters(), o->prev_flat_grad, y);
  bset(n, "prev_loss",      o->parameters(), o->prev_loss, y);
  bset(n, "old_dirs",       o->parameters(), o->old_dirs, y);
  bset(n, "old_stps",       o->parameters(), o->old_stps, y);
 }
 return o;
}

static K lbfgs(B a,F r,LBFGS* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); LBFGSOptions d(r),o=v->options;
 if(a || d.learning_rate()    != o.learning_rate())    OPTSET(x, lr,        kf(o.learning_rate()));
 if(a || d.max_iter()         != o.max_iter())         OPTSET(x, iter,      kj(o.max_iter()));
 if(a || d.max_eval()         != o.max_eval())         OPTSET(x, eval,      kj(o.max_eval()));
 if(a || d.tolerance_grad()   != o.tolerance_grad())   OPTSET(x, gradtol,   kf(o.tolerance_grad()));
 if(a || d.tolerance_change() != o.tolerance_change()) OPTSET(x, changetol, kf(o.tolerance_change()));
 if(a || d.history_size()     != o.history_size())     OPTSET(x, history,   kj(o.history_size()));
 return x;
}

static K lbfgs(LBFGS* v) {  //return internal buffer state as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0));
 OPTBUFFER(x,v,d);
 OPTBUFFER(x,v,t);
 OPTBUFFER(x,v,H_diag);
 OPTBUFFER(x,v,prev_flat_grad);
 OPTBUFFER(x,v,prev_loss);
 OPTBUFFER(x,v,old_dirs);
 OPTBUFFER(x,v,old_stps);
 return x;
}

// ----------------------------------------------------------------------------------------
// rmsprop - parse arg(s) (lr;alpha;eps;decay;momentum;centered) or (..;nm-val pairs/dict)
// ----------------------------------------------------------------------------------------
static void rmsprop(K x,J i,RMSpropOptions& o) {
 Pairs p; J n=xargc(x,i,p); B b; F f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.learning_rate(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.alpha(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.eps(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.momentum(f);}
 if(n && xbool(x,i,b)){i++; n--; o.centered(b);}
 if(n) AT_ERROR("Unrecognized arg(s) for RMSprop optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        f=pdouble(p); if(f==f) o.learning_rate(f); break;
   case Setting::alpha:     f=pdouble(p); if(f==f) o.alpha(f); break;
   case Setting::eps:       f=pdouble(p); if(f==f) o.eps(f); break;
   case Setting::decay:     f=pdouble(p); if(f==f) o.weight_decay(f); break;
   case Setting::momentum:  f=pdouble(p); if(f==f) o.momentum(f); break;
   case Setting::centered:  o.centered(pbool(p)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for RMSprop optimization"); break;
  }
}

static Optptr rmsprop(const TensorVector& w,const RMSpropOptions& a,K y) {
 auto o=std::make_shared<RMSprop>(w,a);
 auto n=osize(o->parameters());
 if(y && n) {
  bset(n, "square_average_buffers", o->parameters(), o->square_average_buffers, y);
  bset(n, "momentum_buffers",       o->parameters(), o->momentum_buffers,       y);
  bset(n, "grad_average_buffers",   o->parameters(), o->grad_average_buffers,   y);
 }
 return o;
}

static K rmsprop(B a,F r,RMSprop* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); RMSpropOptions d(r),o=v->options;
 if(a || d.learning_rate() != o.learning_rate()) OPTSET(x, lr,       kf(o.learning_rate()));
 if(a || d.alpha()         != o.alpha())         OPTSET(x, alpha,    kf(o.alpha()));
 if(a || d.eps()           != o.eps())           OPTSET(x, eps,      kf(o.eps()));
 if(a || d.weight_decay()  != o.weight_decay())  OPTSET(x, decay,    kf(o.weight_decay()));
 if(a || d.momentum()      != o.momentum())      OPTSET(x, momentum, kf(o.momentum()));
 if(a || d.centered()      != o.centered())      OPTSET(x, centered, kb(o.centered()));
 return x;
}

static K rmsprop(RMSprop* v) {  //return internal buffer state as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0));
 OPTBUFFER(x,v,square_average_buffers);
 OPTBUFFER(x,v,momentum_buffers);
 OPTBUFFER(x,v,grad_average_buffers);
 return x;
}

// ----------------------------------------------------------------------------------------
// SGD parse args (lr;momentum;dampening;wtdecay;nesterov) or (..;name-value pairs/dict)
// ----------------------------------------------------------------------------------------
static void sgd(K x,J i,SGDOptions& o) {
 Pairs p; J n=xargc(x,i,p); B b; F f;
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.learning_rate(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.momentum(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.dampening(f);}
 if(n && xnum(x,i,f)) {i++; n--; if(f==f) o.weight_decay(f);}
 if(n && xbool(x,i,b)){i++; n--; o.nesterov(b);}
 if(n) AT_ERROR("Unrecognized arg(s) for SGD optimizer");
 while(xpair(p))
  switch(oset(p.k)) {
   case Setting::lr:        f=pdouble(p); if(f==f) o.learning_rate(f); break;
   case Setting::momentum:  f=pdouble(p); if(f==f) o.momentum(f); break;
   case Setting::dampening: f=pdouble(p); if(f==f) o.dampening(f); break;
   case Setting::decay:     f=pdouble(p); if(f==f) o.weight_decay(f); break;
   case Setting::nesterov:  o.nesterov(pbool(p)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for SGD optimization"); break;
  }
}

static Optptr sgd(const TensorVector& w,const SGDOptions& a,K y) {
 auto o=std::make_shared<SGD>(w,a);
 auto n=osize(o->parameters());
 if(y && n)
  bset(n, "momentum_buffers", o->parameters(), o->momentum_buffers, y);
 return o;
}

static K sgd(B a,F r,SGD* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); SGDOptions d(r),o=v->options;
 if(a || d.learning_rate() != o.learning_rate()) OPTSET(x, lr,        kf(o.learning_rate()));
 if(a || d.momentum()      != o.momentum())      OPTSET(x, momentum,  kf(o.momentum()));
 if(a || d.dampening()     != o.dampening())     OPTSET(x, dampening, kf(o.dampening()));
 if(a || d.weight_decay()  != o.weight_decay())  OPTSET(x, decay,     kf(o.weight_decay()));
 if(a || d.nesterov()      != o.nesterov())      OPTSET(x, nesterov,  kb(o.nesterov()));
 return x;
}

static K sgd(SGD* v) {  //return internal buffer state as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0));
 OPTBUFFER(x,v,momentum_buffers);
 return x;
}

// ---------------------------------------------------------------------------------------
// optparms - return vector of parameters from given tensor/sequential ptr
// optinit - initialize one of the supported optimizers, return pointer to k
// optstate - return optimizer name & options and optionally, internal buffer values
// opt - main optimizer interface function for q
// ---------------------------------------------------------------------------------------
static TensorVector optparms(K x,J i) {
 if(auto *a=xseq(x,i))
  return (*a)->parameters();
 else if(auto *a=xvec(x,i))
  return *a;
 else if(auto *a=xten(x,i))
  return {*a};
 else if(x->t==-KS || xempty(x,i))
  return {};
 else if(xptr(x,i))
  AT_ERROR("Unrecognized pointer, expecting tensor(s) or module(s)");
 else
  AT_ERROR("Unrecognized argument, ",kname(x->t ? x->t : kK(x)[i]->t),", expecting tensor(s) or module(s)");
}

static K optinit(S s,K x,K y=nullptr);  //s:optimizer name, x:options, y:buffers
static K optinit(S s,K x,K y) {
 J i=xdict(x) ? -1 : 2; Cast c; F r; omap(s,c,r);
 if(!(x->t==-KS || xdict(x) || xempty(x,1) || xptr(x,1)))
  AT_ERROR("Optimizer ",s," expects args of form:\n",
           "name\n", "(name; parm(s); option(s)..)\n" "(saved state; parm(s))");
 auto w=optparms(x,1); Optptr o;
 switch(c) {
  case Cast::adagrad: {auto a=AdagradOptions(r); adagrad(x,i,a); o=adagrad(w,a,y); break;}
  case Cast::adam:    {auto a=AdamOptions(r);    adam(x,i,a);    o=adam(w,a,y);    break;}
  case Cast::lbfgs:   {auto a=LBFGSOptions(r);   lbfgs(x,i,a);   o=lbfgs(w,a,y);   break;}
  case Cast::rmsprop: {auto a=RMSpropOptions(r); rmsprop(x,i,a); o=rmsprop(w,a,y); break;}
  case Cast::sgd:     {auto a=SGDOptions(r);     sgd(x,i,a);     o=sgd(w,a,y);     break;}
  default: AT_ERROR("Unrecognized optimizer: ",s); break;
 }
 return kptr(new Kopt(c,o));
}

static K optstate(B a,B b,Cast c,OptimizerBase *o) {
 F r; S s; omap(c,s,r); K k,v,x,y;
 switch(c) {
  case Cast::adagrad: {auto m=(Adagrad*)o; x=adagrad(a,r,m); if(b) y=adagrad(m); break;}
  case Cast::adam:    {auto m=(Adam*)o;    x=adam(a,r,m);    if(b) y=adam(m);    break;}
  case Cast::lbfgs:   {auto m=(LBFGS*)o;   x=lbfgs(a,r,m);   if(b) y=lbfgs(m);   break;}
  case Cast::rmsprop: {auto m=(RMSprop*)o; x=rmsprop(a,r,m); if(b) y=rmsprop(m); break;}
  case Cast::sgd:     {auto m=(SGD*)o;     x=sgd(a,r,m);     if(b) y=sgd(m);     break;}
  default: AT_ERROR("Unrecognized optimizer; ",(I)c); break;
 }
 if(b) {
  k=statekeys(); v=ktn(0,k->n);
  kK(v)[0]=kc('o');   //class="o" for optimizer
  kK(v)[2]=ks((S)""); //no user-defined name
  kK(v)[4]=ktn(0,0);  //empty parms
  kK(v)[5]=y;         //retrieved optimizer buffers
 } else {
  k=ktn(KS,2),v=ktn(0,2);
  kS(k)[0]=statekey(State::module), kS(k)[1]=statekey(State::options);
 }
 kK(v)[b ? 1 : 0]=ks(s);
 kK(v)[b ? 3 : 1]=x;
 return xD(k,v);
}

// this version of optstate() called from generic state() function in k-level api
K optstate(Ktag *g,K x) {
 B a=env().alloptions;
 if(x->n==1 || (x->n==2 && xbool(x,1,a)))
  return optstate(a,true,g->c,((Kopt*)g)->o.get());
 else
  AT_ERROR("Optimizer state requires 1-2 args: previously allocated ptr or (ptr;options flag)");
}

// ---------------------------------------------------------------------------------------
// opt - main optimizer interface function for q
// step
// lr - query or set learning rate from k given pre-allocated optimizer ptr, or ptr & rate
// ---------------------------------------------------------------------------------------
KAPI opt(K x) {
 KTRY
  B a=env().alloptions; S s; Kopt *o;
  if(xsym(x,s) || (xsym(x,0,s) && (xptr(x,1) || xempty(x,1)))) {
   return optinit(s,x);
  } else if(xdict(x)) {
   return optinit(statesym(State::module,x),
                  statedict(State::options,x));
  } else if(xdict(x,0) && x->n==2 && (xptr(x,1) || xempty(x,1))) {
   K d=kK(x)[0];
   return optinit(statesym(State::module,d),
                  statedict(State::options,d),
                  statedict(State::buffers,d));
  } else if((o=xoptim(x)) || (xbool(x,1,a) && (o=xoptim(x,0)) && x->n==2)) {
   return optstate(a,false,o->c,o->get());
  } else if((o=xoptim(x,0)) && xptr(x,1) && x->n==2) {
   return o->get()->add_parameters(optparms(x,1)), (K)0;
  } else {
   AT_ERROR("Unrecognized optimizer arg(s)");
  }
 KCATCH("Optimizer error");
}

KAPI kstep(K x) {
 Kopt *o;
 if((o=xoptim(x))) {
  if(o->c == Cast::lbfgs)
   AT_ERROR("LBFGS optimizer requires model, loss & inputs");
  ((Optimizer*)o->get())->step();
 }
 return (K)0;
}

KAPI lr(K x) {
 KTRY
  B b=false; F r; Ktag *g;
  TORCH_CHECK((g=xtag(x)) || ((g=xtag(x,0)) && (b=x->n==2) && xdouble(x,1,r)),
   "lr: unrecognized arg(s), expecting model/optimizer and optional learning rate");
  Cast c; OptimizerBase *o; Kmodel *m;
  switch(g->a) {
   case Class::optimizer: c=g->c; o=((Kopt*)g)->o.get(); break;
   case Class::model: m=(Kmodel*)g; c=m->oc; o=m->o.get(); break;
   default: AT_ERROR("lr not implemented for ",mapclass(g->a));
  }
  switch(c) {
   case Cast::adagrad: {auto& p=((Adagrad*)o)->options; if(b) p.learning_rate(r); else r=p.learning_rate(); break;}
   case Cast::adam:    {auto& p=((Adam*)o)->options;    if(b) p.learning_rate(r); else r=p.learning_rate(); break;}
   case Cast::lbfgs:   {auto& p=((LBFGS*)o)->options;   if(b) p.learning_rate(r); else r=p.learning_rate(); break;}
   case Cast::rmsprop: {auto& p=((RMSprop*)o)->options; if(b) p.learning_rate(r); else r=p.learning_rate(); break;}
   case Cast::sgd:     {auto& p=((SGD*)o)->options;     if(b) p.learning_rate(r); else r=p.learning_rate(); break;}
   default: AT_ERROR("Unrecognized optimizer; ",(I)c); break;
  }
  return b ? (K)0 : kf(r);
 KCATCH("lr");
}

void optfn(K x) {
 fn(x, "opt",  KFN(opt),1);
 fn(x, "step", KFN(kstep),1);
 fn(x, "lr",   KFN(lr),1);
}
