#include "ktorch.h"

#define OPTBUFFER(x,o,k) dictadd(x, #k, kvec(o->k))
#define OPTSET(x,k,v) dictadd(x, oset(Setting::k), v)

using OptimizerBase  = torch::optim::detail::OptimizerBase;
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
// omap - map to/from optimizer symbol/enumeration and default learning rate
// oset - optimizer settings, map sym <-> enum
// --------------------------------------------------------------------------------------
ZV omap(S s,Cast &c,double &r) {
 for(auto& m:env().opt)
   if(s==std::get<0>(m)) {c=std::get<1>(m); r=std::get<2>(m); return;}
 AT_ERROR("Unrecognized optimizer: ",s);
}

ZV omap(Cast c,S &s,double &r) {
 for(auto& m:env().opt)
   if(c==std::get<1>(m)) {s=std::get<0>(m); r=std::get<2>(m); return;}
 AT_ERROR("Unrecognized optimizer: ",(I)c);
}

Z Setting oset(S s) {
 for(auto& m:env().oset)
  if(s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("Unrecognized optimizer setting: ",s);
}

ZS oset(Setting e) {
 for(auto& m:env().oset) if(e==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized optimizer setting: ",(I)e);
}

// ----------------------------------------------------------------------------------------
// adagrad - parse args (lr;lrdecay;wtdecay) or (..;name/val pairs/dict)
// ----------------------------------------------------------------------------------------
ZV adagrad(K x,J i,AdagradOptions& o) {
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

ZK adagrad(B a,F r,Adagrad* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); AdagradOptions d(r),o=v->options;
 if(a || d.learning_rate() != o.learning_rate()) OPTSET(x, lr,      kf(o.learning_rate()));
 if(a || d.lr_decay()      != o.lr_decay())      OPTSET(x, lrdecay, kf(o.lr_decay()));
 if(a || d.weight_decay()  != o.weight_decay())  OPTSET(x, decay,   kf(o.weight_decay()));
 return x;
}

ZK adagrad(Adagrad* v) {  //return internal buffer state as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0));
 OPTBUFFER(x,v,step_buffers);
 OPTBUFFER(x,v,sum_buffers);
 return x;
}

// ----------------------------------------------------------------------------------------
// adam - parse args (lr;beta1;beta2;eps;wtdecay;amsgrad) or (..;name-value pairs/dict)
// ----------------------------------------------------------------------------------------
ZV adam(K x,J i,AdamOptions& o) {
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

ZK adam(B a,F r,Adam* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); AdamOptions d(r),o=v->options;
 if(a || d.learning_rate() != o.learning_rate()) OPTSET(x, lr,      kf(o.learning_rate()));
 if(a || d.beta1()         != o.beta1())         OPTSET(x, beta1,   kf(o.beta1()));
 if(a || d.beta2()         != o.beta2())         OPTSET(x, beta2,   kf(o.beta2()));
 if(a || d.weight_decay()  != o.weight_decay())  OPTSET(x, decay,   kf(o.weight_decay()));
 if(a || d.eps()           != o.eps())           OPTSET(x, eps,     kf(o.eps()));
 if(a || d.amsgrad()       != o.amsgrad())       OPTSET(x, amsgrad, kb(o.amsgrad()));
 return x;
}

ZK adam(Adam* v) {  //return internal buffer state as k dictionary
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
ZV lbfgs(K x,J i,LBFGSOptions& o) {
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

ZK lbfgs(B a,F r,LBFGS* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); LBFGSOptions d(r),o=v->options;
 if(a || d.learning_rate()    != o.learning_rate())    OPTSET(x, lr,        kf(o.learning_rate()));
 if(a || d.max_iter()         != o.max_iter())         OPTSET(x, iter,      kj(o.max_iter()));
 if(a || d.max_eval()         != o.max_eval())         OPTSET(x, eval,      kj(o.max_eval()));
 if(a || d.tolerance_grad()   != o.tolerance_grad())   OPTSET(x, gradtol,   kf(o.tolerance_grad()));
 if(a || d.tolerance_change() != o.tolerance_change()) OPTSET(x, changetol, kf(o.tolerance_change()));
 if(a || d.history_size()     != o.history_size())     OPTSET(x, history,   kj(o.history_size()));
 return x;
}

ZK lbfgs(LBFGS* v) {  //return internal buffer state as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0));
 //OPTBUFFER(x,v,momentum_buffers);
 return x;
}

// ----------------------------------------------------------------------------------------
// rmsprop - parse arg(s) (lr;alpha;eps;decay;momentum;centered) or (..;nm-val pairs/dict)
// ----------------------------------------------------------------------------------------
ZV rmsprop(K x,J i,RMSpropOptions& o) {
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

ZK rmsprop(B a,F r,RMSprop* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); RMSpropOptions d(r),o=v->options;
 if(a || d.learning_rate() != o.learning_rate()) OPTSET(x, lr,       kf(o.learning_rate()));
 if(a || d.alpha()         != o.alpha())         OPTSET(x, alpha,    kf(o.alpha()));
 if(a || d.eps()           != o.eps())           OPTSET(x, eps,      kf(o.eps()));
 if(a || d.weight_decay()  != o.weight_decay())  OPTSET(x, decay,    kf(o.weight_decay()));
 if(a || d.momentum()      != o.momentum())      OPTSET(x, momentum, kf(o.momentum()));
 if(a || d.centered()      != o.centered())      OPTSET(x, centered, kb(o.centered()));
 return x;
}

ZK rmsprop(RMSprop* v) {  //return internal buffer state as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0));
 OPTBUFFER(x,v,square_average_buffers);
 OPTBUFFER(x,v,momentum_buffers);
 OPTBUFFER(x,v,grad_average_buffers);
 return x;
}

// ----------------------------------------------------------------------------------------
// SGD parse args (lr;momentum;dampening;wtdecay;nesterov) or (..;name-value pairs/dict)
// ----------------------------------------------------------------------------------------
ZV sgd(K x,J i,SGDOptions& o) {
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

ZK sgd(B a,F r,SGD* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); SGDOptions d(r),o=v->options;
 if(a || d.learning_rate() != o.learning_rate()) OPTSET(x, lr,        kf(o.learning_rate()));
 if(a || d.momentum()      != o.momentum())      OPTSET(x, momentum,  kf(o.momentum()));
 if(a || d.dampening()     != o.dampening())     OPTSET(x, dampening, kf(o.dampening()));
 if(a || d.weight_decay()  != o.weight_decay())  OPTSET(x, decay,     kf(o.weight_decay()));
 if(a || d.nesterov()      != o.nesterov())      OPTSET(x, nesterov,  kb(o.nesterov()));
 return x;
}

ZK sgd(SGD* v) {  //return internal buffer state as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0));
 OPTBUFFER(x,v,momentum_buffers);
 return x;
}

// ---------------------------------------------------------------------------------------
// optparms - return vector of parameters from given tensor/sequential ptr
// optinit - initialize one of the supported optimizers, return pointer to k
// optstate - return optimizer name & options and optionally, internal buffer values
// optfree - free previously allocated optimizer module
// opt - main optimizer interface function for q
// ---------------------------------------------------------------------------------------
Z std::vector<Tensor> optparms(Ptr p) {
 switch(p ? p->t : Class::undefined) {
  case Class::tensor:     return {*(Tensor*)p->v};
  case Class::sequential: return {(*(Sequential*)p->v)->parameters()};
  case Class::undefined:  return {};
  default: AT_ERROR("Unrecognized pointer, expecting tensor or module(s)");
 }
}

ZK optinit(S s,K x) {
 Cast c; F r; omap(s,c,r); Ptr p=nullptr;
 if(!(x->t==-KS || xempty(x,1) || xptr(x,1,p)))
  AT_ERROR("Optimizer ",s," expects args of name or (name; tensor/module/empty; option(s)..)");
 auto w=optparms(p); auto u=torch::make_unique<Obj>(); u->t=Class::optimizer; u->c=c;
 switch(c) {
  case Cast::adagrad: {auto a=AdagradOptions(r); adagrad(x,2,a); u->v=new Adagrad(w,a); break;}
  case Cast::adam:    {auto a=AdamOptions(r);    adam(x,2,a);    u->v=new Adam(w,a);    break;}
  case Cast::lbfgs:   {auto a=LBFGSOptions(r);   lbfgs(x,2,a);   u->v=new LBFGS(w,a);   break;}
  case Cast::rmsprop: {auto a=RMSpropOptions(r); rmsprop(x,2,a); u->v=new RMSprop(w,a); break;}
  case Cast::sgd:     {auto a=SGDOptions(r);     sgd(x,2,a);     u->v=new SGD(w,a);     break;}
  default: AT_ERROR("Unrecognized optimizer: ",s); break;
 }
 return kptr(u.release());
}

K optstate(B a,B b,Ptr p) {
 F r; S s; omap(p->c,s,r); K k,v,x,y;
 switch(p->c) {
  case Cast::adagrad: {auto m=(Adagrad*)p->v; x=adagrad(a,r,m); if(b) y=adagrad(m); break;}
  case Cast::adam:    {auto m=(Adam*)p->v;    x=adam(a,r,m);    if(b) y=adam(m);    break;}
  case Cast::lbfgs:   {auto m=(LBFGS*)p->v;   x=lbfgs(a,r,m);   if(b) y=lbfgs(m);   break;}
  case Cast::rmsprop: {auto m=(RMSprop*)p->v; x=rmsprop(a,r,m); if(b) y=rmsprop(m); break;}
  case Cast::sgd:     {auto m=(SGD*)p->v;     x=sgd(a,r,m);     if(b) y=sgd(m);     break;}
  default: AT_ERROR("Unrecognized optimizer; ",(I)p->c); break;
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

V optfree(Cast c,V *v) {
 switch(c) {
  case Cast::adagrad: delete(torch::optim::Adagrad*)v; break;
  case Cast::adam:    delete(torch::optim::Adam*)v;    break;
  case Cast::lbfgs:   delete(torch::optim::LBFGS*)v;   break;
  case Cast::rmsprop: delete(torch::optim::RMSprop*)v; break;
  case Cast::sgd:     delete(torch::optim::SGD*)v;     break;
  default: AT_ERROR("Unrecognized optimizer: ",(I)c); break;
 }
}

KAPI opt(K x) {
 KTRY
  B a=env().alloptions; S s; Ptr p,q;
  if(xsym(x,s) || xsym(x,0,s)) {
   return optinit(s,x);
  } else if(xoptim(x,p) || (xbool(x,1,a) && xoptim(x,0,p) && x->n==2)) {
   return optstate(a,false,p);
  } else if(xoptim(x,0,p) && xptr(x,1,q)) {
   return ((OptimizerBase*)p->v)->add_parameters(optparms(q)), (K)0;
  } else {
   return(K)0;
  }
 KCATCH("Optimizer error");
}

V optfn(K x) {
 fn(x, "opt",  KFN(opt),1);
}

/*
s         a                k                                         
---------------------------------------------------------------------
amsgrad   amsgrad          TORCH_ARG(bool, amsgrad) = false;         
centered  centered         TORCH_ARG(bool, centered) = false;        
nesterov  nesterov         TORCH_ARG(bool, nesterov) = false;        
alpha     alpha            TORCH_ARG(double, alpha) = 0.99;          
beta1     beta1            TORCH_ARG(double, beta1) = 0.9;           
beta2     beta2            TORCH_ARG(double, beta2) = 0.999;         
dampening dampening        TORCH_ARG(double, dampening) = 0;         
eps       eps              TORCH_ARG(double, eps) = 1e-8;            
lr        learning_rate    TORCH_ARG(double, learning_rate);         
lrdecay   lr_decay         TORCH_ARG(double, lr_decay) = 0;          
momentum  momentum         TORCH_ARG(double, momentum) = 0;          
decay     weight_decay     TORCH_ARG(double, weight_decay) = 0;      
changetol tolerance_change TORCH_ARG(float, tolerance_change) = 1e-9;
gradtol   tolerance_grad   TORCH_ARG(float, tolerance_grad) = 1e-5;  
eval      max_eval         TORCH_ARG(int64_t, max_eval) = 25;        
iter      max_iter         TORCH_ARG(int64_t, max_iter) = 20;        
history   history_size     TORCH_ARG(size_t, history_size) = 100;   

KAPI vec(K a) {
 auto m=torch::nn::Linear(5,2); m->to(torch::kCUDA);
 auto o=torch::optim::Adam(m->parameters(),torch::optim::AdamOptions(.001));
 auto x=torch::ones({10,5},torch::kCUDA);
 auto y=m->forward(x).sum();
 y.backward();
 o.step();
 y=m->forward(x).sum();
 y.backward();
 o.step();
 K r=xD(ktn(KS,0),ktn(0,0));
 OPTBUFFER(r,o,step_buffers);
 OPTBUFFER(r,o,exp_average_buffers);
 OPTBUFFER(r,o,exp_average_sq_buffers);
 OPTBUFFER(r,o,max_exp_average_sq_buffers);
 return r;
}
*/
