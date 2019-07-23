#include "ktorch.h"

#define OPTBUFFER(x,o,k) dictadd(x, #k, kvec(o->k))
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

/*
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

// --------------------------------------------------------------------------------------
// omap - map to/from optimizer symbol/enumeration and default learning rate
// oset - optimizer settings, map sym -> enum
// odef - read/write optimizer definition via options structure, e.g. AdamOptions
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

ZS omap(Cast c) {
 for(auto& m:env().opt) if(c==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized optimizer: ",(int)c);
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
ZV odef(Cast c, const V* v, S s, Setting o, Pairs& p, K x) {
 switch(c) {
  case Cast::adagrad: {
   auto a=(torch::optim::AdagradOptions*)v;
   switch(o) {
    case Setting::lr:      if(x) dictadd(x,s,kf(a->learning_rate())); else a->learning_rate(pdouble(p)); break;
    case Setting::lrdecay: if(x) dictadd(x,s,kf(a->lr_decay()));      else a->lr_decay(pdouble(p));      break;
    case Setting::decay:   if(x) dictadd(x,s,kf(a->weight_decay()));  else a->weight_decay(pdouble(p));  break;
    default: if(!x) AT_ERROR("Adagrad setting: ",s," not one of lr,lrdecay,decay"); break;
   }
   break;
  }
  case Cast::adam: {
   auto a=(torch::optim::AdamOptions*)v;
   switch(o) {
    case Setting::lr:      if(x) dictadd(x,s,kf(a->learning_rate())); else a->learning_rate(pdouble(p)); break;
    case Setting::beta1:   if(x) dictadd(x,s,kf(a->beta1()));         else a->beta1(pdouble(p));         break;
    case Setting::beta2:   if(x) dictadd(x,s,kf(a->beta2()));         else a->beta2(pdouble(p));         break;
    case Setting::decay:   if(x) dictadd(x,s,kf(a->weight_decay()));  else a->weight_decay(pdouble(p));  break;
    case Setting::eps:     if(x) dictadd(x,s,kf(a->eps()));           else a->eps(pdouble(p));           break;
    case Setting::amsgrad: if(x) dictadd(x,s,kb(a->amsgrad()));       else a->amsgrad(pbool(p));         break;
    default: if(!x) AT_ERROR("Adam setting: ",s," not one of lr,beta1,beta2,decay,eps,amsgrad"); break;
   }
   break;
  }
  case Cast::lbfgs: {
   auto a=(torch::optim::LBFGSOptions*)v;
   switch(o) {
    case Setting::lr:        if(x) dictadd(x,s,kf(a->learning_rate()));    else a->learning_rate(pdouble(p));    break;
    case Setting::iter:      if(x) dictadd(x,s,kj(a->max_iter()));         else a->max_iter(plong(p));           break;
    case Setting::eval:      if(x) dictadd(x,s,kj(a->max_eval()));         else a->max_eval(plong(p));           break;
    case Setting::gradtol:   if(x) dictadd(x,s,ke(a->tolerance_grad()));   else a->tolerance_grad(pdouble(p));   break;
    case Setting::changetol: if(x) dictadd(x,s,ke(a->tolerance_change())); else a->tolerance_change(pdouble(p)); break;
    case Setting::history:   if(x) dictadd(x,s,kj(a->history_size()));     else a->history_size(plong(p));       break;
    default: if(!x) AT_ERROR("LBFGS setting: ",s," not one of lr,iter,eval,gradtol,changetol,history"); break;
   }
   break;
  }
  case Cast::rmsprop: {
   auto a=(torch::optim::RMSpropOptions*)v;
   switch(o) {
    case Setting::lr:        if(x) dictadd(x,s,kf(a->learning_rate())); else a->learning_rate(pdouble(p));break;
    case Setting::alpha:     if(x) dictadd(x,s,kf(a->alpha()));         else a->alpha(pdouble(p));        break;
    case Setting::eps:       if(x) dictadd(x,s,kf(a->eps()));           else a->eps(pdouble(p));          break;
    case Setting::decay:     if(x) dictadd(x,s,kf(a->weight_decay()));  else a->weight_decay(pdouble(p)); break;
    case Setting::momentum:  if(x) dictadd(x,s,kf(a->momentum()));      else a->momentum(pdouble(p));     break;
    case Setting::centered:  if(x) dictadd(x,s,kb(a->centered()));      else a->centered(pbool(p));       break;
    default: if(!x) AT_ERROR("RMSprop setting: ",s," not one of lr,alpha,eps,decay.momentum,centered");  break;
   }
   break;
  }
  case Cast::sgd: {
   auto a=(torch::optim::SGDOptions*)v;
   switch(o) {
    case Setting::lr:        if(x) dictadd(x,s,kf(a->learning_rate())); else a->learning_rate(pdouble(p)); break;
    case Setting::momentum:  if(x) dictadd(x,s,kf(a->momentum()));      else a->momentum(pdouble(p));      break;
    case Setting::dampening: if(x) dictadd(x,s,kf(a->dampening()));     else a->dampening(pdouble(p));     break;
    case Setting::decay:     if(x) dictadd(x,s,kf(a->weight_decay()));  else a->weight_decay(pdouble(p));  break;
    case Setting::nesterov:  if(x) dictadd(x,s,kb(a->nesterov()));      else a->nesterov(pbool(p));        break;
    default: if(!x) AT_ERROR("SGD setting: ",s," not one of lr,momentum,dampening,decay,nesterov"); break;
   }
   break;
  }
  default: AT_ERROR("Unrecognized optimizer"); break;
 }
}

// ----------------------------------------------------------------------------------------
// adagrad - parse args (parms; lr; lrdecay; weightdecay) or (parms;..;name/val pairs/dict)
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
// adagrad - parse args (parms; lr; beta1; beta2; eps; weightdecay; amsgrad)
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

//LBFGS(parms, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100)
ZK lbfgs(B a,F r,LBFGS* v) { //return all or non-default options as k dictionary
 K x=xD(ktn(KS,0),ktn(0,0)); LBFGSOptions d(r),o=v->options;
 if(a || d.max_iter()         != o.max_iter())         OPTSET(x, iter,      kj(o.max_iter()));
 if(a || d.max_eval()         != o.max_eval())         OPTSET(x, eval,      kj(o.max_eval()));
 if(a || d.learning_rate()    != o.learning_rate())    OPTSET(x, lr,        kf(o.learning_rate()));
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

//RMSprop(parms, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
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

//SGD(parms, lr=??,    momentum=0, dampening=0, weight_decay=0, nesterov=False)
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
// optdict - return a dictionary of optimizer settings
// optdefault - return a dictionary of default settings for a given optimizer
// optdefaults - return dictionary of dictionaries: default settings for all optimizers
// optpairs - process dictionary or list of name,value pairs of optimizer settings
// optparms - set vector of parameters from given tensor(s),layer(s),model
// optinit - initialize one of the supported optimizers, return pointer to k
// opt - main optimizer interface function for q
// ---------------------------------------------------------------------------------------
ZV optdict(K &x,Cast c,const V *v) {
 Pairs p;
 if(v)
  for(auto& m:env().oset)
   odef(c,v,std::get<0>(m),std::get<1>(m),p,x);
}

ZK optdict(Cast c,const V *v) {K x=xD(ktn(KS,0),ktn(0,0)); optdict(x,c,v); return x;}

ZK optdefault(S s,Cast c,double r) {
 K x;
 switch(c) {
  case Cast::adagrad: {auto a=torch::optim::AdagradOptions(r); x=optdict(c,&a); break;}
  case Cast::adam:    {auto a=torch::optim::AdamOptions(r);    x=optdict(c,&a); break;}
  case Cast::lbfgs:   {auto a=torch::optim::LBFGSOptions(r);   x=optdict(c,&a); break;}
  case Cast::rmsprop: {auto a=torch::optim::RMSpropOptions(r); x=optdict(c,&a); break;}
  case Cast::sgd:     {auto a=torch::optim::SGDOptions(r);     x=optdict(c,&a); break;}
  default: x=optdict(c,nullptr); break;
 }
 return x;
}

ZK optdefault(S s) {Cast c; double r; omap(s,c,r); return optdefault(s,c,r);}

ZK optdefaults(V) {
 K d=xD(ktn(KS,0),ktn(0,0)), *y=kK(d);
 for(auto& m:env().opt) {
  auto s=std::get<0>(m);    // sym for optimizer
  auto c=std::get<1>(m);    // corresponding enum
  auto r=std::get<2>(m);    // default learning rate
  js(&y[0],s); jk(&y[1],optdefault(s,c,r));
 }
 return d;
}

ZV optpairs(Cast c,V *v,Pairs &p) {
 while(xpair(p)) odef(c,v,p.k,oset(p.k),p,nullptr);
}

ZV optparms(Ptr p,std::vector<Tensor>& v) {
 switch(p->t) {
  case Class::tensor:     v.emplace_back(*(Tensor*)p->v); break;
  case Class::sequential: 
  default: break;
 }
}

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
/*
  case Cast::lbfgs:   {auto a=LBFGSOptions(r);   lbfgs(x,2,a);   u->v=new LBFGS(w,a);   break;}
  case Cast::rmsprop: {auto a=RMSpropOptions(r); rmsprop(x,2,a); u->v=new RMSprop(w,a); break;}
  case Cast::sgd:     {auto a=SGDOptions(r);     sgd(x,2,a);     u->v=new SGD(w,a);     break;}
*/
  default: AT_ERROR("Unrecognized optimizer: ",s); break;
 }
 return kptr(u.release());
}

ZK optinit(S s,F lr,Ptr &k,Pairs &p) {
 Cast c; double r; std::vector<Tensor> v;
 if(k) optparms(k,v);              // set vector of tensor parms to optimize
 omap(s,c,r);
 if(lr == lr) r=lr; // use learning rate if specified directly
 auto u=torch::make_unique<Obj>(); u->t=Class::optimizer; u->c=c;
 switch(c) {
  case Cast::adagrad: {auto a=torch::optim::AdagradOptions(r); optpairs(c,&a,p); u->v=new torch::optim::Adagrad(v,a); break;}
  case Cast::adam:    {auto a=torch::optim::AdamOptions(r);    optpairs(c,&a,p); u->v=new torch::optim::Adam(v,a);    break;}
  case Cast::lbfgs:   {auto a=torch::optim::LBFGSOptions(r);   optpairs(c,&a,p); u->v=new torch::optim::LBFGS(v,a);   break;}
  case Cast::rmsprop: {auto a=torch::optim::RMSpropOptions(r); optpairs(c,&a,p); u->v=new torch::optim::RMSprop(v,a); break;}
  case Cast::sgd:     {auto a=torch::optim::SGDOptions(r);     optpairs(c,&a,p); u->v=new torch::optim::SGD(v,a);     break;}
  default: AT_ERROR("Unrecognized optimizer: ",s); break;
 }
 return kptr(u.release());
}

ZK optstate(B a,B b,Cast c,V* v) {
 F r; S s; omap(c,s,r); K x,y;
 switch(c) {
  case Cast::adagrad: {auto m=(Adagrad*)v; x=adagrad(a,r,m); if(b) y=adagrad(m); break;}
  case Cast::adam:    {auto m=(Adam*)v;    x=adam(a,r,m);    if(b) y=adam(m);    break;}
  case Cast::lbfgs:   {auto m=(LBFGS*)v;   x=lbfgs(a,r,m);   if(b) y=lbfgs(m);   break;}
  case Cast::rmsprop: {auto m=(RMSprop*)v; x=rmsprop(a,r,m); if(b) y=rmsprop(m); break;}
  case Cast::sgd:     {auto m=(SGD*)v;     x=sgd(a,r,m);     if(b) y=sgd(m);    break;}
  default: break;
 }
 return (K)0;
}

ZK optdetail1(Cast c,V *v,const std::vector<Tensor>& p) {
 S s=omap(c); K x=xD(ktn(KS,0),ktn(0,0)), *k=kK(x); J n=p.size();
 js(&k[0],cs("optimizer")); jk(&k[1],ks(s));
 optdict(x,c,v); js(&k[0],cs("size")); jk(&k[1],kj(n));
 K w=ktn(0,n),g=ktn(0,n);
 for(I i=0;i<n;++i) {
  Tensor t=p[i];
  if(t.defined()) {
   kK(w)[i]=kget(t);
   if(t.grad().defined())
    kK(g)[i]=kget(t.grad());
   else
    kK(g)[i]=ktn(0,0);
  } else {
    kK(w)[i]=ktn(0,0); kK(g)[i]=ktn(0,0);
  }
 }
 js(&k[0],cs("weight"));   jk(&k[1],w);
 js(&k[0],cs("gradient")); jk(&k[1],g);
 return x;
}

K optdetail(Cast c,V *v) {
 switch(c) {
  case Cast::adagrad: {auto a=(torch::optim::Adagrad*)v; return optdetail1(c,&a->options,a->parameters());}
  case Cast::adam:    {auto a=(torch::optim::Adam*)v;    return optdetail1(c,&a->options,a->parameters());}
  case Cast::lbfgs:   {auto a=(torch::optim::LBFGS*)v;   return optdetail1(c,&a->options,a->parameters());}
  case Cast::rmsprop: {auto a=(torch::optim::RMSprop*)v; return optdetail1(c,&a->options,a->parameters());}
  case Cast::sgd:     {auto a=(torch::optim::SGD*)v;     return optdetail1(c,&a->options,a->parameters());}
  default: AT_ERROR("Unrecognized optimizer: ",(int)c); return (K)0;
 }
}

KAPI opt(K x) {
 F r=nf; S s; Ptr k=nullptr; Pairs p={};
 KTRY
 if(xempty(x)) {
  return optdefaults();
/*
 } else if(xsym(x,s)) {
  return optdefault(s);
 } else if(xsym(x,0,s) && (xptr(x,1,k) || xempty(x,1)) && (x->n==2 || (x->n==3 && (xdouble(x,2,r) || xpairs(x,2,p))))) {
  return optinit(s,r,k,p);
*/
 } else if(xsym(x,s) || xsym(x,0,s)) {
  return optinit(s,x);
 } else if(xoptim(x,k)) {
  return optdetail(k->c,k->v);
 } else {
  return(K)0;
 }
 KCATCH("Optimizer error");
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

*/
