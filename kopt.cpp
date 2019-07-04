#include "ktorch.h"

// opt()
// opt`adam
// optdetail(o;..)  and/or opt(o;`parms) opt(o;`settings)
// opt(`adam;p)
// opt(`adam;p;..)
// step(o)
// step(o;fn)

// --------------------------------------------------------------------------------------
// omap - map optimizer symbol -> enum and default learning rate, map enum back to sym
// oset - optimizer settings, map sym -> enum
// odef - read/write optimizer definition via options structure, e.g. AdamOptions
// --------------------------------------------------------------------------------------
ZV omap(S &s,Cast &c,double &r) {
 for(auto& m:env().opt)
   if(s==std::get<0>(m)) {c=std::get<1>(m); r=std::get<2>(m); return;}
 AT_ERROR("Unrecognized optimizer: ",s);
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
  case Cast::lbgfs: {
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
  case Cast::lbgfs:   {auto a=torch::optim::LBFGSOptions(r);   x=optdict(c,&a); break;}
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

ZV optparms(Ptr k,std::vector<Tensor>& v) {
 switch(k->t) {
  case Tag::tensor: v.emplace_back(*(Tensor*)k->v); break;
  default: break;
 }
}

ZK optinit(S s,F lr,Ptr &k,Pairs &p) {
 Cast c; double r; std::vector<Tensor> v;
 if(k) optparms(k,v);              // set vector of tensor parms to optimize
 omap(s,c,r);
 if(lr == lr) r=lr; // use learning rate if specified directly
 auto u=torch::make_unique<Obj>(); u->t=Tag::optimizer; u->c=c;
 switch(c) {
  case Cast::adagrad: {auto a=torch::optim::AdagradOptions(r); optpairs(c,&a,p); u->v=new torch::optim::Adagrad(v,a); break;}
  case Cast::adam:    {auto a=torch::optim::AdamOptions(r);    optpairs(c,&a,p); u->v=new torch::optim::Adam(v,a);    break;}
  case Cast::lbgfs:   {auto a=torch::optim::LBFGSOptions(r);   optpairs(c,&a,p); u->v=new torch::optim::LBFGS(v,a);   break;}
  case Cast::rmsprop: {auto a=torch::optim::RMSpropOptions(r); optpairs(c,&a,p); u->v=new torch::optim::RMSprop(v,a); break;}
  case Cast::sgd:     {auto a=torch::optim::SGDOptions(r);     optpairs(c,&a,p); u->v=new torch::optim::SGD(v,a);     break;}
  default: AT_ERROR("Unrecognized optimizer: ",s); break;
 }
 return kptr(u.release());
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
  case Cast::lbgfs:   {auto a=(torch::optim::LBFGS*)v;   return optdetail1(c,&a->options,a->parameters());}
  case Cast::rmsprop: {auto a=(torch::optim::RMSprop*)v; return optdetail1(c,&a->options,a->parameters());}
  case Cast::sgd:     {auto a=(torch::optim::SGD*)v;     return optdetail1(c,&a->options,a->parameters());}
  default: AT_ERROR("Unrecognized optimizer: ",(int)c); return (K)0;
 }
}

// opt(`name;layer/model/tensor/tensors/(empty));rate)
// opt(`name;layer/model/tensor/tensors/(empty));setting(s))
// opt(o;layer/model/tensor/tensors) -> add

KAPI opt(K x) {
 F r=nf; S s; Ptr k=nullptr; Pairs p={};
 KTRY
 if(xempty(x)) {
  return optdefaults();
 } else if(xsym(x,s)) {
  return optdefault(s);
 } else if(xsym(x,0,s) && (xptr(x,1,k) || xempty(x,1)) && (x->n==2 || (x->n==3 && (xdouble(x,2,r) || xpairs(x,2,p))))) {
  return optinit(s,r,k,p);
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
  case Cast::lbgfs:   delete(torch::optim::LBFGS*)v;   break;
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
