#include "ktorch.h"
#include "knn.h"

// -------------------------------------------------------------------------------------------
//  metric - map k symbol to metric, e.g. `accuracy -> Metric::accuracy
// -------------------------------------------------------------------------------------------
static Metric metric(S s) {
 for(auto& m:env().metric) 
  if(std::get<0>(m)==s) return std::get<1>(m);
 AT_ERROR("Unrecognized metric: ",s);
}

// -------------------------------------------------------------------------------------------
// modelpart - parse args from k to define sequential, loss & optimizer modules
// modelkeys - return list of symbols used for model state dictionary
// modelstate - return a dictionary with state of sequential module, loss fn & optimizer
// modeltable - state dictionary -> table w'rows for sequential module, loss & optimizer
// model - create model from sequential, loss & optimizer modules or retrieve input options
// -------------------------------------------------------------------------------------------
static void modelpart(K x,J i,Kseq*& q,Kmodule*& l,Kopt*& o) {
 for(;i<x->n;++i) {
  auto* g=xtag(x,i);
  switch(g ? g->a : Class::undefined) {
   case Class::sequential: q=(Kseq*)g;  break;
   case Class::loss:       l=(Kmodule*)g; break;
   case Class::optimizer:  o=(Kopt*)g;  break;
   default: AT_ERROR("model arg[",i,"] unrecognized: ",
                    (g ? mapclass(g->a) : kname(x,i))); break;
  }
 }
}

K modelkeys() {
 K x=ktn(KS,env().model.size());
 for(auto &m:env().model)
  if     (std::get<1>(m)==Class::sequential) kS(x)[0]=std::get<0>(m);
  else if(std::get<1>(m)==Class::loss)       kS(x)[1]=std::get<0>(m);
  else if(std::get<1>(m)==Class::optimizer)  kS(x)[2]=std::get<0>(m);
 return x;
}

K modelstate(bool a,bool b,Kmodel *m) {
 return xD(modelkeys(), knk(3, mtable(m->q,a,b), lossdict(a,b,m->lc,m->l), optstate(a,b,m->oc,m->o.get())));
}

// this version of modelstate called from generic state function in k-level api
K modelstate(Ktag *g,K x) {
 bool a=env().alloptions;
 if(x->n==1 || (x->n==2 && xbool(x,1,a)))
  return modelstate(a,true,(Kmodel*)g);
 else
  AT_ERROR("model state requires 1-2 args: previously allocated ptr or (ptr;options flag)");
}

static void modelfree(K x,J i) {
 for(;i<x->n;++i) 
  TORCH_CHECK(kfree(x,i), "model: unable to free arg[",i,"]");
}

KAPI modeltable(K x) {
 KTRY
  TORCH_CHECK(x->t==99, "modeltable not implemented for ",kname(x->t));
  return k(0,(S)"{raze[x where b],x where not b:98=type each x}",r1(x),0);
 KCATCH("modeltable");
}

KAPI model(K x) {
 KTRY
  bool a=env().alloptions;
  Kseq *q=nullptr; Kmodule *l=nullptr; Kopt *o=nullptr; Kmodel *m=nullptr;
  TORCH_CHECK(!x->t, "model not implemented for ",kname(x->t));
  if((m=xmodel(x)) || (x->n==2 && xbool(x,1,a) && (m=xmodel(x,0)))) {
   return modelstate(a,false,m);
  } else {
   m=xmodel(x,0); modelpart(x,m ? 1 : 0,q,l,o);
   if(m) {
    if(q) m->q=q->q;              // reassign model's sequential module
    if(l) m->lc=l->c, m->l=l->m;  // loss function
    if(o) m->oc=o->c, m->o=o->o;  // optimizer
    modelfree(x,1);
    return (K)0;
   } else {
    TORCH_CHECK(q && l && o, (q ? (l ? "optimizer" : "loss") : "sequential module")," not specified");
    m=new Kmodel(q,l,o);
    modelfree(x,0);
    return kptr(m);
   }
  }
 KCATCH("model");
}

// -------------------------------------------------------------------------------------------
// mforward - return tensor from running sequential forward calcs on input(s)
// mbackward - given model, input & target, do forward calcs, get loss, backward prop on loss
// mloss - given model and vector of inputs, e.g. v=x,y, loss=loss(sequential(v[0]),v[1])
// -------------------------------------------------------------------------------------------
Tensor mforward(Kmodel *m,TensorVector& v) {
 return m->q->forward(v[0]);
}

K mbackward(K a) {
 Kmodel *m; Tensor *x,*y,r;
 if((m=xmodel(a,0)) && (x=xten(a,1)) && (y=xten(a,2))) {
  r=losswt(m->lc,m->l,m->q->forward(*x),*y);
 } else {
  AT_ERROR("backward expects (model; inputs; targets)");
 }
 r.backward();
 return kget(r);
}

Tensor mloss(Kmodel *m,const Tensor& x,const TensorVector &v) {
 if(v.size()==2)
  return losswt(m->lc,m->l,x,v[1]);
 else if(v.size()==3)
  return m->l.forward(x,v[1],v[2]);
 else
  AT_ERROR("model: ", v.size()," inputs given, expecting 2-3");
}

Tensor mloss(Kmodel *m,TensorVector &v) { return mloss(m,mforward(m,v),v);}

// -------------------------------------------------------------------------------------------
// trainbatch - run model's forward calc, loss, backward calcs and optimizer step in batches
// train - train model for given batch size and number of passes through the data ("epochs")
// ktrain - k api fn, expects (model;vector;batch size; optional epochs;optional shuffle flag)
// -------------------------------------------------------------------------------------------
Tensor trainbatch(Kmodel *m,TensorVector& v,int64_t w,int64_t n=0);
Tensor trainbatch(Kmodel *m,TensorVector& v,int64_t w,int64_t n) {
 Optimizer *o=nullptr; LossClosureOptimizer *c=nullptr;
 if(m->oc == Cast::lbfgs) c=(LossClosureOptimizer*)m->o.get();
 else                     o=(Optimizer*)m->o.get();

 if(!n) n=maxsize(v);
 if(w>n) w=n;                          // reduce batch size if exceeds total size
 auto s=subsets(w,n);                  // no. of subsets to process
 auto r=torch::empty(s);               // tensor for batch losses
 auto* p=r.data_ptr<float>();          // ptr for quicker assignment

 auto loss=[&]() {                     // define closure for
  m->o->zero_grad();                   // resetting gradients
  auto r=mloss(m,v);                   // calculating model output & loss
  r.backward();                        // calculating gradients
  return r;                            // return loss tensor
 };

 for(int64_t i=0,j=0; i<n; i+=w,++j) {
  subset(v,0,i,w,n);                   // narrow tensors to current batch
  if(o)
   p[j]=loss().item<float>(), o->step(); // single loss evaluation
  else
   p[j]=c->step(loss).item<float>();     // pass closure, e.g. LBFGS
 }
 subset(v,0,0,n,n);                    // reset tensors to full length
 return r;                             // return losses
}

Tensor train(Kmodel *m,TensorVector& v,int64_t w,int64_t e,bool s) {
 auto n=fullsize(v);
 if(e) {
  TensorVector r;
  for(int64_t i=0; i<e; ++i) {
   if(s) shuffle_(v);
   r.emplace_back(trainbatch(m,v,w,n));
  }
  return torch::stack(r);
 } else {
  if(s) shuffle_(v);
  return trainbatch(m,v,w,n);
 }
}

KAPI ktrain(K x) {
 KTRY
  Kmodel *m; TensorVector *v; bool s=true; int64_t w,e=0;
  TORCH_CHECK(!x->t, "train: not implemented for ",kname(x->t));
  auto a=x->n - xbool(x,x->n-1,s);
  if((m=xmodel(x,0)) && (v=xvec(x,1)) && xint64(x,2,w) && (a==3 || (a==4 && xint64(x,3,e)))) {
   TORCH_CHECK(w>0,  "train: batch size must be positive");
   TORCH_CHECK(e>-1, "train: number of passes cannot be negative");
   TORCH_CHECK(v->size(), "train: vector of inputs is empty");
   return kget(train(m,*v,w,e,s));
  } else {
   return KERR("train: unrecognized arg(s)");
  }
 KCATCH("train");
}

// -------------------------------------------------------------------------------------------
// eval
// -------------------------------------------------------------------------------------------
enum class Eval:char {fwd, max, pct};

static Tensor evalfwd(Sequential& q,Tensor& x,int64_t w) {
 bool b=q->is_training(); Tensor y;
 if(b) q->train(false);                // turn off training mode
 if(w) {                               // if batches of window size w
  auto n=maxsize(x);                   // get maxsize
  TensorVector r;
  for(int64_t i=0; i<n; i+=w) {        // batches of w
   subset(x,0,i,w,n);
   r.emplace_back(q->forward(x));      // accumulate forward calcs
  }
  subset(x,0,0,n,n);                   // restore size of inputs
  y=torch::cat(r);                     // and join batch results
 } else {
  y=q->forward(x);                     // no batching, run forward on full inputs
 }
 if(b) q->train(true);
 return y;
}

static K eval(Kmodel *m,TensorVector& v,int64_t w,Eval e) {
 auto y=evalfwd(m->q,v[0],w);
 auto z=mloss(m,y,v).item<double>();
 switch(e) {
  case Eval::fwd: return knk(2,kf(z),kget(y));
  case Eval::max: return knk(2,kf(z),kget(y.argmax(1)));
  case Eval::pct: return knk(2,kf(z),kf(100.0*torch::sum(v[v.size()-1].eq(y.argmax(1))).item<double>()/y.size(0)));
  default: AT_ERROR("Unrecognized evaluation mode");
 }
}

static Tensor metric(Metric e,Kmodel *m,const TensorVector& v,const Tensor& y) {
 switch(e) {
  case Metric::accuracy:  TORCH_CHECK(v.size()>=2, "accuracy metric: no target found");
                          return 100.0*torch::sum(v[1].eq(y.argmax(-1)))/y.size(0);
  case Metric::loss:      TORCH_CHECK(v.size()>=2, "loss metric: no target found");
                          TORCH_CHECK(m,"loss metric: unable to determine loss function");
                          return mloss(m,y,v);
  case Metric::max:       return torch::argmax(y,-1);
  case Metric::out:       return y;
  default: AT_ERROR("Unrecognized metric");
 }
}

static K keval(K a,Eval e,const char* s) {
 KTRY
  torch::NoGradGuard g;
  Sequential *q=xseq(a,0); Kmodel *m=xmodel(a,0);
  Tensor *x=xten(a,1); TensorVector *v=xvec(a,1); int64_t w=0;
  TORCH_CHECK(q || m, s,": 1st arg is a model or sequential module");
  //TORCH_CHECK(x || v, s,": 2nd arg is a tensor or vector of inputs");
  TORCH_CHECK(a->n==2 || (a->n==3 && xint64(a,2,w)), 
              s, ": unrecognized arg(s), expecting (sequential/model; array/tensor/vector of inputs; optional batch size)");
  TORCH_CHECK(!v || v->size(), s, ": vector of inputs is empty");
  TORCH_CHECK(w>-1, s, ": batch size cannot be negative");
  if(q) {
   //return kresult(v||x, evalfwd(*q, v ? (*v)[0] : (x ? *x : kput(a,1)), w));
   return kresult(v||x, evalfwd(*q, v ? (*v)[0] : *x, w));
  } else {
   return eval(m,*v,w,e);
  }
 KCATCH(s);
}

KAPI Eval(K a) {
 KTRY
  torch::NoGradGuard g;
  Sequential *q=xseq(a,0); Kmodel *m=xmodel(a,0); bool b=false; int64_t w=0;
  TORCH_CHECK(q || m, "evaluate: expects model or sequential module as 1st arg");
  J n=a->n; K z=nullptr;
  if(abs(kK(a)[n-1]->t)==KS) n--, z=kK(a)[n];  // metric symbol(s) given as last arg
  if(n>2 && xbool(a,n-1,b)) n--;               // tensor flag at the end of remaining args
  if(n>2 && xint64(a,n-1,w)) n--;              // batch size at the end of remaining args
  TORCH_CHECK(n>1, "evaluate: expects at least one input as 2nd arg");
  if(false) {
  } else {
   TensorVector v;
   for(J i=1;i<n;++n) {
    Tensor* t=xten(a,i);
    v.emplace_back(t ? *t : kput(a,i));
   }
  }
/*
  Tensor *x=xten(a,1); TensorVector *v=xvec(a,1); int64_t w=0;
  //TORCH_CHECK(x || v, s,": 2nd arg is a tensor or vector of inputs");
  TORCH_CHECK(a->n==2 || (a->n==3 && xint64(a,2,w)), 
              s, ": unrecognized arg(s), expecting (sequential/model; array/tensor/vector of inputs; optional batch size)");
  TORCH_CHECK(!v || v->size(), s, ": vector of inputs is empty");
  TORCH_CHECK(w>-1, s, ": batch size cannot be negative");
  if(q) {
   //return kresult(v||x, evalfwd(*q, v ? (*v)[0] : (x ? *x : kput(a,1)), w));
   return kresult(v||x, evalfwd(*q, v ? (*v)[0] : *x, w));
  } else {
   return eval(m,*v,w,e);
  }
*/
 return (K)0;
 KCATCH("evaluate");
}

KAPI evaluate(K x) {return keval(x, Eval::fwd, "evaluate");}
KAPI evalmax (K x) {return keval(x, Eval::max, "evalmax");}
KAPI evalpct (K x) {return keval(x, Eval::pct, "evalpct");}

// -------------------------------------------------------------------------------------------
// xseq - given tag, returns sequential reference or error
// training - query/set training flag given model or sequential module
// -------------------------------------------------------------------------------------------
Sequential& xseq(Ktag *g) {
 switch(g->a) {
  case Class::sequential: return ((Kseq*)g)->q;
  case Class::model:      return ((Kmodel*)g)->q;
  default: AT_ERROR("Unable to retrieve sequential model from ",mapclass(g->a));
 }
}

KAPI training(K x) {
 KTRY
  bool b; Ktag *g;
  TORCH_CHECK((g=xtag(x)) || ((g=xtag(x,0)) && x->n==2 && xbool(x,1,b)),
              "training: unrecognized arg(s), expects sequential module or model and optional flag");
  auto& q=xseq(g);
  return (x->n==2) ? q->train(b),(K)0 : kb(q->is_training());
 KCATCH("training");
}

// -------------------------------------------------------------------------------------------
// add model api functions to library dictionary
// -------------------------------------------------------------------------------------------
void modelfn(K x) {
 fn(x, "model",    KFN(model),    1);
 fn(x, "train",    KFN(ktrain),   1);
 fn(x, "evaluate", KFN(evaluate), 1);
 fn(x, "evalmax",  KFN(evalmax),  1);
 fn(x, "evalpct",  KFN(evalpct),  1);
}
