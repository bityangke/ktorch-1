#include "ktorch.h"
#include "knn.h"

// -------------------------------------------------------------------------------------------
// modelpart - parse args from k to define sequential, loss & optimizer modules
// modelkeys - return list of symbols used for model state dictionary
// modelstate - return a dictionary with state of sequential module, loss fn & optimizer
// modeltable
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

K mbackward(K x) {
 Kmodel *m; Tensor *input,*label,loss;
 if((m=xmodel(x,0)) && (input=xten(x,1)) && (label=xten(x,2))) {
  loss=m->l.forward(m->q->forward(*input),*label);
 } else {
  AT_ERROR("backward expects (model; inputs; labels)");
 }
 loss.backward();
 return kget(loss);
}

Tensor mloss(Kmodel *m,const Tensor& x,TensorVector &v) {
 if(v.size()==2)
  return losswt(m->lc,m->l,x,v[1]);
 else if(v.size()==3)
  return m->l.forward(x,v[1],v[2]);
 else
  AT_ERROR("model: ", v.size()," inputs, expecting 2-3");
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
static K eval(Kmodel *m,TensorVector& v,int64_t w,int64_t a) {
 torch::NoGradGuard g;
 bool b=m->q->is_training(); Tensor x;
 if(b) m->q->train(false);
 auto n=maxsize(v);
 if(w) {
  TensorVector r;
  for(int64_t i=0; i<n; i+=w) {
   subset(v,0,i,w,n);
   r.emplace_back(mforward(m,v));
  }
  subset(v,0,0,n,n);
  x=torch::cat(r);
 } else {
  x=mforward(m,v);
 }
 if(b) m->q->train(true);
 auto z=mloss(m,x,v).item<double>();
 switch(a) {
  case 1: return knk(2,kf(z),kget(x));
  case 2: return knk(2,kf(z),kget(x.argmax(1)));
  case 3: return knk(2,kf(z),kf(100.0*torch::sum(v[v.size()-1].eq(x.argmax(1))).item<double>()/n));
  default: AT_ERROR("Unrecognized evaluation mode: ",a);
 }
}

static K keval(K x,int64_t a,const char* s) {
 KTRY
  Kmodel *m; TensorVector *v; int64_t w=0;
  TORCH_CHECK((m=xmodel(x,0)) && (v=xvec(x,1)) && (x->n==2 || (x->n==3 && xint64(x,2,w))), 
              s, ": unrecognized arg(s), expecting (model;vector of inputs;optional batch size)");
  TORCH_CHECK(v->size(), s, ": vector of inputs is empty");
  TORCH_CHECK(w>-1, s, ": batch size cannot be negative");
  return eval(m,*v,w,a);
 KCATCH(s);
}

KAPI evaluate(K x) {return keval(x, 1, "evaluate");}
KAPI evalmax (K x) {return keval(x, 2, "evalmax");}
KAPI evalpct (K x) {return keval(x, 3, "evalpct");}

Sequential& xseq(Ktag *g) {
 switch(g->a) {
  case Class::sequential: return ((Kseq*)g)->q;
  case Class::model:      return ((Kmodel*)g)->q;
  default: AT_ERROR("Unable to retrieve sequential model from ",mapclass(g->a));
 }
}

// training - query/set flag for module to perform forward calc as part of training(true) or inference(false)
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
