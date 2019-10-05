#include "ktorch.h"
#include "knn.h"
#include "kloss.h"

static void modelpart(K x,J i,Kseq*& q,Kloss*& l,Kopt*& o) {
 for(;i<x->n;++i) {
  auto* g=xtag(x,i);
  switch(g ? g->a : Class::undefined) {
   case Class::sequential: q=(Kseq*)g;  break;
   case Class::loss:       l=(Kloss*)g; break;
   case Class::optimizer:  o=(Kopt*)g;  break;
   default: AT_ERROR("model arg[",i,"] unrecognized: ",
                    (g ? mapclass(g->a) : kname(kK(x)[i]->t))); break;
  }
 }
}

K modelstate(B a,B b,Kmodel *m) {
  std::cerr << "model state..\n";
 return (K)0;
}

KAPI model(K x) {
 KTRY
  B a=env().alloptions;
  Kseq *q=nullptr; Kloss *l=nullptr; Kopt *o=nullptr; Kmodel *m=nullptr;
  TORCH_CHECK(!x->t, "model not implemented for ",kname(x->t));
  if((m=xmodel(x)) || (x->n==2 && xbool(x,1,a) && (m=xmodel(x,0)))) {
   return modelstate(a,false,m);
  } else {
   m=xmodel(x,0); modelpart(x,m ? 1 : 0,q,l,o);
   if(m) {
    if(q) m->q=q->q;              // reassign model's sequential module
    if(l) m->lc=l->c, m->l=l->l;  // loss function
    if(o) m->oc=o->c, m->o=o->o;  // optimizer
    return (K)0;
   } else {
    TORCH_CHECK(q && l && o, (q ? (l ? "optimizer" : "loss") : "sequential module")," not specified");
    return kptr(new Kmodel(q,l,o));
   }
  }
 KCATCH("model");
}

// -------------------------------------------------------------------------------------------
// mforward
// mloss
// -------------------------------------------------------------------------------------------
Tensor mforward(Kmodel *m,TensorVector& v) {
 return m->q->forward(v[0]);
}

Tensor mloss(Kmodel *m,TensorVector &v,const Tensor& x) {
/*
 std::cerr<< x[0] << "\n";
 std::cerr<< x[63] << "\n";
 TORCH_CHECK(false,"stop");
*/
 if(v.size()==2)
  return m->l->forward(x,v[1]);
 else if(v.size()==3)
  return m->l->forward(x,v[1],v[2]);
 else
  AT_ERROR("model: ", v.size()," inputs, expecting 2-3");
}

Tensor mloss(Kmodel *m,TensorVector &v) { return mloss(m,v,mforward(m,v));}

// -------------------------------------------------------------------------------------------
// batch - run model's forward calc, loss, backward calcs and optimizer step in batches
// train - train model for given batch size and number of passes through the data ("epochs")
// ktrain - k api fn, expects (model;vector;batch size; optional epochs;optional shuffle flag)
// -------------------------------------------------------------------------------------------
Tensor batch(Kmodel *m,TensorVector& v,int64_t w,int64_t n=0);
Tensor batch(Kmodel *m,TensorVector& v,int64_t w,int64_t n) {
 Optimizer *o=nullptr; LossClosureOptimizer *c=nullptr;
 if(m->oc == Cast::lbfgs) c=(LossClosureOptimizer*)m->o.get();
 else                     o=(Optimizer*)m->o.get();

 if(!n) n=maxsize(v);
 if(w>n) w=n;                          // reduce batch size if exceeds total size
 auto s=n%w ? n/w + 1 : n/w;           // no. of subsets to process
 auto r=torch::empty(s);               // tensor for batch losses
 auto* p=r.data<float>();              // ptr for quicker assignment

 auto loss=[&]() {                     // define closure for
  m->o->zero_grad();                   // resetting gradients
  auto r=mloss(m,v);                   // calculating model output & loss
  r.backward();                        // calculating gradients
  return r;                            // return loss tensor
 };

 for(int64_t i=0,j=0; i<n; i+=w,++j) {
  //std::cerr << "subset " << j+1 << ", from row " << i << " using " << w << " row(s)\n";
  subset(v,0,i,w,n);                   // narrow tensors to current batch
  if(o)
   p[j]=loss().item<float>(), o->step(); // single loss evaluation
  else
   p[j]=c->step(loss).item<float>();     // pass closure, e.g. LBFGS
 }
 subset(v,0,0,n,n);                    // reset tensors to full length
 return r;                             // return losses
}

Tensor train(Kmodel *m,TensorVector& v,int64_t w,int64_t e,B s) {
 auto n=fullsize(v);
 if(e) {
  TensorVector r;
  for(size_t i=0; i<e; ++i) {
   if(s) shuffle_(v);
   r.emplace_back(batch(m,v,w,n));
  }
  return torch::stack(r);
 } else {
  if(s) shuffle_(v);
  return batch(m,v,w,n);
 }
}

KAPI ktrain(K x) {
 KTRY
  Kmodel *m; TensorVector *v; B s=true; int64_t w,e=0;
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
Tensor evaluate(Kmodel *m,TensorVector& v,int64_t w) {
 torch::NoGradGuard g;
 B b=m->q->is_training();
 if(b) m->q->train(false);
 auto n=maxsize(v); TensorVector r;
 //auto s=n%w ? n/w + 1 : n/w;
 for(int64_t i=0; i<n; i+=w) {
  subset(v,0,i,w,n);
  r.emplace_back(mforward(m,v));
 }
 subset(v,0,0,n,n);
 if(b) m->q->train(true);
 return torch::cat(r);
}


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
  B b; Ktag *g;
  TORCH_CHECK((g=xtag(x)) || ((g=xtag(x,0)) && x->n==2 && xbool(x,1,b)),
              "training: unrecognized arg(s), expects sequential module or model pointer and optional flag");
  auto& q=xseq(g);
  std::cerr << "seq ref count: " << q.ptr().use_count() << "\n";
  return (x->n==2) ? q->train(b),(K)0 : kb(q->is_training());
 KCATCH("training");
}

KAPI kevaluate(K x) {
 KTRY
  Kmodel *m; TensorVector *v; int64_t w;
  if((m=xmodel(x,0)) && (v=xvec(x,1)) && xint64(x,2,w)) {
   TORCH_CHECK(v->size(), "evaluate: vector of inputs is empty");
   return kget(evaluate(m,*v,w));
  } else {
   return KERR("evaluate: unrecognized arg(s)");
  }
 KCATCH("evaluate");
}

// -------------------------------------------------------------------------------------------
// add model api functions to library dictionary
// -------------------------------------------------------------------------------------------
void modelfn(K x) {
 fn(x, "model",  KFN(model),1);
 fn(x, "train",  KFN(ktrain),1);
}
