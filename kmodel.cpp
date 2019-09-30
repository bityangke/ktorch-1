#include "ktorch.h"
#include "knn.h"
#include "kloss.h"

ZV modelpart(K x,J i,Kseq*& q,Kloss*& l,Kopt*& o) {
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
//  subset
// -------------------------------------------------------------------------------------------
V subset(Tensor& t,int64_t d,int64_t i,int64_t n) {
 t.set_(t.storage(), i*t.stride(d), n==t.size(d) ? t.sizes() : newsize(t,d,n), t.strides());
}

V subset(TensorVector& v,int64_t d,int64_t i,int64_t n) {
 for(auto& t:v) subset(t,d,i,n);
}

V setsafe(Tensor& t,const Storage& s,int64_t i,const IntArrayRef& sz,const IntArrayRef& st) {
 TORCH_CHECK(s.size()>=i+computeStorageSize(sz,st), "tensor subset out-of-bounds");
 t.set_(s,i,sz,st);
}

V subset_safe(Tensor& t,int64_t d,int64_t i,int64_t n) {
 setsafe(t, t.storage(), i*t.stride(d), newsize(t,d,n), t.strides());
}

// -------------------------------------------------------------------------------------------
// mforward
// mloss
// -------------------------------------------------------------------------------------------
Tensor mforward(Kmodel *m,TensorVector& v) {
 return m->q->forward(v[0]);
}

Tensor mloss(Kmodel *m,TensorVector &v) {
 auto x=mforward(m,v);
 if(v.size()==2)
  return m->l->forward(x,v[1]);
 else if(v.size()==3)
  return m->l->forward(x,v[1],v[2]);
 else
  AT_ERROR("model: ", v.size()," inputs, expecting 2-3");
}

// -------------------------------------------------------------------------------------------
// batch - 
// fit -
// kbatch -
// kfit -
// -------------------------------------------------------------------------------------------
Tensor batch(Kmodel *m,TensorVector& v,int64_t w,int64_t n) {
 Optimizer *o=nullptr; LossClosureOptimizer *c=nullptr;
 if(m->oc == Cast::lbfgs) c=(LossClosureOptimizer*)m->o.get();
 else                     o=(Optimizer*)m->o.get();

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
  if(w>n-i) w=n-i;                     // final batch may be smaller
  //std::cerr << "subset " << j+1 << ", from row " << i << " using " << w << " row(s)\n";
  subset(v,0,i,w);                     // narrow tensors to current batch
  if(o)
   p[j]=loss().item<float>(), o->step(); // single loss evaluation
  else
   p[j]=c->step(loss).item<float>();     // pass closure, e.g. LBFGS
 }
 subset(v,0,0,n);                      // reset tensors to full length
 return r;                             // return losses
}

Tensor fit(Kmodel *m,TensorVector& v,int64_t w,int64_t e,B s) {
 TensorVector r;
 auto n=fullsize(v);
 for(size_t i=0; i<e; ++i) {
  if(s) shuffle(v);
  r.emplace_back(batch(m,v,w,n));
 }
 return torch::stack(r); //.reshape({e,-1});
}

KAPI kbatch(K x) {
 KTRY
  Kmodel *m; TensorVector *v; int64_t w;
  if((m=xmodel(x,0)) && (v=xvec(x,1)) && xint64(x,2,w) && x->n==3) {
   TORCH_CHECK(v->size(), "model: vector of inputs is empty");
   return kget(batch(m,*v,w,maxsize(*v)));
  } else {
   return KERR("batch: unrecognized arg(s)");
  }
 KCATCH("batch");
}

KAPI kfit(K x) {
 KTRY
  Kmodel *m; TensorVector *v; B s=true; int64_t w,e;
  if((m=xmodel(x,0)) && (v=xvec(x,1)) && xint64(x,2,w) && xint64(x,3,e) &&
      (x->n==4 || (x->n==5 && xbool(x,4,s)))) {
   TORCH_CHECK(v->size(), "model: vector of inputs is empty");
   return kget(fit(m,*v,w,e,s));
  } else {
   return KERR("fit: unrecognized arg(s)");
  }
 KCATCH("fit");
}

