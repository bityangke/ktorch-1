#include "ktorch.h"
#include <torch/csrc/autograd/function.h>

// -------------------------------------------------------------------------
// kten - given tensor, return ptr to struct w'attrs, void ptr to tensor
// kvec - given ptr to vector of tensors, return ptr to struct w'attrs
// -------------------------------------------------------------------------
K kten(const Tensor& t) {return kptr(new Kten(t));}
K kvec(const TensorVector& v) {return kptr(new Kvec(v));}

// -------------------------------------------------------------------------
// kgetscalar - return k scalar given a scalar tensor
// kgets - process tensor at depth, creating k array
// kget - take tensor reference, return k scalar/array
//      - take reference to vector of longs, return k list
//      - take reference to vector of tensors, return k lists
// -------------------------------------------------------------------------
K kgetscalar(const Tensor &t){
 auto s=t.item();
 switch(t.scalar_type()) {
  case torch::kFloat:  return ke(s.toFloat());
  case torch::kDouble: return kf(s.toDouble());
  case torch::kHalf:   return ke(s.toFloat());
  case torch::kShort:  return kh(s.toShort());
  case torch::kInt:    return ki(s.toInt());
  case torch::kLong:   return kj(s.toLong());
  case torch::kBool:   return kb(s.toBool());
  case torch::kByte:   return kg(s.toByte());
  case torch::kChar:   return kc(s.toChar());
  default: AT_ERROR("Unrecognized scalar tensor type: ", t.dtype(), ", cannot return k scalar"); return (K)0;
 }
}

ZK kgets(I i,I j,A k,J b,const int64_t *s,S &p) { //i:depth, j:max depth, k:k type, b:bytes to copy, s:sizes, p:data ptr
 K x=ktn((i<j) ? 0 : k,s[i]);                     //create k list
 if(x->t) {                                       //if base type
  if(x->n) {                                      //   and non-zero length
   memcpy(kG(x),p,b);                             //copy k <- tensor
   p+=b;                                          // and incr data ptr
  }
 } else {                                         // else
   for(J y=0;y<x->n;++y)                          // call down a level
    kK(x)[y]=kgets(i+1,j,k,b,s,p);                // until base data type
 }
 return x;
}

K kget(const Tensor &t) {
 if(!t.defined())
  return ktn(0,0);
 else if (!t.dim())      // if 0-dimensional
  return kgetscalar(t);  // return scalar
 Tensor c;
 if(t.dtype()==torch::kHalf)
  c=t.toType(torch::kFloat).contiguous().toBackend(torch::Backend::CPU);
 else
  c=t.contiguous().toBackend(torch::Backend::CPU);
 I j=c.dim()-1; const int64_t *s=c.sizes().data();  // dimension & sizes at each dim
 J b=s[j]*c.element_size();                   // bytes to copy at lowest depth
 S p=(S)c.data_ptr();                         // contiguous data pointer
 return kgets(0,j,maptype(t.dtype()),b,s,p);
}

K kget(const LongVector& v) {return klist(v.size(),v.data());}

K kget(const TensorVector& v) {
 K x=ktn(0,v.size());
 for(size_t i=0; i<v.size(); ++i) kK(x)[i]=kget(v[i]);
 return x;
}

K kget(const TensorDeque& v) {
 K x=ktn(0,v.size());
 for(size_t i=0; i<v.size(); ++i) kK(x)[i]=kget(v[i]);
 return x;
}

// -------------------------------------------------------------------------
// tento - change tensor device/type, return new tensor if copy flag set
// ktenpair - given a pair of tensors return pair of pointers or array
// kten3 - given a triplet of tensors return triplet of pointers or array
// -------------------------------------------------------------------------
K tento(Kten* t,const TensorOptions& o,B a,B b) {
 auto r=t->t.to(o,a,b);
 if(b)                 // if copy flag set
  return kten(r);      // return new tensor
 if(!t->t.is_same(r))  // else if device/dtype caused new tensor
  t->t=r;              // replace tensor in k ptr
 return (K)0;
}

K vecto(Kvec* v,const TensorOptions& o,B a) {
 for(auto& t:v->v) {
  auto r=t.to(o,a);
  if(!t.is_same(r)) t=std::move(r);
 }
 return (K)0;
}

K ktenpair(B p,Tensor& a,Tensor& b) {  // p:true if returning tensor pointers
 if(p) return knk(2,kten(a),kten(b));
 else  return knk(2,kget(a),kget(b));
}

K kten3(B p,Tensor& a,Tensor& b,Tensor& c) {  // p:true if returning tensor pointers
 if(p) return knk(3,kten(a),kten(b),kten(c));
 else  return knk(3,kget(a),kget(b),kget(c));
}

// ---------------------------------------------------------------------------------------
// kputscalar - copy single k value to CPU tensor scalar
// kdepth - check k array at depth for consistent datatype, size, etc, throw errors
// kputs - descend depth of k array, determining dim & sizes, copying data types to tensor
// kput - controlling function to read k array, create tensor and copy data at depth
// ---------------------------------------------------------------------------------------
V kputscalar(K x,Tensor &t) {
 Scalar s;
 if(xscalar(x,s))
  t=torch::full({},s).to(maptype(x->t));
 else
  AT_ERROR("Unable to translate k ",kname(x->t)," to scalar tensor");
}

ZV kdepth(K x,I i,A k,Ksize &s){
 if(x->t < 0) {
  AT_ERROR("Unable to map mixed array to tensor: ",kname(x->t)," encountered at depth ",i);
// } else if(i && x->n==0) {
//   AT_ERROR("No empty dimension at lower depth");
 } else if(k) {                   // if base type already encountered
  I j=s.size()-1;                 // last size index
  if(x->n != s[i]) {              // check that dimensions are consistent
   AT_ERROR("Dimension mismatch at depth ",i,", ",s[i]," vs ",x->n);
  } else if(x->t != (i<j ? 0 : k)) {  // check for same data type at same depth
   AT_ERROR("Type mismatch at depth ",i,", ",k,kname(k)," vs ",kname(x->t));
  }
 } else {
  s.push_back(x->n);              // no error, no base type yet, accumulate sizes
 }
}

ZV kputs(K x,I i,A &k,Ksize &s,J &b,S &p,Tensor &t) {
 kdepth(x,i,k,s);
 if(x->t) {  // if base data type
  if(!k)  {  // if first encounter w'base data type
   k=x->t;
   t=torch::empty(s, torch::device(torch::kCPU).dtype(maptype(k)).layout(torch::kStrided));
   b=t.element_size() * s[i];  // bytes to copy
   p=(S)t.data_ptr();          // contiguous data pointer
  }
  memcpy(p,kG(x),b); p+=b;
 } else {
  for(I j=0;j<x->n;++j) kputs(kK(x)[j],i+1,k,s,b,p,t);
 }
}

Tensor kput(K x) {        
 A k=0;                    // fill w'base data type for nested k value
 J b=0;                    // fill w'bytes to copy
 Ksize s;                  // fill w'k array size at each depth
 S p=nullptr;              // data pointer for created tensor
 Tensor t;                 // undefined tensor
 if(x->t < 0)              // if scalar
  kputscalar(x,t);         // create scalar backed by tensor
 else                      // else go through the depth of the array
  kputs(x,0,k,s,b,p,t);    // until base data type encountered
 return t;
}

Tensor kput(K x,J i) {
 if(xind(x,i)) 
  return kput(kK(x)[i]);
 else
  AT_ERROR("Unable to index ",kname(x->t),", element: ",i);
}
// --------------------------------------------------------------------------------------
// tensorlike - tensor creation routines, e.g. ones_like() where tensor given as template
// tensorout - tensor creation reouties, e.g. ones_out(), where output tensor is given
// tensoropt - tensor creation routines where tensor size and option(s) given
// tensormode - determines whether a template tensor or output tensor given w'other args
// tensorput - put k value(s) -> tensor, return new tensor ptr unless output tensor given
// tensor - high level function to create/retrieve/move/recast tensor from k
// --------------------------------------------------------------------------------------
ZV tensorlike(K x,Tensormode m,Tensor &t,Tensor &r) {  // t:input, r:result tensor
 //use tensor options from input tensor, override if any supplied in final arg
 using Tensormode=Tensormode; J i,j; Scalar s; TensorOptions o=t.options();
 B b=xopt(x,x->n-1,o); I nx=x->n-b;  //set flag if options given, count non-option args
 switch(m) {
  case Tensormode::empty: if(nx==2) r=b ? torch::empty_like(t,o) : torch::empty_like(t); break;
  case Tensormode::zeros: if(nx==2) r=b ? torch::zeros_like(t,o) : torch::zeros_like(t); break;
  case Tensormode::ones:  if(nx==2) r=b ? torch::ones_like(t,o)  : torch::ones_like(t);  break;
  case Tensormode::rand:  if(nx==2) r=b ? torch::rand_like(t,o)  : torch::rand_like(t);  break;
  case Tensormode::randn: if(nx==2) r=b ? torch::randn_like(t,o) : torch::randn_like(t); break;
  case Tensormode::full:  if(nx==3 && xnum(x,1,s))r=b ? torch::full_like(t,s,o) : torch::full_like(t,s); break;
  case Tensormode::randint:
   if     (nx==3 && xlong(x,2,j))                 r=b ? torch::randint_like(t,j,o)   : torch::randint_like(t,j);
   else if(nx==4 && xlong(x,2,i) && xlong(x,3,j)) r=b ? torch::randint_like(t,i,j,o) : torch::randint_like(t,i,j);
   break;
  default:
   AT_ERROR("Tensor creation via: ",x->s," not implemented with input tensor"); break;
 }
}

ZV tensorout(K x,Tensormode m,Tensor &t,Tensor &r) {  // t:output, r:result tensor
 F e; J i,j; Scalar a,z,n; JRef s;
 B b=xsize(x,1,s);  //true if size is given as 2nd arg (last arg is output tensor)
 switch(m) {
  case Tensormode::empty: if(b && x->n==3) r=torch::empty_out(t,s); break;
  case Tensormode::zeros: if(b && x->n==3) r=torch::zeros_out(t,s); break;
  case Tensormode::ones:  if(b && x->n==3) r=torch::ones_out(t,s); break;
  case Tensormode::rand:  if(b && x->n==3) r=torch::rand_out(t,s); break;
  case Tensormode::randn: if(b && x->n==3) r=torch::randn_out(t,s); break;
  case Tensormode::full:  if(b && x->n==4 && xnum(x,2,n)) r=torch::full_out(t,s,n); break;
  case Tensormode::randperm: if (x->n==3 && xlong(x,1,i)) r=torch::randperm_out(t,i); break;
  case Tensormode::randint:
   b=xsize(x,x->n-2,s);
   if     (b && x->n==4 && xlong(x,1,j))                 r=torch::randint_out(t,j,s);
   else if(b && x->n==5 && xlong(x,1,i) && xlong(x,2,j)) r=torch::randint_out(t,i,j,s);
   break;
  case Tensormode::eye:
    if     (x->n==3 && xlong(x,1,i))                 r=torch::eye_out(t,i);
    else if(x->n==4 && xlong(x,1,i) && xlong(x,2,j)) r=torch::eye_out(t,i,j);
    break;
  case Tensormode::range:
  case Tensormode::arange:
   b=m==Tensormode::range;
   if     (x->n==3 && xnum(x,1,z))                              r = b ? torch::range_out(t,0,z)   : torch::arange_out(t,z);
   else if(x->n==4 && xnum(x,1,a) && xnum(x,2,z))               r = b ? torch::range_out(t,a,z)   : torch::arange_out(t,a,z);
   else if(x->n==5 && xnum(x,1,a) && xnum(x,2,z) && xnum(x,3,n))r = b ? torch::range_out(t,a,z,n) : torch::arange_out(t,a,z,n);
   break;
  case Tensormode::linspace:
  case Tensormode::logspace:
   b=m==Tensormode::logspace; i=100; e=10.0; //default of 100 steps, base 10
   if(xnum(x,1,a) && xnum(x,2,z) && (x->n==4 || (xlong(x,3,i) && (x->n==5 || (x->n==6 && b && xnum(x,4,e))))))
    r = b ? torch::logspace_out(t,a,z,i,e) : torch::linspace_out(t,a,z,i);
   break;
  default: break;
 }
}

ZV tensoropt(K x,Tensormode m,Tensor &r) {
 F e; J i,j; Scalar a,z,n; JRef s; TensorOptions o;
 B b=xopt(x,x->n-1,o); I nx=x->n-b;                        //track if options in last arg
 B sz=xsize(x,1,s) && nx==((m==Tensormode::full) ? 3 : 2); //2nd arg is size & correct arg count
 switch(m) {
  case Tensormode::empty: if(sz) r=torch::empty(s,o); break;
  case Tensormode::zeros: if(sz) r=torch::zeros(s,o); break;
  case Tensormode::ones:  if(sz) r=torch::ones(s,o); break;
  case Tensormode::rand:  if(sz) r=torch::rand(s,o); break;
  case Tensormode::randn: if(sz) r=torch::randn(s,o); break;
  case Tensormode::full:  if(sz && xnum(x,2,n)) r=torch::full(s,n,o); break;
  case Tensormode::randperm:
   if (!o.has_dtype()) o=o.dtype(torch::kLong);
   if (nx==2 && xlong(x,1,i)) r = torch::randperm(i,o);
   break;
  case Tensormode::randint:
   sz=xsize(x,nx-1,s); // true if size is supplied as last non-options arg
   if     (sz && nx==3 && xlong(x,1,j))                 r=torch::randint(j,s,o);
   else if(sz && nx==4 && xlong(x,1,i) && xlong(x,2,j)) r=torch::randint(i,j,s,o);
   break;
  case Tensormode::eye:
    if     (xn==2 && xlong(x,1,i))                 r=torch::eye(i,o);
    else if(xn==3 && xlong(x,1,i) && xlong(x,2,j)) r=torch::eye(i,j,o);
    break;
  case Tensormode::range:
   if     (nx==3 && xnum(x,1,a) && xnum(x,2,z))               r=torch::range(a,z,o);
   else if(nx==4 && xnum(x,1,a) && xnum(x,2,z) && xnum(x,3,n))r=torch::range(a,z,n,o);
   break;
  case Tensormode::arange:
   b=!o.has_dtype();
   if(nx==2 && xnum(x,1,z)) {
    if(b && z.isIntegral()) o=o.dtype(torch::kLong);
    r=torch::arange(z,o);
   } else if(nx==3 && xnum(x,1,a) && xnum(x,2,z)) {
    if(b && a.isIntegral() && z.isIntegral()) o=o.dtype(torch::kLong);
    r=torch::arange(a,z,o);
   } else if(nx==4 && xnum(x,1,a) && xnum(x,2,z) && xnum(x,3,n)) {
    if(b && a.isIntegral() && z.isIntegral() && n.isIntegral()) o=o.dtype(torch::kLong);
    r=torch::arange(a,z,n,o);
   }
   break;
  case Tensormode::linspace:
  case Tensormode::logspace:
   b=m==Tensormode::logspace; i=100; e=10.0; //default of 100 steps, base 10
   if(xnum(x,1,a) && xnum(x,2,z) && (nx==3 || (xlong(x,3,i) && (nx==4 || (nx==5 && b && xnum(x,4,e))))))
    r = b ? torch::logspace(a,z,i,e,o) : torch::linspace(a,z,i,o);
   break;
  default: break;
 }
}

ZK tensormode(K x,S s,Tensormode m) {
 Tensor t,r; B in=false,out=false;
 if((in=xten(x,1,t)))            tensorlike(x,m,t,r); // input tensor is 2nd arg
 else if((out=xten(x,x->n-1,t))) tensorout(x,m,t,r);  // output tensor is final arg
 else                            tensoropt(x,m,r);    // no input/output tensor
 if(!r.defined())
  AT_ERROR("Unrecognized argument(s) for tensor creation mode: ",s);
 return out ? (K)0 : kten(r);
}

ZK tensorput(K x) {
 Tensor r,t; TensorOptions o;
 if(xempty(x))
  t=torch::empty({0});
 else if((xopt(x,1,o) || xten(x,1,r)) && x->n==2)
  t=xempty(x,0) ? torch::empty({0},o) : kput(x,0);
 else
  t=kput(x);
 if(r.defined()) {
  r.resize_(t.sizes()).copy_(t,true);
  return (K)0;
 } else {
  if(!o.has_dtype()) 
   o=o.dtype(t.dtype()); //if no explicit data type given, use k type
  t=t.to(o.device(),o.dtype()).set_requires_grad(o.requires_grad());
  return kten(t);
 }
}

KAPI tensor(K x) {
 S s; Tensormode m; Tensor t;
 KTRY
  if(xten(x,t))
   return kget(t);
  else if(xmode(x,0,s,m))
   return tensormode(x,s,m);
  else
   return tensorput(x);
 KCATCH("Unable to complete tensor operation");
}

// ------------------------------------------------------------------------------------------------
// diagnostic functions -- check underlying pointers, storage data, reference counts, etc.
// ------------------------------------------------------------------------------------------------
// grad = return gradient data or empty, if ptr enlisted, return gradient ptr, which must be free'd
// graphdetail - return information related to autograd graph: leaf variable flag & backward fn
// tensordata - return CPU tensor/storage data as k list
// tensordetail - return dictionary of attributes given tensor and detail level 0,1,2
// ------------------------------------------------------------------------------------------------
KAPI grad(K x) {
 KTRY
  B p=false; Tensor t;
  if(xten(x,t) || (p=(xten(x,0,t) && x->n==1))) {
   if(p) return t.grad().defined() ? kten(t.grad()) : KERR("No gradient defined");
   else  return t.grad().defined() ? kget(t.grad()) : (K)0;
 } else {
  return KERR("Unexpected arg(s) for gradient, expectining tensor (enlist to return gradient ptr)");
 }
 KCATCH("Unable to get gradient");
}

ZV graphdetail(K *a,K *b,const Tensor& t) {
 if(t.is_variable()) {
  auto& v=torch::autograd::as_variable_ref(t);
  cS s=v.grad_fn() ? v.grad_fn()->name().c_str() : "";
  js(a,cs("leaf"));   jk(b,kb(v.is_leaf()));
  js(a,cs("gradfn")); jk(b,ks((S)s));
 } else {
  js(a,cs("leaf"));   jk(b,kb(false));
  js(a,cs("gradfn")); jk(b,ks((S)""));
 }
}

ZK tensordata(B b,Tensor &t) {  //tensor flag: true-use tensor, false-use storage
 J e,n; V *v;
 if (t.is_cuda())
  return KERR("CUDA tensor not supported -- no memcpy on CUDA data");
 if(b) {
  e=t.storage().elementSize(); n=t.numel(); v=t.data_ptr();
 } else {
   auto s=t.storage(); e=s.elementSize(); n=s.size(); v=s.data();
 }
 K x=ktn(maptype(t.dtype()),n);
 memcpy(kG(x),v,n*e);
 return x;
}

K tensordetail(const Tensor& t,I y) {
 B s=t.is_sparse(); J n=t.dim();
 K x=xD(ktn(KS,0),ktn(0,0)),*a=&kK(x)[0],*b=&kK(x)[1];
 js(a,cs("device"));   jk(b,ks(optsym(t.device())));
 js(a,cs("dtype"));    jk(b,ks(optsym(t.dtype())));
 js(a,cs("layout"));   jk(b,ks(optsym(t.layout())));
 js(a,cs("gradient")); jk(b,ks(optsym(t.is_variable() ? t.requires_grad() : false)));
 graphdetail(a,b,t);
 js(a,cs("ktype"));    jk(b,kh(maptype(t.dtype())));
 js(a,cs("dim"));      jk(b,kj(n));
 js(a,cs("size"));     jk(b,ktn(KJ,n)); memcpy(kG(kK(*b)[(*b)->n-1]),t.sizes().data(),n*sizeof(J));
 js(a,cs("stride"));   jk(b,ktn(KJ,s ? 0 : n));
                       if(!s) memcpy(kG(kK(*b)[(*b)->n-1]),t.strides().data(),n*sizeof(J));
 if(y<1) return x;
 js(a,cs("contiguous"));  jk(b,s ? 0 : kb(t.is_contiguous()));
 js(a,cs("distributed")); jk(b,kb(t.is_distributed()));
 js(a,cs("tensorptr"));   jk(b,kj((intptr_t)t.unsafeGetTensorImpl()));
 js(a,cs("offset"));      jk(b,kj(t.storage_offset()));
 js(a,cs("dataptr"));     jk(b,kj((intptr_t)t.data_ptr()));
 js(a,cs("datasize"));    jk(b,kj(t.numel()));
 js(a,cs("storageptr"));  jk(b,kj((intptr_t)t.storage().data()));
 js(a,cs("storagesize")); jk(b,kj(t.storage().size()));
 js(a,cs("elementsize")); jk(b,kj(t.storage().elementSize()));
 js(a,cs("ref"));         jk(b,kj(t.use_count()));
 js(a,cs("weakref"));     jk(b,kj(t.weak_use_count()));
 js(a,cs("storageref"));  jk(b,kj(t.storage().use_count()));
 if(y<2) return x;
 auto c=t.toBackend(torch::Backend::CPU);
 js(a,cs("storagedata")); jk(b,tensordata(false,c));
 js(a,cs("data"));        jk(b,tensordata(true, c));
 return x;
}

// ------------------------------------------------------------------------------------------
// tensor vector fns: 
// ------------------------------------------------------------------------------------------
// vecinit - initialize vector of tensors from k array, tensor ptr(s) or some mix of both
// vector -
// ------------------------------------------------------------------------------------------
K vecinit(K x) {
 TensorVector v;
 if(x->t) {
  Tensor t=kput(x);
  if(t.dim())
   for(int64_t i=0;i<t.size(0);++i)
    v.emplace_back(t[i]);
  else
   v.emplace_back(t);
 } else if(auto *t=xten(x)) {
  v.emplace_back(*t);
 } else {
  for(J i=0;i<x->n;++i) {
   Tensor t;
   v.emplace_back(xten(x,i,t) ? std::move(t) : kput(kK(x)[i]));
  }
 }
 return kvec(v);
}

KAPI vector(K x) {
 KTRY
  J i; Tensor t;
  if(auto* v=xvec(x)) {
   return kget(*v);
  } else if(auto* v=xvec(x,0)) {
   if(xlong(x,1,i)) {
    if(x->n==2)
     return kget(v->at(i));
    else if(x->n==3)
     return (v->at(i)=xten(x,2,t) ? std::move(t) : kput(x,2)), (K)0;
   }
   //} else if(auto *w=xvec(x,1) && x->n==2)
   // v.insert(v.end(), w.begin(), w.end()), (K)0;
   AT_ERROR("Vector given with unrecognized arg(s)");
  } else {
   return vecinit(x);
  }
 KCATCH("vector");
}

// ----------------------------------------------------------------------------------------------
// dim - return no. of tensor dimensions, or dimensions of each tensor in vector of tensors
// size - return sizes of each tensor dimension, or list of sizes for vector
// stride - return strides of each tensor dimension, or list of strides for vector
// device - return device(s) for given tensor(s)
// options - return options for tensor(s)
// ----------------------------------------------------------------------------------------------
KAPI dim(K x) {
 KTRY
  Tensor t;
  if(xten(x,t)) {
   return kj(t.dim());
  } else if(auto* v=xvec(x)) {
   K d=ktn(KJ,v->size());
   for(size_t i=0; i<v->size(); ++i)
    kJ(d)[i]=v->at(i).dim();
   return d;
  } else {
   return kj(kput(x).dim());
  }
 KCATCH("dim");
}

ZK size1(const Tensor& t,B b) {
 return klist(t.dim(),b ? t.sizes().data() : t.strides().data());
}

ZK size2(K x,B b) {
 KTRY
  Tensor t;
  if(xten(x,t)) {
   return size1(t,b);
  } else if(auto* v=xvec(x)) {
   K s=ktn(0,v->size());
   for(size_t i=0; i<v->size(); ++i)
    kK(s)[i]=size1(v->at(i),b);
   return s;
  } else {
   return size1(kput(x),b);
  }
 KCATCH("size/stride");
}

KAPI size  (K x) {return size2(x,true);}
KAPI stride(K x) {return size2(x,false);}

ZS tensym(const Tensor& t,I m) {
 switch(m) {
  case 0: return optsym(t.device());
  case 1: return optsym(t.dtype());
  case 2: return optsym(t.layout());
  case 3: return optsym(t.requires_grad());
  default: AT_ERROR("Invalid mode for tensor setting: ",m);
 }
}

K tensym(K x,I m,cS e) {
 KTRY
  Tensor t;
  if(xten(x,t)) {
   return ks(tensym(t,m));
  } else if(auto* v=xvec(x)) {
   K y=ktn(KS,v->size());
   for(size_t i=0; i<v->size(); ++i)
    kS(y)[i]=tensym(v->at(i),m);
   return y;
  } else {
   AT_ERROR("Unrecognized arg(s) for ", e, ", expected tensor(s)");
  }
 KCATCH(e);
}

KAPI device(K x) {return tensym(x,0,"device");}
KAPI dtype(K x)  {return tensym(x,1,"dtype");}
KAPI layout(K x) {return tensym(x,2,"layout");}
//KAPI layout(K x) {return tensym(x,2,"layout");}

KAPI options(K x) {
 KTRY
  Tensor t;
  if(xten(x,t)) {
   return optmap(t.options());
  } else if(auto* v=xvec(x)) {
   K y=ktn(0,4);
   for(size_t i=0; i<4; ++i) 
    kK(y)[i]=ktn(KS,v->size());
   for(size_t i=0; i<v->size(); ++i)
    optval(v->at(i).options(),y,i);
   return xT(xD(optkey(),y));
  } else {
   AT_ERROR("Unrecognized arg(s) for options, expected tensor(s)");
  }
 KCATCH("options");
}

// --------------------------------------------------------------------------------------------
// perm - return permutation indices given tensor and dimension
// vperm - check dimension size & device for each tensor in vector, return permutation indices
// shuffle - shuffle tensor or vector of same size along given dimension
// shuffle_ - in-place version of tensor/vector shuffle
// kshuffle,1,2 - k api functions, expects tensor/vector input or (input;dim;inplace flag)
// --------------------------------------------------------------------------------------------
Z Tensor perm(const Tensor& t,int64_t d) {
 return torch::randperm(t.size(d),torch::dtype(torch::kLong).device(t.device()));
}

Z Tensor vperm(const TensorVector& v,int64_t d) {
 size_t i=0; Tensor p;
 for(auto& t:v) {
  if(!p.defined())
   p=perm(t,d);
  else if(t.size(d) != p.size(0))
   AT_ERROR("Size mismatch: tensor[", i, "], dim[", d, "]=",t.size(d), ", but permutation size=", p.size(0));
  else if (t.device() != p.device())
   AT_ERROR("Device mismatch: tensor[", i, "] is on ", t.device(), ", but permutation indices are on ", p.device());
  ++i;
 }
 return p;
}

Z Tensor shuffle(const Tensor &t,int64_t d) {return t.index_select(d,perm(t,d));}
ZV shuffle_(Tensor &t,int64_t d) {t=shuffle(t,d);}

Z TensorVector shuffle(const TensorVector& v,int64_t d) {
 auto p=vperm(v,d); TensorVector r;
 for(auto& t:v) r.emplace_back(t.index_select(d,p));
 return r;
}
 
ZV shuffle_(TensorVector& v,int64_t d) {
 auto p=vperm(v,d);
 for(auto& t:v) t=t.index_select(d,p);
}

ZK kshuffle1(Tensor &t,int64_t d,B b) {return b ? shuffle_(t,d),(K)0 : kten(shuffle(t,d));}
ZK kshuffle2(TensorVector& v,int64_t d,B b) {return b ? shuffle_(v,d),(K)0 : kvec(shuffle(v,d));
}

KAPI kshuffle(K x) {
 KTRY
  B b=true; int64_t d=0; Ktag *g; //default is in-place, along 1st dim
  if((g=xtag(x)) || ((g=xtag(x,0)) && (x->n==2 && xint64(x,1,d)))) {
   switch(g->a) {
    case Class::tensor: return kshuffle1(((Kten*)g)->t,d,b);
    case Class::vector: return kshuffle2(((Kvec*)g)->v,d,b);
    default: AT_ERROR("shuffle not implemented for ",mapclass(g->a));
   }
  } else {
   AT_ERROR("unrecognized arg(s) for shuffle");
  }
 KCATCH("shuffle");
}

// ------------------------------------------------------------------------------------------
// tensor utiity fns: 
// ------------------------------------------------------------------------------------------
// tensorcopy - tgt <- src values, must have same type & device, tgt resized if src larger
// backward: backprop given tensor, optional tensor and sym for retain/create gradient graph
// ------------------------------------------------------------------------------------------
V tensorcopy(Tensor &tgt,const Tensor &src,B async) {
 if(src.dtype() != tgt.dtype()) {
  AT_ERROR("Unable to copy values from ",src.dtype()," tensor to ",tgt.dtype()," tensor");
 } else if(src.device() != tgt.device()) {
  AT_ERROR("Unable to copy values across devices, from ",src.device()," to ",tgt.device());
 } else {
  tgt.resize_as_(src).copy_(src,async);
 }
}

KAPI backward(K x) {
 KTRY
  Tensor t; B ok=false;
  if(xten(x,t)) {
   t.backward(); ok=true;
  } else if(xten(x,0,t)) {
   B a=false,b=false; Tensor g; J n=x->n - xbacksym(x,x->n-1,a,b);
   if(n==1) {
     //PATCH t.backward(torch::nullopt,a,b); ok=true;
     t.backward({},a,b); ok=true;
   } else if(n==2) {
    if(!xten(x,1,g)) g=kput(x,1).to(t.device());
    if(!g.dim() && t.dim()) g.resize_as_(t).fill_(g[0]);
    t.backward(g,a,b); ok=true;
   } else if(n==1) {
     //PATCH t.backward(torch::nullopt,a,b); ok=true;
     t.backward({},a,b); ok=true;
   }
  }
  if(!ok)
   AT_ERROR("Unexpected arg(s) for backward call, expecting tensor, (tensor;sym), (tensor;grad tensor/array) or (tensor;grad tensor/array;sym)");
  return (K)0;
 KCATCH("backward");
}

// ----------------------------------
// tensor fns defined in k namespace
// ----------------------------------
V tensorfn(K x) {
 fn(x, "tensor",    KFN(tensor),1);
 fn(x, "backward",  KFN(backward),1);
 fn(x, "grad",      KFN(grad),1);
 fn(x, "vector",    KFN(vector),1);
 fn(x, "dim",       KFN(dim),1);
 fn(x, "size",      KFN(size),1);
 fn(x, "stride",    KFN(stride),1);
 fn(x, "device",    KFN(device),1);
 fn(x, "dtype",     KFN(dtype),1);
 fn(x, "layout",    KFN(layout),1);
 fn(x, "options",   KFN(options),1);
 fn(x, "shuffle",   KFN(kshuffle),1);
}
