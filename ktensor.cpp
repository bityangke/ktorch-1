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
// tensorlong - tensor/vector attributes returned to k as long scalar
// tensorsym - tensor/vector attributes returned to k as a symbol, e.g. device
// tensorflag - tensor/vector attributes returned to k as a boolean, e.g. leaf
// tensorsize - tensor/vector attributes returned to k as a long list, e.g. size/stride
// tensorattr - handle tensor attribute queries according to k datatype returned
// vectorattr - handle tensor vector attribute queries according to k datatype returned
// options - return dictionary/table of tensor/vector attributes
// ----------------------------------------------------------------------------------------------
ZJ storlong(const Storage& s,Attr a) {
 switch(a) {
  case Attr::elementsize: return s.elementSize();
  case Attr::size:        return s.size();
  case Attr::ptr:         return (intptr_t)s.data();
  case Attr::ref:         return s.use_count();
  default: AT_ERROR(mapattr(a),": not implemented for storage");
 }
}

ZJ tensorlong(const Tensor& t,Attr a) {
 switch(a) {
  case Attr::dim:         return t.dim();
  case Attr::elementsize: return t.is_sparse() ? tensorlong(t.values(),a) : storlong(t.storage(),a);
  case Attr::numel:       return t.numel();
  case Attr::offset:      return t.is_sparse() ? nj : t.storage_offset();
  case Attr::ref:         return t.use_count();
  case Attr::weakref:     return t.weak_use_count();
  case Attr::ptr:         return (intptr_t)t.unsafeGetTensorImpl();
  case Attr::sparsedim:   return t.is_sparse() ? t.sparse_dim() : 0;
  case Attr::storage:     return t.is_sparse() ? nj : (intptr_t)t.storage().data();
  default: AT_ERROR(mapattr(a),": not implemented for tensors");
 }
}

ZS tensorsym(const Tensor& t,Attr a) {
 switch(a) {
  case Attr::device:   return optsym(t.device());
  case Attr::dtype:    return optsym(t.dtype());
  case Attr::layout:   return optsym(t.layout());
  case Attr::gradient: return optsym(t.requires_grad());
  case Attr::gradfn:   return (S)(torch::autograd::as_variable_ref(t).grad_fn() ?
                                  torch::autograd::as_variable_ref(t).grad_fn()->name().c_str() : "");
  default: AT_ERROR(mapattr(a),": not implemented for tensors");
 }
}

Z B tensorflag(const Tensor &t,Attr a) {
 switch(a) {
  case Attr::coalesced:  return t.is_sparse() ? t.is_coalesced() : false;
  case Attr::contiguous: return t.is_sparse() ? false : t.is_contiguous();
  case Attr::leaf:       return torch::autograd::as_variable_ref(t).is_leaf();
  case Attr::pinned:     return t.is_pinned();
  default: AT_ERROR(mapattr(a),": not implemented for tensors");
 }
}

ZK tensorsize(const Tensor &t,Attr a) {
 switch(a) {
  case Attr::size:    return klist(t.dim(),t.sizes().data());
  case Attr::stride:  return t.is_sparse() ? ktn(0,0) : klist(t.dim(),t.strides().data());
  default: AT_ERROR(mapattr(a),": not implemented for tensors");
 }
}

K tensorattr(const Tensor &t,A k,Attr a) {
 switch(k) {
  case -KJ: return kj(tensorlong(t,a));
  case  KJ: return tensorsize(t,a);
  case -KS: return ks(tensorsym(t,a));
  case -KB: return kb(tensorflag(t,a));
  default: AT_ERROR(mapattr(a),": not implemented for tensors");
 }
}

K vectorattr(const TensorVector &v,A k,Attr a) {
 size_t i=0; K x=ktn(k<0 ? abs(k) : 0, v.size());
 try {
  for(auto&t:v) {
   switch(k) {
    case -KJ: kJ(x)[i]=tensorlong(t,a); break;
    case  KJ: kK(x)[i]=tensorsize(t,a); break;
    case -KS: kS(x)[i]=tensorsym(t,a);  break;
    case -KB: kG(x)[i]=tensorflag(t,a); break;
    default: AT_ERROR(mapattr(a),": not implemented for tensors");
   }
   ++i;
  }
 } catch(...) {
  if(x) r0(x);
  throw;
 }
 return x;
}

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

// ------------------------------------------------------------------------------------------------
// diagnostic functions -- check underlying pointers, storage data, reference counts, etc.
// ------------------------------------------------------------------------------------------------
// stordata - return CPU storage data as k list
// storinfo - return storage attributes & data as dictionary
// tensorinfo - return dictionary of attributes given tensor and detail level 0,1,2
// ------------------------------------------------------------------------------------------------
K stordata(const Storage& s) {
 TORCH_CHECK(s.device().is_cpu(), "Cannot copy CUDA storage via memcpy");
 std::cerr << "    dtype: " << s.dtype() << "\n";
 std::cerr << "     size: " << s.size() << "\n";
 std::cerr << " capacity: " << s.capacity() << "\n";
 K x=ktn(maptype(s.dtype()),s.size());
 memcpy(kG(x),s.data(),s.capacity());
 return x;
}

K storinfo(const Storage& s,const Storage& c) {
 K x=xD(ktn(KS,0),ktn(0,0)),*a=&kK(x)[0],*b=&kK(x)[1];
 js(a, mapattr(Attr::size));        jk(b, kj(storlong(s, Attr::size)));
 js(a, mapattr(Attr::elementsize)); jk(b, kj(storlong(s, Attr::elementsize)));
 js(a, mapattr(Attr::ref));         jk(b, kj(storlong(s, Attr::ref)));
 js(a, mapattr(Attr::ptr));         jk(b, kj(storlong(s, Attr::ptr)));
 js(a, mapattr(Attr::data));        jk(b, stordata(c));
 return x;
}

K tensorinfo(const Tensor& t,B d) {
 if(d && t.is_sparse()) {
  K x=xD(ktn(KS,0),ktn(0,0)),*a=&kK(x)[0],*b=&kK(x)[1];
  js(a, cs("indices")); jk(b, tensorinfo(t._indices(),d));
  js(a, cs("values"));  jk(b, tensorinfo(t._values(),d));
  return x;
 }
 K x=xD(ktn(KS,0),ktn(0,0)),*a=&kK(x)[0],*b=&kK(x)[1];
 js(a, mapattr(Attr::device));      jk(b, ks(tensorsym(t,  Attr::device)));
 js(a, mapattr(Attr::dtype));       jk(b, ks(tensorsym(t,  Attr::dtype)));
 js(a, mapattr(Attr::layout));      jk(b, ks(tensorsym(t,  Attr::layout)));
 js(a, mapattr(Attr::gradient));    jk(b, ks(tensorsym(t,  Attr::gradient)));
 js(a, mapattr(Attr::leaf));        jk(b, kb(tensorflag(t, Attr::leaf)));
 js(a, mapattr(Attr::gradfn));      jk(b, ks(tensorsym(t,  Attr::gradfn)));
 js(a, mapattr(Attr::dim));         jk(b, kj(tensorlong(t, Attr::dim)));
 js(a, mapattr(Attr::sparsedim));   jk(b, kj(tensorlong(t, Attr::sparsedim)));
 js(a, mapattr(Attr::size));        jk(b, tensorsize(t,    Attr::size));
 js(a, mapattr(Attr::stride));      jk(b, tensorsize(t,    Attr::stride));
 js(a, mapattr(Attr::numel));       jk(b, kj(tensorlong(t, Attr::numel)));
 js(a, mapattr(Attr::elementsize)); jk(b, kj(tensorlong(t, Attr::elementsize)));
 js(a, mapattr(Attr::contiguous));  jk(b, kb(tensorflag(t, Attr::contiguous)));
 js(a, mapattr(Attr::coalesced));   jk(b, kb(tensorflag(t, Attr::coalesced)));
 js(a, mapattr(Attr::offset));      jk(b, kj(tensorlong(t, Attr::offset)));
 js(a, mapattr(Attr::ptr));         jk(b, kj(tensorlong(t, Attr::ptr)));
 js(a, mapattr(Attr::ref));         jk(b, kj(tensorlong(t, Attr::ref)));
 if(d) {
  js(a, mapattr(Attr::storage));   
  jk(b, storinfo(t.storage(),
        t.dtype()==torch::kHalf ? t.cpu().to(torch::kFloat).storage() : t.cpu().storage()));
 }
 return x;
}

// --------------------------------------------------------------------------------------------
// perm - return permutation indices given tensor and dimension
// vcheck - check for matching dimension size & device for each tensor in vector
// vperm - check vector for same dim size & device, return permutation indices
// shuffle - shuffle tensor or vector of same size along given dimension
// shuffle_ - in-place version of tensor/vector shuffle
// kshuffle,1,2 - k api functions, expects tensor/vector input or (input;dim;inplace flag)
// --------------------------------------------------------------------------------------------
Z Tensor perm(const Tensor& t,int64_t d) {
 return torch::randperm(t.size(d),torch::dtype(torch::kLong).device(t.device()));
}

ZV vcheck(const TensorVector& v,int64_t d) {
 int64_t i=0,n; torch::Device c=torch::kCPU;
 for(auto& t:v) {
  if(!i)
   n=t.size(d),c=t.device();
  else if(n != t.size(d))
   AT_ERROR("Size mismatch: tensor[",i,"] size=",t.size(d),", but previous tensor(s) have size=",n," for dim ",d);
  else if (c != t.device())
   AT_ERROR("Device mismatch: tensor[",i,"] is on ",t.device(),", but previous tensor(s) are on ", c);
  ++i;
 }
}

Z Tensor vperm(const TensorVector& v,int64_t d) {vcheck(v,d); return v.size() ? perm(v[0],d) : Tensor();}

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
ZK kshuffle2(TensorVector& v,int64_t d,B b) {return b ? shuffle_(v,d),(K)0 : kvec(shuffle(v,d));}

KAPI kshuffle(K x) {
 KTRY
  B b=true; int64_t d=0; Ktag *g; //default is in-place, along 1st dim
  if((g=xtag(x)) || 
    ((g=xtag(x,0)) && ((x->n==2 && (xint64(x,1,d) || xbool(x,1,b))) ||
                       (x->n==3 &&  xint64(x,1,d) && xbool(x,2,b)))))
   switch(g->a) {
    case Class::tensor: return kshuffle1(((Kten*)g)->t,d,b);
    case Class::vector: return kshuffle2(((Kvec*)g)->v,d,b);
    default: AT_ERROR("shuffle not implemented for ",mapclass(g->a));
   }
  else
   AT_ERROR("unrecognized arg(s) for shuffle");
 KCATCH("shuffle");
}

// ------------------------------------------------------------------------------------------
// tensor utility fns: 
// ------------------------------------------------------------------------------------------
// tensorcopy - tgt <- src values, must have same type & device, tgt resized if src larger
// grad = return gradient data or empty, if ptr enlisted, return gradient ptr, which must be free'd
// backward - backprop given tensor, optional tensor and sym for retain/create gradient graph
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
 fn(x, "options",   KFN(options),1);
 fn(x, "shuffle",   KFN(kshuffle),1);
}
