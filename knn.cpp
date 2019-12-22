#include "ktorch.h"
#include "knn.h"

// PDIM/XDIM fill expanding array from k pairs/dict or kth element of general list
#define PDIM(p,d,a) psize(p,d,(*a).data())
#define XDIM(x,k,d,a) xsize(x,k,d,(*a).data())

// append a module option to a k dictionary given dict,name & value
#define OPTION(x,k,v) dictadd(x, mset(Setting::k), v)

// append a module with name if not null (method needs `std::string` ??)
#define PUSH(q,n,m) n ? q->push_back(std::string(n),m) : q->push_back(m)

// ----------------------------------------------------------------------------
// kseq - allocate an object to store a pointer to a sequential module
// seqto - given sequential module & options, change device/data type
// ----------------------------------------------------------------------------
K kseq(const Sequential& q) {return kptr(new Kseq(q));}

K seqto(Kseq* q,const TensorOptions& o,bool a) {
 auto s=torch::typeMetaToScalarType(o.dtype());
 if(o.has_device() && o.has_dtype()) q->q->to(o.device(),s,a);
 else if(o.has_device())             q->q->to(o.device(),a);
 else                                q->q->to(s,a);
 return (K)0;
}

// --------------------------------------------------------------------------------------------
// enum<-rnnfn(sym)    match symbol to enum for activation function
// sym<-rnnfn(options) return symbol matching activation fn, else null (e.g. for gru/lstm)
// rnnfn(options,sym)  set activation function if rnn options, else no-op
// --------------------------------------------------------------------------------------------
static torch::nn::RNNActivation rnnfn(S s) {
 for(auto& m:env().rnnfn) if (s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("Unrecognized rnn activiation function: ",s);
}

template<typename O> static S rnnfn(O& o) {return nullptr;}
template<> S rnnfn<torch::nn::RNNOptions>(torch::nn::RNNOptions& o) {
 for(auto& m:env().rnnfn) if (o.activation()==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized rnn activiation function: ",(I)o.activation());
}

template<typename O> static void rnnfn(O& o,torch::nn::RNNActivation f) {}
template<> void rnnfn<torch::nn::RNNOptions>(torch::nn::RNNOptions& o,torch::nn::RNNActivation f) {o.activation(f);}

// -----------------------------------------------------------------------------------
// msym - map to/from sym & enum for module, e.g. `conv3d <-> Cast::conv3d
// mset - map to/from sym & enum for module options, e.g. `bias <-> Setting::bias
// -----------------------------------------------------------------------------------
static S msym(Cast c) {
 for(auto& m:env().module) if(c==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized module: ",(I)c);
}

static Cast msym(S s) {
 for(auto& m:env().module) if(s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("Unrecognized module: ",s);
}

static S mset(Setting o) {
 for(auto& m:env().mset) if(o==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized module option: ",(I)o);
}

static Setting mset(S s) {
 for(auto& m:env().mset) if(s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("Unrecognized option: ",s);
}

// ------------------------------------------------------------------------------------------
// mkeys - initialize keys for dict/table of module state: `module`name`options`parms`buffers
// mvals - initialize corresponding k values for state dictionary or table
// ------------------------------------------------------------------------------------------
static K mkeys(bool b) { // b:true if including class, parms & buffers
 if(b) return statekeys();
 K x=ktn(KS,3);
 for(auto &m:env().state) {
       if(std::get<1>(m)==State::module)  kS(x)[0]=std::get<0>(m);
  else if(std::get<1>(m)==State::name)    kS(x)[1]=std::get<0>(m);
  else if(std::get<1>(m)==State::options){kS(x)[2]=std::get<0>(m); break;}
 }
 return x;
}

static K mvals(bool b,J n) {
 K x=ktn(0,b ? 6 : 3);
 if(n<0) {
  if(b) kK(x)[0]=kc('m');
 } else {
  if(b) kK(x)[0]=kp((S)std::string(n,'m').data());
  for(J i=b;i<x->n;++i) kK(x)[i]=ktn(i<(2+b) ? KS : 0,n);
 }
 return x;
}
 
// ----------------------------------------------------------------------------------------------------
// covers of input checking fns with error msg specific to module settings and module names:
// ----------------------------------------------------------------------------------------------------
// mbool - check positional args or name-value pairs for boolean, else error w'module & option
// int64 - check positional args or name-value pairs for long int, else error w'module & option
// int64n - int64 but returns optional, i.e. nullopt if k value is null
// mdouble - check for double(or long) from positional or name-value pair arg
// exarray - check positional or name-value args for long(s), return expanding array,  else error
// exdouble - similar to exarray, but for double array
// ----------------------------------------------------------------------------------------------------
static bool mbool(K x,J i,Cast c,Setting s) {
 bool b;
 TORCH_CHECK(xbool(x,i,b), msym(c)," ",mset(s),": expected boolean scalar, given ",kname(kK(x)[i]->t));
 return b;
}

static bool mbool(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KB, msym(c)," ",p.k,": expected boolean scalar, given ",kname(p.t));
 return p.b;
}

static int64_t int64(K x,J i,Cast c,Setting s) {
 int64_t n;
 TORCH_CHECK(xint64(x,i,n), msym(c)," ",mset(s),": expected long scalar, given ",kname(kK(x)[i]->t));
 return n;
}

static int64_t int64(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ, msym(c)," ",p.k,": expected long scalar, given ",kname(p.t));
 return p.j;
}

static c10::optional<int64_t> int64n(K x,J i,Cast c,Setting s) {auto n=int64(x,i,c,s); if(n==nj) return c10::nullopt; else return n;}
static c10::optional<int64_t> int64n(const Pairs& p,Cast c)    {auto n=int64(p,c);     if(n==nj) return c10::nullopt; else return n;}

static double mdouble(K x,J i,Cast c,Setting s) {
 double f;
 TORCH_CHECK(xnum(x,i,f), msym(c)," ",mset(s),": expected double, given ",kname(kK(x)[i]->t));
 return f;
}

static double mdouble(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==-KF, msym(c)," ",p.k,": expected double, given ",kname(p.t));
 return pdouble(p);
}

template<size_t D> ExpandingArray<D> exarray(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KJ || x->t==KJ, msym(c)," ",mset(s),": expected long(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KJ || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 if(x->t==-KJ)
  return ExpandingArray<D>(x->j);
 else
  return ExpandingArray<D>(IntArrayRef((int64_t*)kJ(x),x->n));
}

template<size_t D> ExpandingArray<D> exarray(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==KJ,   msym(c)," ",p.k,": expected long(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KJ || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 if(p.t==-KJ)
  return ExpandingArray<D>(p.j);
 else
  return ExpandingArray<D>(IntArrayRef((int64_t*)kJ(p.v),p.v->n));
}

template<size_t D> Exdouble<D> exdouble(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KF || x->t==KF, msym(c)," ",mset(s),": expected double(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KF || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 if(x->t==-KF)
  return Exdouble<D>(x->f);
 else
  return Exdouble<D>(torch::ArrayRef<double>(kF(x),x->n));
}

template<size_t D> Exdouble<D> exdouble(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KF || p.t==KF,   msym(c)," ",p.k,": expected double(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KF || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 if(p.t==-KF)
  return Exdouble<D>(p.f);
 else
  return Exdouble<D>(torch::ArrayRef<double>(kF(p.v),p.v->n));
}

// --------------------------------------------------------------------------------------
// bnorm - create batchnorm module given options/set dictionary of options given module
// --------------------------------------------------------------------------------------
torch::nn::BatchNorm bnorm(K x,J k) {
 bool a=true,t=true; double e=1e-5,m=0.1; Pairs p; J i=-1,n=xargc(x,k,p);
 if(!((n==0 && p.n) || (n==1 && xlong(x,k,i))))
  AT_ERROR("Unrecognized arguments for batch normalization");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:        i=plong(p); break;
   case Setting::affine:    a=pbool(p); break;
   case Setting::track:     t=pbool(p); break;
   case Setting::eps:       e=pdouble(p); break;
   case Setting::momentum:  m=pdouble(p); break;
   default: AT_ERROR("Unrecognized batch norm option: ",p.k); break;
  }
 if(i<0) AT_ERROR("number of input features must be set, currently in = ",i);
 return torch::nn::BatchNorm(torch::nn::BatchNormOptions(i).affine(a).track_running_stats(t).eps(e).momentum(m));
}

static void bnorm(bool a,K x,const torch::nn::BatchNormImpl* m) {
 torch::nn::BatchNormOptions o=m->options, d(o.num_features());
 OPTION(x, in, kj(o.num_features()));
 if(a || (o.eps()      != d.eps()))      OPTION(x, eps,       kf(o.eps()));
 if(a || (o.momentum() != d.momentum())) OPTION(x, momentum,  kf(o.momentum().value()));
 if(a || (o.affine()   != d.affine()))   OPTION(x, affine,    kb(o.affine()));
 if(a || (o.track_running_stats() != d.track_running_stats())) OPTION(x, track, kb(o.track_running_stats()));
}

// --------------------------------------------------------------------------------------
// conv - create 1-3d convolution/transposed convolution, set dictionary given module
// --------------------------------------------------------------------------------------
template<size_t D> static torch::nn::ConvOptions<D> conv(K x,J i,Cast c) {
 torch::nn::ConvOptions<D> o(0,0,0);
 bool in=false,out=false,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.in_channels    (int64(x,i+j,c,Setting::in));        in=true; break;
    case 1: o.out_channels   (int64(x,i+j,c,Setting::in));       out=true; break;
    case 2: o.kernel_size    (exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    case 3: o.stride         (exarray<D>(x,i+j,c,Setting::stride));   break;
    case 4: o.padding        (exarray<D>(x,i+j,c,Setting::pad));      break;
    case 5: o.output_padding (exarray<D>(x,i+j,c,Setting::outpad));   break;
    case 6: o.dilation       (exarray<D>(x,i+j,c,Setting::dilate));   break;
    case 7: o.groups         (int64(x,i+j,c,Setting::groups));        break;
    case 8: o.bias           (mbool    (x,i+j,c,Setting::bias));      break;
    case 9: o.transposed     (mbool    (x,i+j,c,Setting::transpose)); break;
    default: AT_ERROR(msym(c),": up to 10 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:        o.in_channels   (int64(p,c));     in=true; break;
   case Setting::out:       o.out_channels  (int64(p,c));    out=true; break;
   case Setting::size:      o.kernel_size   (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:    o.stride        (exarray<D>(p,c)); break;
   case Setting::pad:       o.padding       (exarray<D>(p,c)); break;
   case Setting::outpad:    o.output_padding(exarray<D>(p,c)); break;
   case Setting::dilate:    o.dilation      (exarray<D>(p,c)); break;
   case Setting::groups:    o.groups         (int64(p,c));     break;
   case Setting::bias:      o.bias           (mbool(p,c));     break;
   case Setting::transpose: o.transposed     (mbool(p,c));     break;
   default: AT_ERROR("Unrecognized convolution option: ",p.k); break;
  }
 TORCH_CHECK(in,  msym(c),": number of input channels not defined");
 TORCH_CHECK(out, msym(c),": number of output channels not defined");
 TORCH_CHECK(sz,  msym(c),": no kernel size(s) given");
 return o;
}

template<size_t D,typename M>
static void conv(bool a,K x,const M* m) {
 torch::nn::ConvOptions<D> o=m->options, d(o.in_channels(),o.out_channels(),o.kernel_size());
 OPTION(x, in,   kj(o.in_channels()));
 OPTION(x, out,  kj(o.out_channels()));
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || (*o.stride()         != *d.stride()))         OPTION(x, stride,    KEX(o.stride()));
 if(a || (*o.padding()        != *d.padding()))        OPTION(x, pad,       KEX(o.padding()));
 if(a || (*o.output_padding() != *d.output_padding())) OPTION(x, outpad,    KEX(o.output_padding()));
 if(a || (*o.dilation()       != *d.dilation()))       OPTION(x, dilate,    KEX(o.dilation()));
 if(a || ( o.groups()         !=  d.groups()))         OPTION(x, groups,    kj(o.groups()));
 if(a || ( o.bias()           !=  d.bias()))      OPTION(x, bias,      kb(o.bias()));
 if(a || ( o.transposed()     !=  d.transposed()))     OPTION(x, transpose, kb(o.transposed()));
}

// --------------------------------------------------------------------------------------
// drop - create dropout module given probability/set dictionary given module
// --------------------------------------------------------------------------------------
static torch::nn::DropoutOptions drop(K x,J i,Cast c) {
 torch::nn::DropoutOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
    case 1: o.inplace(mbool(x,i+j,c,Setting::inplace)); break;
    default: AT_ERROR(msym(c),": up to 2 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR("Unrecognized dropout option: ",p.k); break;
  }
 return o;
}

static void drop(bool a,K x,const torch::nn::DropoutOptions& o) {
 torch::nn::DropoutOptions d;
 if(a || o.p()       != d.p())       OPTION(x, p,       kf(o.p()));
 if(a || o.inplace() != d.inplace()) OPTION(x, inplace, kb(o.inplace()));
}

// --------------------------------------------------------------------------------------
// embed - create embedding module given options/set dictionary of options given module
// --------------------------------------------------------------------------------------
torch::nn::Embedding embed(K x,J k) {
 Pairs p; J i=-1,j=-1,n=xargc(x,k,p);
 if(!((n==0 && p.n) || (xlong(x,k,i) && (n==1 || (n==2 && xlong(x,k+1,j))))))
  AT_ERROR("Unrecognized arguments for embedding module");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::rows: i=plong(p); break;
   case Setting::cols: j=plong(p); break;
   default: AT_ERROR("Embedding option: ",p.k," unrecognized, expected one of rows,cols");
  }
 if(i<0 || j<0) {
  AT_ERROR("Embedding rows & cols of embedding size must be non-negative, rows=",i,", cols=",j);
 }
 return torch::nn::Embedding(i,j);
}

static void embed(K x,const torch::nn::EmbeddingImpl* m) {
 auto o=m->options;
 OPTION(x, rows, kj(o.num_embeddings()));
 OPTION(x, cols, kj(o.embedding_dim()));
}

// --------------------------------------------------------------------------------------
// linear - parse/retrieve args, invoke functional form
// --------------------------------------------------------------------------------------
static torch::nn::LinearOptions linear(K x,J i,Cast c) {
 bool b=true; int64_t in=nj,out=nj; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:  in=int64(x,i+j,c,Setting::in);   break;
    case 1: out=int64(x,i+j,c,Setting::out);  break;
    case 2:   b=mbool(x,i+j,c,Setting::bias); break;
    default: AT_ERROR(msym(c),": up to 3 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:   in=int64(p,c); break;
   case Setting::out:  out=int64(p,c); break;
   case Setting::bias: b=mbool(p,c); break;
   default: AT_ERROR("Unrecognized linear option: ",p.k); break;
  }
 TORCH_CHECK(in>0,  msym(c), ": positive input size required");
 TORCH_CHECK(out>0, msym(c), ": positive output size required");
 return torch::nn::LinearOptions(in,out).bias(b);
}

static void linear(bool a,K x,const torch::nn::LinearImpl *m) {
 torch::nn::LinearOptions o=m->options, d(o.in_features(),o.out_features());
 OPTION(x, in,  kj(o.in_features()));
 OPTION(x, out, kj(o.out_features()));
 if(a || (o.bias() != d.bias())) OPTION(x, bias, kb(o.bias()));
}

KAPI klinear(K x) {
 KTRY
  TORCH_CHECK(!x->t, "linear not implemented for ",kname(x->t));
  TORCH_CHECK(x->n==2 || x->n==3, "linear requires 2-3 args, (input; weight; optional bias)");
  Tensor r, *a=xten(x,0), *w=xten(x,1), *b=xten(x,2);
  if(x->n==2)
   r=torch::linear(a ? *a : kput(x,0), w ? *w : kput(x,1));
  else
   r=torch::linear(a ? *a : kput(x,0), w ? *w : kput(x,1), b ? *b : kput(x,2));
  return kresult(a||w||b, r);
 KCATCH("linear");
}

// --------------------------------------------------------------------------------------
// rnn - create rnn/gru/lstm module given options/set dictionary of options from module
// --------------------------------------------------------------------------------------
template<typename M,typename O>
static M rnn(S s,K x,J k) {
 auto f=torch::nn::RNNActivation::ReLU;
 bool b=true,bi=false,ba=false; Pairs p; J i=-1,h=-1,l=1,n=xargc(x,k,p); double d=0.0;
 if(!((n==0 && p.n) || (xlong(x,k,i) && (n==1 || (n==2 && xlong(x,k+1,h))))))
  AT_ERROR("Unrecognized arguments for ",s," module");
 bool r=std::is_same<M,torch::nn::RNN>::value;
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:          i=plong(p); break;
   case Setting::hidden:      h=plong(p); break;
   case Setting::layers:      l=plong(p); break;
   case Setting::bias:        b=pbool(p); break;
   case Setting::bi:         bi=pbool(p); break;
   case Setting::batchfirst: ba=pbool(p); break;
   case Setting::drop:      d=pdouble(p); break;
   case Setting::fn: if(r) f=rnnfn(psym(p)); else AT_ERROR("activation function only for RNN module"); break;
   default: AT_ERROR(s," option: ",p.k," unrecognized, expected one of in,hidden,layers, bias,bi,batchfirst, drop,fn");
  }
 auto o=O(i,h).layers(l).dropout(d).with_bias(b).bidirectional(bi).batch_first(ba);
 if(r) rnnfn(o,f);
 return M(o);
}

template<typename M,typename O>
static void rnn(bool a,K x,const M* m) {
 O o=m->options, d(o.input_size(),o.hidden_size()); S f=rnnfn(o);
 OPTION(x, in,     kj(o.input_size()));
 OPTION(x, hidden, kj(o.hidden_size()));
 if(a || (o.layers()        != d.layers()))       OPTION(x, layers,     kj(o.layers()));
 if(a || (o.dropout()       != d.dropout()))      OPTION(x, drop,       kf(o.dropout()));
 if((a && f) || f           != rnnfn(d))          OPTION(x, fn,         ks(f));
 if(a || (o.with_bias()     != d.with_bias()))    OPTION(x, bias,       kb(o.with_bias()));
 if(a || (o.bidirectional() != d.bidirectional()))OPTION(x, bi,         kb(o.bidirectional()));
 if(a || (o.batch_first()   != d.batch_first()))  OPTION(x, batchfirst, kb(o.batch_first()));
}

// ----------------------------------------------------------------------------------
//  maxpool - process args, return dictionary of options, call functional form
// ----------------------------------------------------------------------------------
template<size_t D> static torch::nn::MaxPoolOptions<D> maxpool(K x,J i,Cast c) {
 torch::nn::MaxPoolOptions<D> o(0);
 bool sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.kernel_size(exarray<D>(x,i+j,c,Setting::size));    sz=true; break;
    case 1: o.stride     (exarray<D>(x,i+j,c,Setting::stride));  st=true; break;
    case 2: o.padding    (exarray<D>(x,i+j,c,Setting::pad));     break;
    case 3: o.dilation   (exarray<D>(x,i+j,c,Setting::dilate));  break;
    case 4: o.ceil_mode  (mbool     (x,i+j,c,Setting::ceiling)); break;
    default: AT_ERROR(msym(c),": up to 5 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride     (exarray<D>(p,c)); st=true; break;
   case Setting::pad:     o.padding    (exarray<D>(p,c)); break;
   case Setting::dilate:  o.dilation   (exarray<D>(p,c)); break;
   case Setting::ceiling: o.ceil_mode  (mbool(p,c)); break;
   default: AT_ERROR("Unrecognized max pooling option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<size_t D,typename M> static void maxpool(bool a,K x,const M* m) {
 torch::nn::MaxPoolOptions<D> o=m->options, d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   OPTION(x, stride,  KEX(o.stride()));
 if(a || *o.padding()  != *d.padding())  OPTION(x, pad,     KEX(o.padding()));
 if(a || *o.dilation() != *d.dilation()) OPTION(x, dilate,  KEX(o.dilation()));
 if(a || o.ceil_mode() != d.ceil_mode()) OPTION(x, ceiling, kb(o.ceil_mode()));
}

static K maxpool(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  switch(c) {
   case Cast::maxpool1d: r=torch::nn::functional::max_pool1d(t ? *t : kput(x,0), maxpool<1>(x,1,c)); break;
   case Cast::maxpool2d: r=torch::nn::functional::max_pool2d(t ? *t : kput(x,0), maxpool<2>(x,1,c)); break;
   case Cast::maxpool3d: r=torch::nn::functional::max_pool3d(t ? *t : kput(x,0), maxpool<3>(x,1,c)); break;
   default: AT_ERROR("Unrecognized max pooling function");
  }
  return kresult(t,r);
 KCATCH("maxpool");
}

KAPI maxpool1d(K x) {return maxpool(x,Cast::maxpool1d);}
KAPI maxpool2d(K x) {return maxpool(x,Cast::maxpool2d);}
KAPI maxpool3d(K x) {return maxpool(x,Cast::maxpool3d);}

// ----------------------------------------------------------------------------------
//  avgpool - process args, return dictionary of options, call functional form
// ----------------------------------------------------------------------------------
template<size_t D> static torch::nn::AvgPoolOptions<D> avgpool(K x,J i,Cast c) {
 torch::nn::AvgPoolOptions<D> o(0);
 bool sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.kernel_size      (exarray<D>(x,i+j,c,Setting::size));   sz=true; break;
    case 1: o.stride           (exarray<D>(x,i+j,c,Setting::stride)); st=true; break;
    case 2: o.padding          (exarray<D>(x,i+j,c,Setting::pad));      break;
    case 3: o.ceil_mode        (mbool     (x,i+j,c,Setting::ceiling));  break;
    case 4: o.count_include_pad(mbool     (x,i+j,c,Setting::countpad)); break;
    case 5: o.divisor_override (int64n    (x,i+j,c,Setting::divisor));  break;
    default: AT_ERROR(msym(c),": up to 6 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::size:    o.kernel_size (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride      (exarray<D>(p,c)); st=true; break;
   case Setting::pad:     o.padding     (exarray<D>(p,c)); break;
   case Setting::ceiling: o.ceil_mode        (mbool(p,c)); break;
   case Setting::countpad:o.count_include_pad(mbool(p,c)); break;
   case Setting::divisor: o.divisor_override(int64n(p,c)); break;
   default: AT_ERROR("Unrecognized avg pooling option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<size_t D,typename M> static void avgpool(bool a,K x,const M* m) {
 torch::nn::AvgPoolOptions<D> o=m->options, d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()           != *d.stride())           OPTION(x, stride,   KEX(o.stride()));
 if(a || *o.padding()          != *d.padding())          OPTION(x, pad,      KEX(o.padding()));
 if(a || o.ceil_mode()         != d.ceil_mode())         OPTION(x, ceiling,  kb(o.ceil_mode()));
 if(a || o.count_include_pad() != d.count_include_pad()) OPTION(x, countpad, kb(o.count_include_pad()));
 if(a || o.divisor_override().has_value())               OPTION(x, divisor,  kj(o.divisor_override() ? o.divisor_override().value() : nj));
}

static K avgpool(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  switch(c) {
   case Cast::avgpool1d: r=torch::nn::functional::avg_pool1d(t ? *t : kput(x,0), avgpool<1>(x,1,c)); break;
   case Cast::avgpool2d: r=torch::nn::functional::avg_pool2d(t ? *t : kput(x,0), avgpool<2>(x,1,c)); break;
   case Cast::avgpool3d: r=torch::nn::functional::avg_pool3d(t ? *t : kput(x,0), avgpool<3>(x,1,c)); break;
   default: AT_ERROR("Unrecognized avg pooling function");
  }
  return kresult(t,r);
 KCATCH("avgpool");
}

KAPI avgpool1d(K x) {return avgpool(x,Cast::avgpool1d);}
KAPI avgpool2d(K x) {return avgpool(x,Cast::avgpool2d);}
KAPI avgpool3d(K x) {return avgpool(x,Cast::avgpool3d);}

// ------------------------------------------------------------------------------------
//  adaptive pooling - process args, return dictionary of options, call functional form
// ------------------------------------------------------------------------------------
template<size_t D,typename T> static T adapt(K x,J i,Cast c) {
 T o(0); bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.output_size(exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    default: AT_ERROR(msym(c),": 1 positional argument expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::size: o.output_size(exarray<D>(p,c)); sz=true; break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no output size given");
 return o;
}

template<typename M> static void adapt(K x,const M* m) {
 OPTION(x, size, KEX(m->options.output_size()));
}

static K adapt(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  switch(c) {
   case Cast::adaptmax1d: r=torch::nn::functional::adaptive_max_pool1d(t ? *t : kput(x,0), adapt<1,torch::nn::AdaptiveMaxPool1dOptions>(x,1,c)); break;
   case Cast::adaptmax2d: r=torch::nn::functional::adaptive_max_pool2d(t ? *t : kput(x,0), adapt<2,torch::nn::AdaptiveMaxPool2dOptions>(x,1,c)); break;
   case Cast::adaptmax3d: r=torch::nn::functional::adaptive_max_pool3d(t ? *t : kput(x,0), adapt<3,torch::nn::AdaptiveMaxPool3dOptions>(x,1,c)); break;

   case Cast::adaptavg1d: r=torch::nn::functional::adaptive_avg_pool1d(t ? *t : kput(x,0), adapt<1,torch::nn::AdaptiveAvgPool1dOptions>(x,1,c)); break;
   case Cast::adaptavg2d: r=torch::nn::functional::adaptive_avg_pool2d(t ? *t : kput(x,0), adapt<2,torch::nn::AdaptiveAvgPool2dOptions>(x,1,c)); break;
   case Cast::adaptavg3d: r=torch::nn::functional::adaptive_avg_pool3d(t ? *t : kput(x,0), adapt<3,torch::nn::AdaptiveAvgPool3dOptions>(x,1,c)); break;
   default: AT_ERROR("Unrecognized adaptive pooling function");
  }
  return kresult(t,r);
 KCATCH("adaptive pooling");
}

KAPI adaptmax1d(K x) {return adapt(x,Cast::adaptmax1d);}
KAPI adaptmax2d(K x) {return adapt(x,Cast::adaptmax2d);}
KAPI adaptmax3d(K x) {return adapt(x,Cast::adaptmax3d);}
KAPI adaptavg1d(K x) {return adapt(x,Cast::adaptavg1d);}
KAPI adaptavg2d(K x) {return adapt(x,Cast::adaptavg2d);}
KAPI adaptavg3d(K x) {return adapt(x,Cast::adaptavg3d);}

// ----------------------------------------------------------------------------------
// fpool - fractional max pooling for 2 & 3d layers
// ----------------------------------------------------------------------------------
template<size_t D> static FractionalMaxPoolOptions<D> fpool(K x,J i,Cast c) {
 FractionalMaxPoolOptions<D> o(0);
 bool e,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   e=xempty(x,i+j);
   switch(j) {
    case 0: o.size(exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    case 1: if(e) o.outsize(c10::nullopt); else o.outsize(exarray  <D>(x,i+j,c,Setting::outsize)); break;
    case 2: if(e) o.ratio  (c10::nullopt); else o.ratio  (exdouble<D>(x,i+j,c,Setting::ratio));   break;
    case 3: o.indices(mbool(x,i+j,c,Setting::indices)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p)) {
  e=pempty(p);
  switch(mset(p.k)) {
   case Setting::size:    o.size(exarray<D>(p,c)); sz=true; break;
   case Setting::outsize: if(e) o.outsize(c10::nullopt); else o.outsize(exarray  <D>(p,c)); break;
   case Setting::ratio:   if(e) o.ratio  (c10::nullopt); else o.ratio  (exdouble<D>(p,c)); break;
   case Setting::indices: o.indices(mbool(p,c)); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 TORCH_CHECK(o.outsize()||o.ratio(), msym(c),": no output size or ratio given");
 TORCH_CHECK(!(o.outsize()&&o.ratio()), msym(c),": cannot specify both output size & ratio");
 return o;
}

template<size_t D,typename M> static void fpool(bool a,K x,const M* m) {
 FractionalMaxPoolOptions<D> o=m->options, d(o.size());
 OPTION(x, size, KEX(o.size()));
 if(a || o.outsize().has_value())    OPTION(x, outsize, o.outsize() ? KEX(o.outsize().value()) : ktn(0,0));
 if(a || o.ratio().has_value())      OPTION(x, ratio,   o.ratio()   ? KEX(o.ratio().value())   : ktn(0,0));
 if(a || o.indices() != d.indices()) OPTION(x, indices, kb(o.indices()));
}

// ----------------------------------------------------------------------------------
// lppool - power-average pooling
// ----------------------------------------------------------------------------------
template<size_t D> static torch::nn::LPPoolOptions<D> lppool(K x,J i,Cast c) {
 torch::nn::LPPoolOptions<D> o(0,0);
 bool pw=false,sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.norm_type  (mdouble(x,i+j,  c,Setting::power));   pw=true; break;
    case 1: o.kernel_size(exarray<D>(x,i+j,c,Setting::size));   sz=true; break;
    case 2: o.stride     (exarray<D>(x,i+j,c,Setting::stride)); st=true; break;
    case 3: o.ceil_mode  (mbool    (x,i+j,c,Setting::ceiling)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::power:   o.norm_type  (mdouble   (p,c)); pw=true; break;
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride     (exarray<D>(p,c)); st=true; break;
   case Setting::ceiling: o.ceil_mode  (mbool(p,c)); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(pw, msym(c),": no power given");
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<size_t D,typename M> static void lppool(bool a,K x,const M* m) {
 torch::nn::LPPoolOptions<D> o=m->options, d(o.norm_type(),o.kernel_size());
 OPTION(x, power, kf(o.norm_type()));
 OPTION(x, size,  KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   OPTION(x, stride,  KEX(o.stride()));
 if(a || o.ceil_mode() != d.ceil_mode()) OPTION(x, ceiling, kb(o.ceil_mode()));
}

static K lppool(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  switch(c) {
   case Cast::lppool1d: r=torch::nn::functional::lp_pool1d(t ? *t : kput(x,0), lppool<1>(x,1,c)); break;
   case Cast::lppool2d: r=torch::nn::functional::lp_pool2d(t ? *t : kput(x,0), lppool<2>(x,1,c)); break;
   default: AT_ERROR("Unrecognized LP pooling function");
  }
  return kresult(t,r);
 KCATCH("lppool");
}

KAPI lppool1d(K x) {return lppool(x,Cast::lppool1d);}
KAPI lppool2d(K x) {return lppool(x,Cast::lppool2d);}

// ----------------------------------------------------------------------------------
// padmode - match k xymbol to std::variant style enumeration
// pad - n-dimensional padding, specify even number of sizes and optional pad value
// ----------------------------------------------------------------------------------
static void padmode(torch::nn::functional::PadFuncOptions& o,S s) {
 switch(emap(s)) {
  case Enum::constant:  o.mode(torch::kConstant); break;
  case Enum::reflect:   o.mode(torch::kReflect); break;
  case Enum::replicate: o.mode(torch::kReplicate); break;
  case Enum::circular:  o.mode(torch::kCircular); break;
  default: AT_ERROR("unrecognized padding mode: ",s); break;
 }
}

static torch::nn::functional::PadFuncOptions pad(K x,J i,Cast c) {
 torch::nn::functional::PadFuncOptions o({}); S s; Pairs p; J n=xargc(x,i,p); IntArrayRef a;
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: TORCH_CHECK(xsize(x,i+j,a), msym(c),": expecting 1st arg of padding size(s)"); break;
    case 1:
     if(xsym(x,i+j,s)) padmode(o,s);
     else if(n==2)     o.value(mdouble(x,i+j,c,Setting::value));
     else AT_ERROR("pad: unrecognized 2nd arg, expecting mode or value");
     break;
    case 2: o.value(mdouble(x,i+j,c,Setting::value)); break;
    default: AT_ERROR(msym(c),": up to 3 positional args expected(padding;mode;value), ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::pad:    psize(p,a); break;
   case Setting::mode:   padmode(o,psym(p)); break;
   case Setting::value:  o.value(mdouble(p,c)); break;
   default: AT_ERROR("padding option: ",p.k," not recognized");
  }
 TORCH_CHECK(a.size()>0 && !(a.size() % 2),
             a.size()," pad size(s) given, expecting pairs for left,right or left,right,top,bottom.. etc");
 return o.pad(a.vec());
}

static void pad(bool a,K x,const PadImpl* m) {
 const torch::nn::functional::PadFuncOptions d({}), &o=m->options;
 OPTION(x, pad, klist(o.pad().size(),o.pad().data()));
 if(a || ESYM(o.mode()) != ESYM(d.mode())) OPTION(x, mode,  ks(ESYM(o.mode())));
 if(a || o.value()      != d.value())      OPTION(x, value, kf(o.value()));
}

// ----------------------------------------------------------------------------------
// cpad - constant pad w'fixed dimension and optional value (defaults to zero)
// ----------------------------------------------------------------------------------
template<size_t D,typename M> static M cpad(K x,J i,Cast c) {
 M o(0,0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.padding(exarray<D*2>(x,i+j,c,Setting::pad)); sz=true; break;
    case 1: o.value(mdouble(x,i+j,c,Setting::value)); break;
    default: AT_ERROR(msym(c),": up to 2 positional args expected(padding;value), ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::pad: o.padding(exarray<D*2>(p,c)); sz=true; break;
   case Setting::value: o.value(mdouble(p,c)); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no padding sizes given");
 return o;
}

template<typename M> static void cpad(K x,const M* m) {
 OPTION(x, pad, KEX(m->options.padding()));
 OPTION(x, value, kf(m->options.value()));
}

// ----------------------------------------------------------------------------------
// npad - reflect/replicate/zero pad w'fixed dimension
// ----------------------------------------------------------------------------------
template<size_t D,typename M> static M npad(K x,J i,Cast c) {
 M o(0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.padding(exarray<D*2>(x,i+j,c,Setting::pad)); sz=true; break;
    default: AT_ERROR(msym(c),": only 1 positional argument expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::pad: o.padding(exarray<D*2>(p,c)); sz=true; break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no padding sizes given");
 return o;
}

template<typename M> static void npad(K x,const M* m) {
 OPTION(x, pad, KEX(m->options.padding()));
}

// ------------------------------------------------------------------------------------
// noarg:  activation fns w'out args, logsigmoid,sigmoid,softsign,tanh,tanhshrink
// ------------------------------------------------------------------------------------
static void noarg(S s,K x,J i) {if(!xnone(x,i))AT_ERROR("No arguments expected for ",s," module");}

using Ft = Tensor (*)(const Tensor&);
static K noarg(const char* s,Ft f, K x) {
 KTRY
  Tensor *t=xten(x); return kresult(t, f(t ? *t : kput(x)));
 KCATCH(s);
}

KAPI gelu(K x)       {return noarg("gelu",       torch::gelu,                       x);}
KAPI logsigmoid(K x) {return noarg("logsigmoid", torch::log_sigmoid,                x);}
KAPI softsign(K x)   {return noarg("softsign",   torch::nn::functional::softsign,   x);}
KAPI tanhshrink(K x) {return noarg("tanhshrink", torch::nn::functional::tanhshrink, x);}

// ------------------------------------------------------------------------------------
// activation fns with inplace flag as only arg: relu,relu6,selu
// ------------------------------------------------------------------------------------
static bool inplace(K x,J i,Cast c) {
 bool b=false; Pairs p; J n=xargc(x,i,p);
 if(n)
  TORCH_CHECK(xbool(x,i,b) && n==1, msym(c),": unrecognized option(s), expecting single boolean flag");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k)==Setting::inplace, msym(c),": unrecognized option: ",p.k);
  b=mbool(p,c);
 }
 return b;
}

static void inplace(bool a,K x,bool b) {if(a || b) OPTION(x, inplace, kb(b));}

// ------------------------------------------------------------------------------------
//  elu,celu - exponential & continuously differentiable linear unit
//             accepts optional alpha & inplace flag
// ------------------------------------------------------------------------------------
template<typename O> static O alpha(K x,J i,Cast c) {
 O o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.alpha(mdouble(x,i,c,Setting::alpha));
 } else if(n==2) {
   o.alpha(mdouble(x,i,   c, Setting::alpha));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  AT_ERROR(msym(c), ": unrecognized positional option(s), expecting alpha, inplace flag, or (alpha;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::alpha:   o.alpha(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 return o;
}

template<typename O>static void alpha(bool a,Cast c,K x,const O& o) {
 O d;
 if(a || o.alpha()   != d.alpha())   OPTION(x, alpha,   kf(o.alpha()));
 if(a || o.inplace() != d.inplace()) OPTION(x, inplace, kb(o.inplace()));
}

// ------------------------------------------------------------------------------------
//  leakyrelu - allow a small positive gradient(slope) when x<0
// ------------------------------------------------------------------------------------
static torch::nn::LeakyReLUOptions slope(K x,J i,Cast c) {
 torch::nn::LeakyReLUOptions o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.negative_slope(mdouble(x,i,c,Setting::slope));
 } else if(n==2) {
   o.negative_slope(mdouble(x, i, c, Setting::slope));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  AT_ERROR(msym(c), ": unrecognized positional option(s), expecting slope, inplace flag, or (slope;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::slope:   o.negative_slope(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 return o;
}

static void slope(bool a,Cast c,K x,const torch::nn::LeakyReLUOptions& o) {
 torch::nn::LeakyReLUOptions d;
 if(a || o.negative_slope()   != d.negative_slope()) OPTION(x, slope,   kf(o.negative_slope()));
 if(a || o.inplace()          != d.inplace())        OPTION(x, inplace, kb(o.inplace()));
}

// ------------------------------------------------------------------------------------
// hardshrink, softshrink - module/function requires single parm: lambda
// ------------------------------------------------------------------------------------
static double lambda(Cast c) {
 return c==Cast::hardshrink ? torch::nn::HardshrinkOptions().lambda() 
                            : torch::nn::SoftshrinkOptions().lambda();
}

static double lambda(K x,J i,Cast c) {
 double l=lambda(c); Pairs p; J n=xargc(x,i,p);
 if(n==1) l=mdouble(x,i,c,Setting::lambda);
 TORCH_CHECK(n<2,msym(c),": unrecognized positional option(s), expecting lambda, e.g. 0.5");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k)==Setting::lambda,"Unrecognized option: ",p.k); l=mdouble(p,c);
 }
 return l;
}

static void lambda(bool a,Cast c,K x,double l) {if(a || l != lambda(c)) OPTION(x,lambda,kf(l));}

// ------------------------------------------------------------------------------------
// glu & softmax,softmax,logsoftmax (modules only) accepts single dimension
// ------------------------------------------------------------------------------------
static int64_t dim(Cast c) { return c==Cast::glu ? torch::nn::GLUOptions().dim() : nj;}

static int64_t dim(K x,J i,Cast c) {
 int64_t d=dim(c); Pairs p; J n=xargc(x,i,p);
 if(n==1) d=int64(x,i,c,Setting::dim);
 TORCH_CHECK(n<2, msym(c),": unrecognized positional option(s), expecting single dimension");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k)==Setting::dim,"Unrecognized option: ",p.k); d=int64(p,c);
 }
 TORCH_CHECK(d!=nj, msym(c),": no dimension given");
 return d;
}

static void dim(bool a,Cast c,K x,int64_t d) {if(a || d != dim(c)) OPTION(x,dim,kj(d));}

// ----------------------------------------------------------------------------------
// softmax,softmin,logsoftmax: functional form requires dim & optional data type
// softdim: get default dimension from input tensor dimensions (deprecated)
// ----------------------------------------------------------------------------------
static J softdim(size_t d) {return !(d==0 || d==1 || d==3);}

static void softargs(K x,J i,Cast c,J &d,c10::optional<ScalarType>& s) { 
 s=c10::nullopt; Pairs p; J n=xargc(x,i,p);
 if(!((n==0 && p.n) || (xlong(x,i,d) && (n==1 || (n==2 && xtype(x,i+1,s))))))
  AT_ERROR(msym(c),": unrecognized arg(s), expecting dim or (dim;data type)");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::dim:  d=plong(p); break;
   case Setting::type: s=ptype(p); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 if(d==nj) 
  AT_ERROR("specify the dimension along which ",msym(c)," will be computed");
}

// -----------------------------------------------------------------------------------
// rrelu - randomized leaky relu, functional form has an additional flag for training
// -----------------------------------------------------------------------------------
static void rrelu(K x,J i,Cast c,bool fn,bool& tr,bool& in,double& lo,double& up) {
 Pairs p; J n=xargc(x,i,p); torch::nn::functional::RReLUFuncOptions o;
 lo=o.lower(); up=o.upper(); in=o.inplace(); tr=o.training();
 if(n) {
  if(fn) {
   TORCH_CHECK((n==1 && (xnum(x,i,lo) || xbool(x,i,tr))) ||
               (n==2 &&  xnum(x,i,lo) && (xnum(x,i+1,up) || xbool(x,i+1,tr))) ||
               (n==3 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,tr))  ||
               (n==4 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,tr) && xbool(x,i+3,in)),
               "rrelu: unexpected positional arg(s), expects (lower;upper;train flag;inplace flag)");
  } else {
   TORCH_CHECK((n==1 && (xnum(x,i,lo) || xbool(x,i,in))) ||
               (n==2 &&  xnum(x,i,lo) && (xnum(x,i+1,up) || xbool(x,i+1,in))) ||
               (n==3 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,in)),
               "rrelu: unexpected positional arg(s), expects (lower;upper;inplace flag)");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::lower:   lo=mdouble(p,c); break;
   case Setting::upper:   up=mdouble(p,c); break;
   case Setting::train:   TORCH_CHECK(fn,"rrelu: training flag not set for module"); tr=mbool(p,c);   break;
   case Setting::inplace: in=mbool(p,c);   break;
   default: AT_ERROR("rrelu option: ",p.k," not recognized");
  }
}

// return options for rrelu module
static torch::nn::RReLUOptions rrelu(K x,J i,Cast c) {
 double lo,up; bool in,tr; rrelu(x,i,c,false,tr,in,lo,up);
 return torch::nn::RReLUOptions().lower(lo).upper(up).inplace(in);
}

// retrieve options from rrelu module
static void rrelu(bool a,K x,const torch::nn::RReLUOptions& o) {
 torch::nn::RReLUOptions d;
 if(a || d.lower()   != o.lower())   OPTION(x, lower,   kf(o.lower()));
 if(a || d.upper()   != o.upper())   OPTION(x, upper,   kf(o.upper()));
 if(a || d.inplace() != o.inplace()) OPTION(x, inplace, kb(o.inplace()));
}

// -----------------------------------------------------------------------------------------
// hardtanh - computationally cheaper version of tanh, straight line at min,max
// -----------------------------------------------------------------------------------------
static torch::nn::HardtanhOptions hardtanh(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::HardtanhOptions o;
 bool b=o.inplace(); double v1=o.min_val(),v2=o.max_val();
 if(n) {
  TORCH_CHECK((n==1 && (xnum(x,i,v1) || xbool(x,i,b))) ||
              (n==2 &&  xnum(x,i,v1) && (xnum(x,i+1,v2) || xbool(x,i+1,b))) ||
              (n==3 &&  xnum(x,i,v1) &&  xnum(x,i+1,v2) && xbool(x,i+2,b)),
              "hardtanh: unexpected positional arg(s), expects (min;max;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::min:     v1=mdouble(p,c); break;
   case Setting::max:     v2=mdouble(p,c); break;
   case Setting::inplace: b=mbool(p,c); break;
   default: AT_ERROR("hardtanh option: ",p.k," not recognized");
  }
 return o.min_val(v1).max_val(v2).inplace(b);
}

static void hardtanh(bool a,K x,const torch::nn::HardtanhOptions& o) {
 torch::nn::HardtanhOptions d;
 if(a || d.min_val() != o.min_val()) OPTION(x, min,     kf(o.min_val()));
 if(a || d.max_val() != o.max_val()) OPTION(x, max,     kf(o.max_val()));
 if(a || d.inplace() != o.inplace()) OPTION(x, inplace, kb(o.inplace()));
}

// -----------------------------------------------------------------------------------------
// softplus - smooth approximation to relu, can constrain to always be positive
// -----------------------------------------------------------------------------------------
static torch::nn::SoftplusOptions softplus(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::SoftplusOptions o; double v1=o.beta(),v2=o.threshold();
 if(n) {
  TORCH_CHECK(xnum(x,i,v1) && (n==1 || (n==2 && xnum(x,i+1,v2))),
              "softplus: unexpected positional arg(s), expects (beta;threshold)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::beta:      v1=mdouble(p,c); break;
   case Setting::threshold: v2=mdouble(p,c); break;
   default: AT_ERROR("softplus option: ",p.k," not recognized");
  }
 return o.beta(v1).threshold(v2);
}

static void softplus(bool a,K x,const torch::nn::SoftplusOptions& o) {
 torch::nn::SoftplusOptions d;
 if(a || d.beta()      != o.beta())      OPTION(x, beta,      kf(o.beta()));
 if(a || d.threshold() != o.threshold()) OPTION(x, threshold, kf(o.threshold()));
}

// ----------------------------------------------------------------------------------------------
// threshold - thresholds each element of input tensor, fns set/get threshold,value,inplace flag
// ----------------------------------------------------------------------------------------------
static torch::nn::ThresholdOptions threshold(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); bool b=false; double v1=nf,v2=nf;
 if(n) {
  TORCH_CHECK((n==1 && (xnum(x,i,v1) || xbool(x,i,b))) ||
              (n==2 &&  xnum(x,i,v1) && (xnum(x,i+1,v2) || xbool(x,i+1,b))) ||
              (n==3 &&  xnum(x,i,v1) &&  xnum(x,i+1,v2) && xbool(x,i+2,b)),
              "threshold: unexpected positional arg(s), expects (threshold;value;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::threshold: v1=mdouble(p,c); break;
   case Setting::value:     v2=mdouble(p,c); break;
   case Setting::inplace:   b=mbool(p,c); break;
   default: AT_ERROR("threshold option: ",p.k," not recognized");
  }
 TORCH_CHECK(v1 == v1 && v2 == v2, "threshold: both threshold level & replacement value must be given");
 return torch::nn::ThresholdOptions(v1,v2).inplace(b);
}

static void threshold(bool a,K x,const torch::nn::ThresholdOptions& o) {
 OPTION(x, threshold, kf(o.threshold()));
 OPTION(x, value,     kf(o.value()));
 if(a || o.inplace()) OPTION(x, inplace, kb(o.inplace()));
}

// -----------------------------------------------------------------------------------------
// functional form of activation fns:
//   relu,relu6,selu (inplace flag), elu,celu(alpha & inplace), leakyrelu(slope & inplace),
//   hardshrink,softshrink(lambda), glu(dim), rrelu(lower,upper & inplace flag)
// -----------------------------------------------------------------------------------------
static K act(K x,Cast c,const char* s) {
 KTRY
  bool a,p; Tensor r,t;
  if(xten(x,t))        p=true, a=false;
  else if(xten(x,0,t)) p=true, a=true;
  else if(xmixed(x,3)) p=false,a=true, t=kput(x,0);
  else                 p=false,a=false,t=kput(x);
  switch(c) {
   case Cast::relu:  r=torch::nn::functional::relu (t,a ? inplace(x,1,c) : false); break;
   case Cast::relu6: r=torch::nn::functional::relu6(t,a ? inplace(x,1,c) : false); break;
   case Cast::selu:  r=torch::nn::functional::selu (t,a ? inplace(x,1,c) : false); break;
   case Cast::elu:   r=torch::nn::functional::elu  (t,a ? alpha<torch::nn::ELUOptions>(x,1,c) : torch::nn::ELUOptions()); break;
   case Cast::celu:  r=torch::nn::functional::celu (t,a ? alpha<torch::nn::CELUOptions> (x,1,c) : torch::nn::CELUOptions()); break;
   case Cast::leakyrelu: r=torch::nn::functional::leaky_relu(t,a ? slope(x,1,c) : torch::nn::LeakyReLUOptions()); break;
   case Cast::hardshrink: r=torch::hardshrink(t,a ? lambda(x,1,c) : lambda(c)); break;
   case Cast::softshrink: r=torch::softshrink(t,a ? lambda(x,1,c) : lambda(c)); break;
   case Cast::glu:        r=torch::nn::functional::glu(t,a ? dim(x,1,c) : dim(c)); break;
   case Cast::softmin:
   case Cast::softmax:
   case Cast::logsoftmax: {
    auto d=softdim(t.dim()); c10::optional<ScalarType> s; if(a) softargs(x,1,c,d,s);
    switch(c) {
     case Cast::softmin:    r=torch::nn::functional::detail::softmin(t,d,s); break;
     case Cast::softmax:    r=torch::nn::functional::detail::softmax(t,d,s); break;
     case Cast::logsoftmax: r=torch::nn::functional::detail::log_softmax(t,d,s); break;
     default: AT_ERROR("Unrecognized activation function");
    }
    break;
   }
   case Cast::rrelu: {
    double lo,up; bool in,tr; rrelu(a ? x : nullptr,1,c,false,tr,in,lo,up);
    r=torch::nn::functional::detail::rrelu(t,lo,up,tr,in);
    break;
   }
   case Cast::hardtanh:  r=torch::nn::functional::hardtanh (t, a ? hardtanh(x,1,c) : torch::nn::HardtanhOptions()); break;
   case Cast::softplus:  r=torch::nn::functional::softplus (t, a ? softplus(x,1,c) : torch::nn::SoftplusOptions()); break;
   case Cast::threshold: r=torch::nn::functional::threshold(t, threshold(a ? x : nullptr,1,c)); break;
   default: AT_ERROR("Unrecognized activation function"); break;
  }
  return p && r.is_same(t) ? (K)0 : kresult(p,r);
 KCATCH(s);
}

KAPI       relu(K x) {return act(x, Cast::relu,       "relu");}
KAPI      relu6(K x) {return act(x, Cast::relu6,      "relu6");}
KAPI       selu(K x) {return act(x, Cast::selu,       "selu");}
KAPI        elu(K x) {return act(x, Cast::elu,        "elu");}
KAPI       celu(K x) {return act(x, Cast::celu,       "celu");}
KAPI  leakyrelu(K x) {return act(x, Cast::leakyrelu,  "leakyrelu");}
KAPI hardshrink(K x) {return act(x, Cast::hardshrink, "hardshrink");}
KAPI softshrink(K x) {return act(x, Cast::softshrink, "softshrink");}
KAPI        glu(K x) {return act(x, Cast::glu,        "glu");}
KAPI    softmin(K x) {return act(x, Cast::softmin,    "softmin");}
KAPI    softmax(K x) {return act(x, Cast::softmax,    "softmax");}
KAPI logsoftmax(K x) {return act(x, Cast::logsoftmax, "logsoftmax");}
KAPI      Rrelu(K x) {return act(x, Cast::rrelu,      "rrelu");}
KAPI   Hardtanh(K x) {return act(x, Cast::hardtanh,   "hardtanh");}
KAPI   Softplus(K x) {return act(x, Cast::softplus,   "softplus");}
KAPI  Threshold(K x) {return act(x, Cast::threshold,  "threshold");}

// -------------------------------------------------------------------------------------------
// prelu: parameterized relu
//        module accepts 1 or number of input parameters and optional initalization value
//        functional form requires weight directly rather than module's count & initial value
// -------------------------------------------------------------------------------------------
static torch::nn::PReLUOptions prelu(K x,J i,Cast c) {
 torch::nn::PReLUOptions o; auto m=o.num_parameters();auto w=o.init(); Pairs p; J n=xargc(x,i,p);
 if(n) TORCH_CHECK((n==1 && (xint64(x,i,m) || xdouble(x,i,w))) ||
                   (n==2 &&  xint64(x,i,m) && xdouble(x,i+1,w)),
                   "prelu: expecting 1-2 positional args in,init or (in;init)");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:    m=int64(p,c); break;
   case Setting::init:  w=mdouble(p,c); break;
   default: AT_ERROR("prelu option: ",p.k," not recognized");
  }
 return o.num_parameters(m).init(w);
}

static void prelu(bool a,K x,const torch::nn::PReLUOptions& o) {
 torch::nn::PReLUOptions d;
 if(a || d.num_parameters() != o.num_parameters()) OPTION(x, in,   kj(o.num_parameters()));
 if(a || d.init()           != o.init())           OPTION(x, init, kf(o.init()));
}

KAPI Prelu(K x) {
 KTRY
  bool p; Tensor t,w;
  if(!x->t && x->n==2)
   p=xtenarg(x,t,w);
  else if(0<x->t && x->t<98 && x->n==2)
   p=false, t=kput(x), w=t[1], t=t[0];
  else
   AT_ERROR("prelu expects 2 args: input & weight, received ",kname(x->t),", count: ",xlen(x));
  return kresult(p, torch::prelu(t,w));
 KCATCH("prelu");
}

// ----------------------------------------------------------------------------------------------------
// flatten - process arg(s) from k and return options
//         - return options used given a flatten module used
//         - call flatten as function given input/tensor and optional start & end dimensions
// ----------------------------------------------------------------------------------------------------
static torch::nn::FlattenOptions flatten(K x,J i) {
 torch::nn::FlattenOptions o; int64_t s=o.start_dim(),e=o.end_dim(); Pairs p; J n=xargc(x,i,p);
 if(!(n==0 || (xint64(x,i,s) && (n==1 || (n==2 && xint64(x,i+1,e))))))
  AT_ERROR("flatten: unrecognized arg(s)");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::start: s=plong(p); break;
   case Setting::end:   e=plong(p); break;
   default: AT_ERROR("flatten option: ",p.k," not recognized");
  }
 return o.start_dim(s).end_dim(e);
}

static void flatten(bool a,K x,const torch::nn::FlattenImpl* m) {
 torch::nn::FlattenOptions d,o=m->options;
 if(a || d.start_dim() != o.start_dim()) OPTION(x, start, kj(o.start_dim()));
 if(a || d.end_dim()   != o.end_dim())   OPTION(x, end,   kj(o.end_dim()));
}

KAPI kflatten(K x) {
 KTRY
  bool m=false; Tensor t;
  auto o=flatten((xten(x,t) || xten(x,0,t) || (m=xmixed(x,3))) ? x : nullptr, 1);
  if(t.defined())
   return kten(torch::flatten(t, o.start_dim(), o.end_dim()));
  else
   return kget(torch::flatten(m ? kput(x,0) : kput(x), o.start_dim(), o.end_dim()));
 KCATCH("flatten");
}

// ----------------------------------------------------------------------------------------------------
// squeeze/unsqueeze - squeeze works with/without a dimension specified, unsqueeze requires it
// ----------------------------------------------------------------------------------------------------
static SqueezeOptions squeeze(K x,J i,Cast c) {
 SqueezeOptions o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.dim(int64n(x,i,c,Setting::dim));
 } else if(n==2) {
   o.dim(   int64n(x,i,   c, Setting::dim));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  AT_ERROR(msym(c), ": unrecognized positional arg(s), expecting dim, inplace flag, or (dim;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::dim:     o.dim(int64n(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 TORCH_CHECK(c==Cast::squeeze || o.dim().has_value(), msym(c),": no dimension given");
 return o;
}

static void squeeze(bool a,K x,const SqueezeOptions& o) {
 if(o.dim().has_value()) OPTION(x, dim,     kj(o.dim().value()));
 if(a || o.inplace())    OPTION(x, inplace, kb(o.inplace()));
}

// ----------------------------------------------------------------------------------------------------
// getsize - get size(s) for expand, reshape, ..
// expand
// reshape
// ----------------------------------------------------------------------------------------------------
static SizeOptions getsize(K x,J i,Cast c) {
 IntArrayRef a; LongVector v; Pairs p; J n=xargc(x,i,p);
 TORCH_CHECK(!n || (xsize(x,i,a) && n==1), msym(c)," expects size(s) as argument");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::size: psize(p,a); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 for(auto j:a) v.push_back(j);
 return SizeOptions(v);
}

static void getsize(bool a,K x,const SizeOptions& o) {
 OPTION(x, size, klist(o.size().size(),o.size().data()));
}

// ----------------------------------------------------------------------------------------------------
// mparms - set parameters/buffers in a defined module from k values in dictionary with matching names
// mdefine - define module and add to a sequence, reading options (and sometimes parms/buffers) from k
// ----------------------------------------------------------------------------------------------------
void mparms(S s,Module &m,K x,bool p) { // set named parms/buffers in module m from dict x, p true if parms
 K k=kK(x)[0],v=kK(x)[1]; Tensor V; if(v->t) V=kput(v);
 for(auto &a:p ? m.named_parameters() : m.named_buffers()) {
  J i=kfind(k,a.key());
  if(i<0) {
   AT_ERROR("Unable to find ",s,(p ? " parameter" : " buffer"),": ",a.key());
   break;
  }
  Tensor t=v->t ? V[i] : kput(kK(v)[i]);
  if(a.value().defined()) {
   torch::NoGradGuard g;
   if(a.value().dtype() != t.dtype())
    AT_ERROR("Type mismatch: ",s,(p ? " parameter " : " buffer "),a.key()," is ",a.value().dtype(),", input is ",t.dtype());
   if(!a.value().is_same_size(t))
    AT_ERROR("Size mismatch: ",s,(p ? " parameter " : " buffer "),a.key()," is ",a.value().sizes(),", input is ",t.sizes());
   if (a.value().device() != t.device())
    a.value().set_data(t);
   else
    a.value().set_(t);
  } else {
   a.value()=std::move(t);
  }
 }
}

void mparms(S s,Sequential &q,K p,K f) {
 J i=q->size()-1;
 if(p) mparms(s,*q[i],p,true);
 if(f) mparms(s,*q[i],f,false);
}

//s:type, n:name(optional), i:offset into x, x:options(list/dictionary), p:parms, f:buffers
void mdefine(Sequential &q,S s,S n=nullptr,J i=-1,K x=nullptr,K p=nullptr,K f=nullptr);
void mdefine(Sequential &q,S s,S n,J i,K x,K p,K f) { 
 Cast c=msym(s); Scalar v,w;
 switch(c) {
  case Cast::batchnorm:    PUSH(q,n,bnorm(x,i)); break;
  case Cast::embed:        PUSH(q,n,embed(x,i)); break;
  case Cast::linear:       PUSH(q,n,torch::nn::Linear(linear(x,i,c))); break;

  case Cast::drop:         PUSH(q,n,torch::nn::Dropout(drop(x,i,c))); break;
  case Cast::drop2d:       PUSH(q,n,torch::nn::Dropout2d(drop(x,i,c))); break;
  case Cast::drop3d:       PUSH(q,n,torch::nn::Dropout3d(drop(x,i,c))); break;
  case Cast::fdrop:        PUSH(q,n,torch::nn::FeatureDropout(drop(x,i,c))); break;
  case Cast::adrop:        PUSH(q,n,torch::nn::AlphaDropout(drop(x,i,c))); break;
  case Cast::fadrop:       PUSH(q,n,torch::nn::FeatureAlphaDropout(drop(x,i,c))); break;

  case Cast::conv1d:       PUSH(q,n,torch::nn::Conv1d(conv<1>(x,i,c))); break;
  case Cast::conv2d:       PUSH(q,n,torch::nn::Conv2d(conv<2>(x,i,c))); break;
  case Cast::conv3d:       PUSH(q,n,torch::nn::Conv3d(conv<3>(x,i,c))); break;

  case Cast::maxpool1d:    PUSH(q,n,torch::nn::MaxPool1d(maxpool<1>(x,i,c))); break;
  case Cast::maxpool2d:    PUSH(q,n,torch::nn::MaxPool2d(maxpool<2>(x,i,c))); break;
  case Cast::maxpool3d:    PUSH(q,n,torch::nn::MaxPool3d(maxpool<3>(x,i,c))); break;

  case Cast::avgpool1d:    PUSH(q,n,torch::nn::AvgPool1d(avgpool<1>(x,i,c))); break;
  case Cast::avgpool2d:    PUSH(q,n,torch::nn::AvgPool2d(avgpool<2>(x,i,c))); break;
  case Cast::avgpool3d:    PUSH(q,n,torch::nn::AvgPool3d(avgpool<3>(x,i,c))); break;

  case Cast::adaptmax1d:   PUSH(q,n,torch::nn::AdaptiveMaxPool1d(adapt<1,torch::nn::AdaptiveMaxPool1dOptions>(x,i,c))); break;
  case Cast::adaptmax2d:   PUSH(q,n,torch::nn::AdaptiveMaxPool2d(adapt<2,torch::nn::AdaptiveMaxPool2dOptions>(x,i,c))); break;
  case Cast::adaptmax3d:   PUSH(q,n,torch::nn::AdaptiveMaxPool3d(adapt<3,torch::nn::AdaptiveMaxPool3dOptions>(x,i,c))); break;

  case Cast::adaptavg1d:   PUSH(q,n,torch::nn::AdaptiveAvgPool1d(adapt<1,torch::nn::AdaptiveAvgPool1dOptions>(x,i,c))); break;
  case Cast::adaptavg2d:   PUSH(q,n,torch::nn::AdaptiveAvgPool2d(adapt<2,torch::nn::AdaptiveAvgPool2dOptions>(x,i,c))); break;
  case Cast::adaptavg3d:   PUSH(q,n,torch::nn::AdaptiveAvgPool3d(adapt<3,torch::nn::AdaptiveAvgPool3dOptions>(x,i,c))); break;

  case Cast::fmaxpool2d:   PUSH(q,n,FractionalMaxPool2d(fpool<2>(x,i,c))); break;
  case Cast::fmaxpool3d:   PUSH(q,n,FractionalMaxPool3d(fpool<3>(x,i,c))); break;

  case Cast::lppool1d:     PUSH(q,n,torch::nn::LPPool1d(lppool<1>(x,i,c))); break;
  case Cast::lppool2d:     PUSH(q,n,torch::nn::LPPool2d(lppool<2>(x,i,c))); break;

  case Cast::pad:          PUSH(q,n,Pad(pad(x,i,c))); break;
  case Cast::pad1d:        PUSH(q,n,torch::nn::ConstantPad1d(cpad<1,torch::nn::ConstantPad1dOptions>(x,i,c))); break;
  case Cast::pad2d:        PUSH(q,n,torch::nn::ConstantPad2d(cpad<2,torch::nn::ConstantPad2dOptions>(x,i,c))); break;
  case Cast::pad3d:        PUSH(q,n,torch::nn::ConstantPad3d(cpad<3,torch::nn::ConstantPad3dOptions>(x,i,c))); break;
  case Cast::reflect1d:    PUSH(q,n,torch::nn::ReflectionPad1d(npad<1,torch::nn::ReflectionPad1dOptions>(x,i,c))); break;
  case Cast::reflect2d:    PUSH(q,n,torch::nn::ReflectionPad2d(npad<2,torch::nn::ReflectionPad2dOptions>(x,i,c))); break;
  case Cast::replicate1d:  PUSH(q,n,torch::nn::ReplicationPad1d(npad<1,torch::nn::ReplicationPad1dOptions>(x,i,c))); break;
  case Cast::replicate2d:  PUSH(q,n,torch::nn::ReplicationPad2d(npad<2,torch::nn::ReplicationPad2dOptions>(x,i,c))); break;
  case Cast::replicate3d:  PUSH(q,n,torch::nn::ReplicationPad3d(npad<3,torch::nn::ReplicationPad3dOptions>(x,i,c))); break;
  case Cast::zeropad2d:    PUSH(q,n,torch::nn::ZeroPad2d(npad<2,torch::nn::ZeroPad2dOptions>(x,i,c))); break;

  case Cast::rnn:          PUSH(q,n,(rnn<torch::nn::RNN, torch::nn::RNNOptions> (s,x,i))); break;
  case Cast::gru:          PUSH(q,n,(rnn<torch::nn::GRU, torch::nn::GRUOptions> (s,x,i))); break;
  case Cast::lstm:         PUSH(q,n,(rnn<torch::nn::LSTM,torch::nn::LSTMOptions>(s,x,i))); break;

  case Cast::logsigmoid:   noarg(s,x,i); PUSH(q,n,torch::nn::LogSigmoid()); break;
  case Cast::sigmoid:      noarg(s,x,i); PUSH(q,n,torch::nn::Sigmoid()); break;
  case Cast::softsign:     noarg(s,x,i); PUSH(q,n,torch::nn::Softsign()); break;
  case Cast::softmax2d:    noarg(s,x,i); PUSH(q,n,torch::nn::Softmax2d()); break;
  case Cast::tanh:         noarg(s,x,i); PUSH(q,n,torch::nn::Tanh()); break;
  case Cast::tanhshrink:   noarg(s,x,i); PUSH(q,n,torch::nn::Tanhshrink()); break;
  case Cast::gelu:         noarg(s,x,i); PUSH(q,n,torch::nn::GELU()); break;

  case Cast::relu:         PUSH(q,n, torch::nn::ReLU(inplace(x,i,c))); break;
  case Cast::relu6:        PUSH(q,n,torch::nn::ReLU6(inplace(x,i,c))); break;
  case Cast::selu:         PUSH(q,n, torch::nn::SELU(inplace(x,i,c))); break;

  case Cast::softmax:      PUSH(q,n,torch::nn::Softmax(dim(x,i,c))); break;
  case Cast::softmin:      PUSH(q,n,torch::nn::Softmin(dim(x,i,c))); break;
  case Cast::logsoftmax:   PUSH(q,n,torch::nn::LogSoftmax(dim(x,i,c))); break;
  case Cast::flatten:      PUSH(q,n,torch::nn::Flatten(flatten(x,i))); break;
  case Cast::squeeze:      PUSH(q,n,Squeeze(squeeze(x,i,c))); break;
  case Cast::unsqueeze:    PUSH(q,n,Unsqueeze(squeeze(x,i,c))); break;
  case Cast::expand:       PUSH(q,n,Expand(getsize(x,i,c))); break;
  case Cast::reshape:      PUSH(q,n,Reshape(getsize(x,i,c))); break;

  case Cast::elu:          PUSH(q,n,torch::nn::ELU (alpha<torch::nn::ELUOptions> (x,i,c))); break;
  case Cast::celu:         PUSH(q,n,torch::nn::CELU(alpha<torch::nn::CELUOptions>(x,i,c))); break;
  case Cast::leakyrelu:    PUSH(q,n,torch::nn::LeakyReLU(slope(x,i,c))); break;
  case Cast::glu:          PUSH(q,n,torch::nn::GLU(dim(x,i,c))); break;
  case Cast::hardshrink:   PUSH(q,n,torch::nn::Hardshrink(lambda(x,i,c))); break;
  case Cast::softshrink:   PUSH(q,n,torch::nn::Softshrink(lambda(x,i,c))); break;
  case Cast::prelu:        PUSH(q,n,torch::nn::PReLU(prelu(x,i,c))); break;
  case Cast::rrelu:        PUSH(q,n,torch::nn::RReLU(rrelu(x,i,c))); break;
  case Cast::hardtanh:     PUSH(q,n,torch::nn::Hardtanh(hardtanh(x,i,c))); break;
  case Cast::softplus:     PUSH(q,n,torch::nn::Softplus(softplus(x,i,c))); break;
  case Cast::threshold:    PUSH(q,n,torch::nn::Threshold(threshold(x,i,c))); break;
  default: AT_ERROR("Unrecognized module: ",s); break;
 }
 if(p || f) mparms(s,q,p,f);  // set parms/buffers if k dictionaries supplied
}

void mdefine(Sequential &q,K x) { // define modules from k table of options or full state
 J n=x->t==99 ? 0 : xlen(x);
 for(J i=98-x->t;i<n;++i)
   mdefine(q,
    statesym(State::module,x,i),
    statesym(State::name,x,i),
    -1,
    statedict(State::options,x,i),
    statedict(State::parms,x,i),
    statedict(State::buffers,x,i));
}

// --------------------------------------------------------------------------------------------
//  functions to extract module settings and state -> q dictionary/table
// --------------------------------------------------------------------------------------------
// mopt - given module, cast at runtime to known type and extract options as k dictionary
// mget - extract module options and, optionally, parameters & buffers to k array
// mtable - extract child modules and return as k table, one row per module
// --------------------------------------------------------------------------------------------
void mopt(Module &g,bool a,K &v,J i) { //g:generic module, a:true if all options, v:k values, i:table row
 auto c=Cast::undefined;
 K x=xD(ktn(KS,0),ktn(0,0));
 if       (auto* m=g.as<torch::nn::BatchNorm>())      { c=Cast::batchnorm; bnorm(a,x,m);
 } else if(auto* m=g.as<torch::nn::Embedding>())      { c=Cast::embed;     embed(x,m);
 } else if(auto* m=g.as<torch::nn::Linear>())         { c=Cast::linear;    linear(a,x,m);

 } else if(auto* m=g.as<torch::nn::Dropout>())             { c=Cast::drop;   drop(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Dropout2d>())           { c=Cast::drop2d; drop(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Dropout3d>())           { c=Cast::drop3d; drop(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::FeatureDropout>())      { c=Cast::fdrop;  drop(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::AlphaDropout>())        { c=Cast::adrop;  drop(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::FeatureAlphaDropout>()) { c=Cast::fadrop; drop(a,x,m->options);

 } else if(auto* m=g.as<torch::nn::Conv1d>())         { c=Cast::conv1d; conv<1,torch::nn::Conv1dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::Conv2d>())         { c=Cast::conv2d; conv<2,torch::nn::Conv2dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::Conv3d>())         { c=Cast::conv3d; conv<3,torch::nn::Conv3dImpl>(a,x,m);

 } else if(auto* m=g.as<torch::nn::MaxPool1d>())      { c=Cast::maxpool1d; maxpool<1,torch::nn::MaxPool1dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::MaxPool2d>())      { c=Cast::maxpool2d; maxpool<2,torch::nn::MaxPool2dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::MaxPool3d>())      { c=Cast::maxpool3d; maxpool<3,torch::nn::MaxPool3dImpl>(a,x,m);

 } else if(auto* m=g.as<torch::nn::AvgPool1d>())      { c=Cast::avgpool1d; avgpool<1,torch::nn::AvgPool1dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::AvgPool2d>())      { c=Cast::avgpool2d; avgpool<2,torch::nn::AvgPool2dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::AvgPool3d>())      { c=Cast::avgpool3d; avgpool<3,torch::nn::AvgPool3dImpl>(a,x,m);

 } else if(auto* m=g.as<torch::nn::AdaptiveMaxPool1d>())   { c=Cast::adaptmax1d; adapt<torch::nn::AdaptiveMaxPool1dImpl>(x,m);
 } else if(auto* m=g.as<torch::nn::AdaptiveMaxPool2d>())   { c=Cast::adaptmax2d; adapt<torch::nn::AdaptiveMaxPool2dImpl>(x,m);
 } else if(auto* m=g.as<torch::nn::AdaptiveMaxPool3d>())   { c=Cast::adaptmax3d; adapt<torch::nn::AdaptiveMaxPool3dImpl>(x,m);

 } else if(auto* m=g.as<torch::nn::AdaptiveAvgPool1d>())   { c=Cast::adaptmax1d; adapt<torch::nn::AdaptiveAvgPool1dImpl>(x,m);
 } else if(auto* m=g.as<torch::nn::AdaptiveAvgPool2d>())   { c=Cast::adaptmax2d; adapt<torch::nn::AdaptiveAvgPool2dImpl>(x,m);
 } else if(auto* m=g.as<torch::nn::AdaptiveAvgPool3d>())   { c=Cast::adaptmax3d; adapt<torch::nn::AdaptiveAvgPool3dImpl>(x,m);

 } else if(auto* m=g.as<FractionalMaxPool2d>()) { c=Cast::fmaxpool2d; fpool<2,FractionalMaxPool2dImpl>(a,x,m);
 } else if(auto* m=g.as<FractionalMaxPool3d>()) { c=Cast::fmaxpool3d; fpool<3,FractionalMaxPool3dImpl>(a,x,m);

 } else if(auto* m=g.as<torch::nn::LPPool1d>())         { c=Cast::lppool1d; lppool<1,torch::nn::LPPool1dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::LPPool2d>())         { c=Cast::lppool2d; lppool<2,torch::nn::LPPool2dImpl>(a,x,m);

 } else if(auto* m=g.as<Pad>())                         { c=Cast::pad;         pad(a,x,m);
 } else if(auto* m=g.as<torch::nn::ConstantPad1d>())    { c=Cast::pad1d;       cpad(x,m);
 } else if(auto* m=g.as<torch::nn::ConstantPad2d>())    { c=Cast::pad2d;       cpad(x,m);
 } else if(auto* m=g.as<torch::nn::ConstantPad3d>())    { c=Cast::pad3d;       cpad(x,m);
 } else if(auto* m=g.as<torch::nn::ReflectionPad1d>())  { c=Cast::reflect1d;   npad(x,m);
 } else if(auto* m=g.as<torch::nn::ReflectionPad2d>())  { c=Cast::reflect2d;   npad(x,m);
 } else if(auto* m=g.as<torch::nn::ReplicationPad1d>()) { c=Cast::replicate1d; npad(x,m);
 } else if(auto* m=g.as<torch::nn::ReplicationPad2d>()) { c=Cast::replicate2d; npad(x,m);
 } else if(auto* m=g.as<torch::nn::ReplicationPad3d>()) { c=Cast::replicate3d; npad(x,m);
 } else if(auto* m=g.as<torch::nn::ZeroPad2d>())        { c=Cast::zeropad2d;   npad(x,m);

 } else if(auto* m=g.as<torch::nn::RNN>())   { c=Cast::rnn;  rnn<torch::nn::RNNImpl,  torch::nn::RNNOptions> (a,x,m);
 } else if(auto* m=g.as<torch::nn::GRU>())   { c=Cast::gru;  rnn<torch::nn::GRUImpl,  torch::nn::GRUOptions> (a,x,m);
 } else if(auto* m=g.as<torch::nn::LSTM>())  { c=Cast::lstm; rnn<torch::nn::LSTMImpl, torch::nn::LSTMOptions>(a,x,m);

 } else if(g.as<torch::nn::LogSigmoid>())    { c=Cast::logsigmoid;
 } else if(g.as<torch::nn::Sigmoid>())       { c=Cast::sigmoid;
 } else if(g.as<torch::nn::Softsign>())      { c=Cast::softsign;
 } else if(g.as<torch::nn::Softmax2d>())     { c=Cast::softmax2d;
 } else if(g.as<torch::nn::Tanh>())          { c=Cast::tanh;
 } else if(g.as<torch::nn::Tanhshrink>())    { c=Cast::tanhshrink;
 } else if(g.as<torch::nn::GELU>())          { c=Cast::gelu;

 } else if(auto* m=g.as<torch::nn::ReLU>())  { c=Cast::relu;  inplace(a,x,m->options.inplace());
 } else if(auto* m=g.as<torch::nn::SELU>())  { c=Cast::selu;  inplace(a,x,m->options.inplace());
 } else if(auto* m=g.as<torch::nn::ReLU6>()) { c=Cast::relu6; inplace(a,x,m->options.inplace());

 } else if(auto* m=g.as<torch::nn::Softmax>())    { c=Cast::softmax;    OPTION(x, dim, kj(m->options.dim()));
 } else if(auto* m=g.as<torch::nn::Softmin>())    { c=Cast::softmin;    OPTION(x, dim, kj(m->options.dim()));
 } else if(auto* m=g.as<torch::nn::LogSoftmax>()) { c=Cast::logsoftmax; OPTION(x, dim, kj(m->options.dim()));
 } else if(auto* m=g.as<torch::nn::Flatten>())    { c=Cast::flatten;    flatten(a,x,m);
 } else if(auto* m=g.as<Squeeze>())    { c=Cast::squeeze;    squeeze(a,x,m->options);
 } else if(auto* m=g.as<Unsqueeze>())  { c=Cast::unsqueeze;  squeeze(a,x,m->options);
 } else if(auto* m=g.as<Expand>())     { c=Cast::expand;     getsize(a,x,m->options);
 } else if(auto* m=g.as<Reshape>())    { c=Cast::reshape;    getsize(a,x,m->options);

 } else if(auto* m=g.as<torch::nn::ELU>())        { c=Cast::elu;  alpha(a,c,x,m->options);
 } else if(auto* m=g.as<torch::nn::CELU>())       { c=Cast::celu; alpha(a,c,x,m->options);
 } else if(auto* m=g.as<torch::nn::LeakyReLU>())  { c=Cast::leakyrelu;  slope(a,c,x,m->options);
 } else if(auto* m=g.as<torch::nn::GLU>())        { c=Cast::glu;        dim(a,c,x,m->options.dim());
 } else if(auto* m=g.as<torch::nn::Hardshrink>()) { c=Cast::hardshrink; lambda(a,c,x,m->options.lambda());
 } else if(auto* m=g.as<torch::nn::Softshrink>()) { c=Cast::softshrink; lambda(a,c,x,m->options.lambda());

 } else if(auto* m=g.as<torch::nn::PReLU>())      { c=Cast::prelu;      prelu(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::RReLU>())      { c=Cast::rrelu;      rrelu(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Hardtanh>())   { c=Cast::hardtanh;   hardtanh(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Softplus>())   { c=Cast::softplus;   softplus(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Threshold>())  { c=Cast::threshold;  threshold(a,x,m->options);

 } else { AT_ERROR("Unrecognized module: ",g.name());
 }
 S s=msym(c);J j=v->n==3 ? 0 : 1;
 if(i<0)    kK(v)[j]=ks(s),    kK(v)[j+2]=x;     //dictionary, assign module & options
 else    kS(kK(v)[j])[i]=s, kK(kK(v)[j+2])[i]=x; //table, assign module,options in i'th row
}

void mget(const char* s,Module &m,bool a,K &v,J i) {
 //s:name in sequence, m:type-erased module, a:true for all options, v:array for values, i:i'th row of table result
 bool b=v->n==6;
 mopt(m,a,v,i);
 if(i<0) {                               //fill in dictionary values[1 3 4], name,parms,buffers
  kK(v)[1+b]=ks(cs(s));
  if(b) {
   kK(v)[4]=kdict(m.named_parameters());
   kK(v)[5]=kdict(m.named_buffers());
  }
 } else {
  kS(kK(v)[1+b])[i]=cs(s);                //fill in i'th table row
  if(b) {
   kK(kK(v)[4])[i]=kdict(m.named_parameters());
   kK(kK(v)[5])[i]=kdict(m.named_buffers());
  }
 }
}

K mtable(const Sequential& q,bool a,bool b) {
 J i=0; K k=mkeys(b),v=mvals(b,q->size());
 for(const auto&c:q->named_children()) mget(c.key().c_str(),*c.value(),a,v,i++);
 return xT(xD(k,v));
}

// --------------------------------------------------------------------------------------
// tchild - extract named parameter/buffer tensor from child module in sequential
// mchild - extract child module by index/name, return module state or individual tensor
// --------------------------------------------------------------------------------------
static K tchild(S s,const Module& c) {
 Tensor t;
 if(c.named_parameters().contains(s))
  t=c.named_parameters()[s];
 else if(c.named_buffers().contains(s))
  t=c.named_buffers()[s];
 else
  AT_ERROR("No parameter or buffer named: ",s);
 return kget(t);
}

static K mchild(bool a,J i,S s,const Sequential &q) {
 if(i<0 || (unsigned)i>=q->size())
  AT_ERROR("Invalid module index: ",i);
 if(s) {
  return tchild(s,*q->children()[i]);
 } else {
  K k=mkeys(true),v=mvals(true,-1);
  //direct access by index[0] fails to pick up name(?)
  //const auto& c=q->named_children()[i];
  //mget(c.key().c_str(),*c.value(),a,v,-1);
  const auto& c=q->named_children();
  mget(c.keys()[i].c_str(),*c.values()[i],a,v,-1);
  return xD(k,v);
 }
}

static K mchild(bool a,S s1,S s2,const Sequential &q) {
 const auto& m=q->named_children()[s1];
 if(s2) {
  return tchild(s2,*m);
 } else {
  K k=mkeys(true),v=mvals(true,-1);
  mget(s1,*m,a,v,-1);
  return xD(k,v);
 }
}

// ------------------------------------------------------------------------------------------
//  main api functions defined in k
// ------------------------------------------------------------------------------------------
// margs - helper function used to parse module creation args (if not table/dictionary)
// seq - create/append sequential module
// mstate - class,module,name,options,parms,buffers for module(s) or selected parm/buffer
// seqforward - given sequential module and input, run forward calcs, return tensor to k
// ------------------------------------------------------------------------------------------
static void margs(bool p,Sequential q,K x) {
 J i=p; S s=nullptr,nm=nullptr;
 if(xsym(x,s) || xsym(x,i,s)) {
  if(xsym(x,i+1,nm)) i++;
  mdefine(q,s,nm,i+1,x);
 } else if(x->t == KS) {
  if(x->n==1 || x->n==2) {
   s=kS(x)[0];
   if(x->n==2) nm=kS(x)[1];
   mdefine(q,s,nm,x->n,x);
  } else {
   AT_ERROR("Unable to process list of ",x->n," symbols");
  }
 } else if(x->t==0 || (p && kK(x)[1]->t==0)) {
  K y=p ? kK(x)[1] : x;
  if(y->t)
   margs(false,q,y);
  else
   for(J j=0;j<y->n;++j) margs(false,q,kK(y)[j]);
 } else {
   AT_ERROR("Unrecognized module arg(s): ",kname(x->t)," supplied");
 }
}

KAPI seq(K x) {
 KTRY
  Sequential q,u; bool a=env().alloptions,p=xseq(x,0,q);
  if(xempty(x)) {
   return kseq(q);
  } else if(xseq(x,q) || (p && x->n==2 && xbool(x,1,a))) {
   return mtable(q,a,false);
  } else if(p && x->n==2 && xseq(x,1,u)) {
    return q->extend(*u), (K)0;
  } else if(xstate(x) || (p && x->n==2 && xstate(x,1))) {
   return mdefine(q,p ? kK(x)[1] : x), p ? (K)0 : kseq(q);
  } else {
   return margs(p,q,x), p ? (K)0 : kseq(q);
  }
 KCATCH("Sequential module");
}

K mstate(K x) {
 bool a=env().alloptions; S s1=nullptr,s2=nullptr; J i; Sequential q;
 if(xseq(x,q) || (xbool(x,1,a) && x->n==2 && xseq(x,0,q))) {
  return mtable(q,a);
 } else if(xseq(x,0,q)
   && (xsym(x,1,s1) || xlong(x,1,i)) 
   && (x->n==2 || (x->n==3 && (xsym(x,2,s2) || xbool(x,2,a))))) {
  return s1 ? mchild(a,s1,s2,q) : mchild(a,i,s2,q);
 } else {
  return KERR("Unexpected arg(s) for module state");
 }
}

K seqforward(Sequential& q,K x) {
 TORCH_CHECK(!x->t && x->n==2, "forward expects two args: sequential/model & input tensor or array");
 Tensor *t=xten(x,1);
 return kten(q->forward(t ? *t : kput(x,1)));
}

// ---------------------------------------------------------------------------------------
// seqattr - return requested attribute of given sequential module
// ---------------------------------------------------------------------------------------
K seqattr(const Sequential& q,A k,Attr a) {
 switch(a) {
  case Attr::ptr:     return kj((intptr_t)q.get());
  case Attr::ref:     return kj(q.ptr().use_count());
  default: AT_ERROR(mapattr(a),": not implemented for sequential module");
 }
}

// ----------------------------------
// module fns defined in k namespace
// ----------------------------------
void nnfn(K x) {
 fn(x, "seq",        KFN(seq), 1);            // api function for module create/query
 fn(x, "adaptavg1d", KFN(adaptavg1d),  1);    // functional form of modules/activations
 fn(x, "adaptavg2d", KFN(adaptavg2d),  1);
 fn(x, "adaptavg3d", KFN(adaptavg3d),  1);
 fn(x, "adaptmax1d", KFN(adaptmax1d),  1);
 fn(x, "adaptmax2d", KFN(adaptmax2d),  1);
 fn(x, "adaptmax3d", KFN(adaptmax3d),  1);
 fn(x, "avgpool1d",  KFN(avgpool1d),   1);
 fn(x, "avgpool2d",  KFN(avgpool2d),   1);
 fn(x, "avgpool3d",  KFN(avgpool3d),   1);
 fn(x, "celu",       KFN(celu),        1);
 fn(x, "elu",        KFN(elu),         1);
 fn(x, "flatten",    KFN(kflatten),    1);
 fn(x, "glu",        KFN(glu),         1);
 fn(x, "hardshrink", KFN(hardshrink),  1);
 fn(x, "hardtanh",   KFN(Hardtanh),    1);
 fn(x, "leakyrelu",  KFN(leakyrelu),   1);
 fn(x, "linear",     KFN(klinear),     1);
 fn(x, "logsigmoid", KFN(logsigmoid),  1);
 fn(x, "logsoftmax", KFN(logsoftmax),  1);
 fn(x, "lppool1d",   KFN(lppool1d),    1);
 fn(x, "lppool2d",   KFN(lppool2d),    1);
 fn(x, "maxpool1d",  KFN(maxpool1d),   1);
 fn(x, "maxpool2d",  KFN(maxpool2d),   1);
 fn(x, "maxpool3d",  KFN(maxpool3d),   1);
 fn(x, "prelu",      KFN(Prelu),       1);
 fn(x, "gelu",       KFN(gelu),        1);
 fn(x, "relu",       KFN(relu),        1);
 fn(x, "relu6",      KFN(relu6),       1);
 fn(x, "rrelu",      KFN(Rrelu),       1);
 fn(x, "selu",       KFN(selu),        1);
 fn(x, "softmax",    KFN(softmax),     1);
 fn(x, "softmin",    KFN(softmin),     1);
 fn(x, "softplus",   KFN(Softplus),    1);
 fn(x, "softsign",   KFN(softsign),    1);
 fn(x, "softshrink", KFN(softshrink),  1);
 fn(x, "tanhshrink", KFN(tanhshrink),  1);
 fn(x, "threshold",  KFN(Threshold),   1);
}

KAPI anytest(K x) {
 Sequential q(
  torch::nn::AdaptiveAvgPool1d(2),
  torch::nn::AdaptiveAvgPool2d(2),
  torch::nn::AdaptiveAvgPool3d(2),
  torch::nn::AdaptiveMaxPool1d(2),
  torch::nn::AdaptiveMaxPool2d(2),
  torch::nn::AdaptiveMaxPool3d(2),
  torch::nn::AvgPool1d(2),
  torch::nn::AvgPool2d(2),
  torch::nn::AvgPool3d(2),
  torch::nn::BatchNorm(5),
  torch::nn::CELU(torch::nn::CELUOptions().alpha(1.0)),
  torch::nn::Conv1d(1,2,3),
  torch::nn::Conv2d(1,2,3),
  torch::nn::Conv3d(1,2,3),
  torch::nn::Dropout(.5),
  torch::nn::FeatureDropout(.5),
  torch::nn::FeatureDropout(),
  torch::nn::FeatureDropout(.2),
  torch::nn::AlphaDropout(.5),
  torch::nn::AlphaDropout(),
  torch::nn::AlphaDropout(.2),
  torch::nn::FeatureAlphaDropout(.5),
  torch::nn::FeatureAlphaDropout(),
  torch::nn::FeatureAlphaDropout(.25),
  torch::nn::ELU(),
  torch::nn::Embedding(4,10),
  torch::nn::FeatureDropout(.5),
  FractionalMaxPool2d(FractionalMaxPoolOptions<2>(3).ratio(.5)),
  FractionalMaxPool3d(FractionalMaxPoolOptions<3>(3).ratio(.5)),
  torch::nn::GLU(),
  torch::nn::GLU(2),
  torch::nn::GRU(4,5),
  torch::nn::Hardshrink(.5),
  torch::nn::Hardtanh(),
  torch::nn::LSTM(4,5),
  torch::nn::LeakyReLU(),
  torch::nn::Linear(3,4),
  torch::nn::LogSigmoid(),
  torch::nn::LogSoftmax(1), //,torch::kDouble),
  torch::nn::LPPool1d(2,3),
  torch::nn::LPPool2d(2,3),
  torch::nn::MaxPool1d(2),
  torch::nn::MaxPool2d(2),
  torch::nn::MaxPool3d(2),
  torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(1)),
  Pad(LongVector{1,1}),
  torch::nn::RNN(4,5),
  torch::nn::RReLU(),
  torch::nn::GELU(),
  torch::nn::ReLU(),
  torch::nn::ReLU(true),
  torch::nn::ReLU6(),
  torch::nn::ReLU6(true),
  torch::nn::ReflectionPad1d(2),
  torch::nn::ReflectionPad2d(2),
  torch::nn::ReplicationPad1d(2),
  torch::nn::ReplicationPad2d(2),
  torch::nn::ReplicationPad3d(2),
  torch::nn::SELU(),
  torch::nn::SELU(true),
  torch::nn::Sigmoid(),
  torch::nn::Softsign(),
  torch::nn::Softmax(-1),
  torch::nn::Softmin(1),
  torch::nn::Softplus(),
  torch::nn::Softshrink(.5),
  torch::nn::Tanh(),
  torch::nn::Tanhshrink(),
  torch::nn::Threshold(.1,20),
  torch::nn::Flatten(),
  //torch::nn::Flatten(1),
  //torch::nn::Flatten(1,-1),
  Squeeze(),
  Squeeze(SqueezeOptions().inplace(true)),
  Squeeze(-1),
  Squeeze(-1,true),
  Unsqueeze(-1),
  Unsqueeze(-1,true),
  Unsqueeze(-1,false)
 );
 return kseq(q);
}
