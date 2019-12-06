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

/*
template<size_t D,typename M>
static M conv(K x,J k) {
 bool b=true,t=false; Pairs p; size_t d; J i=-1,j=-1,g=1,n=xargc(x,k,p);
 ExpandingArray<D> sz(-1),st(1),pd(0),po(0),dl(1);
 if(!((!n && p.n) || (xlong(x,k,i) && (n==1 || (xlong(x,k+1,j) && (n==2 || (n==3 && XDIM(x,k+2,D,sz))))))))
  AT_ERROR("Unrecognized arguments for conv",D,"d module");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:        i=plong(p); break;
   case Setting::out:       j=plong(p); break;
   case Setting::size:      PDIM(p,D,sz); break;
   case Setting::stride:    PDIM(p,D,st); break;
   case Setting::pad:       PDIM(p,D,pd); break;
   case Setting::outpad:    PDIM(p,D,po); break;
   case Setting::dilate:    PDIM(p,D,dl); break;
   case Setting::groups:    g=plong(p); break;
   case Setting::bias:      b=pbool(p); break;
   case Setting::transpose: t=pbool(p); break;
   default: AT_ERROR("Unrecognized convolution option: ",p.k); break;
  }
 if(i<0) {
  AT_ERROR("number of channels in the input must be set, currently in = ",i);
 } else if(j<0) {
  AT_ERROR("number of channels produced from the convolution must be set, currently out = ",j);
 } else {
  for(d=0;d<D;d++)
   if((*sz)[d]<0) AT_ERROR("Size of ",D,"-d colvolutional kernel, dim[",d,"] = ",(*sz)[d]);
 }
 using O=torch::nn::ConvOptions<D>;
 return M(O(i,j,sz).with_bias(b).transposed(t).groups(g).stride(st).padding(pd).output_padding(po).dilation(dl));
}
*/

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
static double drop(S s,K x,J i) {
 double f=torch::nn::DropoutOptions().p(); Pairs p; J n=xargc(x,i,p);
 if(!(n==0 || (n==1 && xdouble(x,i,f))))
  AT_ERROR("Unrecognized arguments for dropout module: ",s);
 while(xpair(p))
  if(mset(p.k)==Setting::drop)
   f=pdouble(p);
  else
   AT_ERROR("Dropout option: ",p.k," unrecognized, expected option: drop, with a probability from 0.0 to 1.0");
 return f;
}

static void drop(bool a,K x,double f) {
 double p=torch::nn::DropoutOptions().p();
 if(a || p != f) OPTION(x, drop, kf(f));
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
   default: AT_ERROR("Unrecognized avg pooling function");
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
template<size_t D> static LPPoolOptions<D> lppool(K x,J i,Cast c) {
 LPPoolOptions<D> o(0,0);
 bool pw=false,sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.power      (mdouble(x,i+j,  c,Setting::power));   pw=true; break;
    case 1: o.kernel_size(exarray<D>(x,i+j,c,Setting::size));   sz=true; break;
    case 2: o.stride     (exarray<D>(x,i+j,c,Setting::stride)); st=true; break;
    case 3: o.ceil_mode  (mbool    (x,i+j,c,Setting::ceiling)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::power:   o.power      (mdouble   (p,c)); pw=true; break;
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
 LPPoolOptions<D> o=m->options, d(o.power(),o.kernel_size());
 OPTION(x, power, kf(o.power()));
 OPTION(x, size,  KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   OPTION(x, stride,  KEX(o.stride()));
 if(a || o.ceil_mode() != d.ceil_mode()) OPTION(x, ceiling, kb(o.ceil_mode()));
}

// no torch::nn::functional:lp_pool implemented yet (for version 1.3??)

// ----------------------------------------------------------------------------------
// pad - n-dimensional padding, specify even number of sizes and optional pad value
// ----------------------------------------------------------------------------------
static Pad pad(K x,J i) {
 IntArrayRef a; Scalar s=PadOptions().value(); Pairs p; J n=xargc(x,i,p); LongVector v;
 if(!((n==0 && p.n) || (xsize(x,i,a) && (n==1 || (n==2 && xnum(x,i+1,s))))))
  AT_ERROR("Unrecognized arguments for padding module");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::pad:    psize(p,a); break;
   case Setting::value:  pnum(p,s); break;
   default: AT_ERROR("padding option: ",p.k," not recognized");
  }
 if(a.size()>0 && !(a.size() % 2)) {
  for(auto j:a) v.push_back(j);
 } else {
  AT_ERROR(a.size()," pad size(s) supplied, expecting pairs for left,right or left,right,top,bottom.. etc");
 }
 return Pad(PadOptions(v).value(s));
}

static void pad(bool a,K x,const PadImpl* m) {
 auto& p=m->options.pad(); 
 OPTION(x, pad, klist(p.size(),p.data()));
 if(a || !match(PadOptions().value(),m->options.value()))
  OPTION(x, value, kscalar(m->options.value()));
}

// ----------------------------------------------------------------------------------
// rpad - reflect/replicate fixed dimension padding
// ----------------------------------------------------------------------------------
template<size_t D,typename M> static M rpad(K x,J i,Cast c) {
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

template<typename M> static void rpad(K x,const M* m) {
 OPTION(x, pad, KEX(m->options.padding()));
}

// ----------------------------------------------------------------------------------
//  softmax, softmin, logsoftmax layers
// ----------------------------------------------------------------------------------
static J softdim(size_t d) {return !(d==0 || d==1 || d==3);}

static void softargs(const char* s,K x,J i,J &d,c10::optional<ScalarType>& t) { 
 t=c10::nullopt; Pairs p; J n=xargc(x,i,p);
 if(!((n==0 && p.n) || (xlong(x,i,d) && (n==1 || (n==2 && xtype(x,i+1,t))))))
  AT_ERROR("Unrecognized arguments for ",s,", expecting dim or (dim;type)");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::dim:  d=plong(p); break;
   case Setting::type: t=ptype(p); break;
   default: AT_ERROR("Unrecognized ",s," option: ",p.k); break;
  }
 if(d==nj) 
  AT_ERROR("specify the dimension along which ",s," will be computed");
}

template<typename M>
static M soft(S s,K x,J i) {J d=nj; c10::optional<ScalarType> t; softargs(s,x,i,d,t); return M(d,t);}

template<typename M>
static void soft(bool a,K x,const M *m) {
 SoftOptions o=m->options, d(o.dim());
 OPTION(x, dim, kj(o.dim()));
 if((a || o.dtype() != d.dtype()) && o.dtype())
  OPTION(x, type, ks(stype(o.dtype())));
}

using SoftFn = Tensor (*)(const Tensor&, int64_t, c10::optional<ScalarType>);

static Tensor softmin(const Tensor& a,int64_t d,c10::optional<ScalarType> t) {
 return torch::softmax(-a,d,t);
}

static K soft(K x,const char* s,SoftFn f) {
 KTRY
  J d; c10::optional<ScalarType> t=c10::nullopt; Tensor a;
  if(xten(x,a)) {
    return kten(f(a,softdim(a.dim()),t));
  } else if(xten(x,0,a)) {
    d=softdim(a.dim()); softargs(s,x,1,d,t);
    return kten(f(a,d,t));
  } else if(xmixed(x,3)) {
    a=kput(kK(x)[0]); d=softdim(a.dim()); softargs(s,x,1,d,t);
    return kget(f(a,d,t));
  } else {
    a=kput(x);
    return kget(f(a,softdim(a.dim()),t));
  }
 KCATCH(s);
}

KAPI ksoftmin   (K x) {return soft(x, "softmin",    softmin);}
KAPI ksoftmax   (K x) {return soft(x, "softmax",    torch::softmax);}
KAPI klogsoftmax(K x) {return soft(x, "logsoftmax", torch::log_softmax);}

// ------------------------------------------------------------------------------------
// noarg:  activation functions without arguments or only inplace=true/false
//         logsigmoid,sigmoid,softsign,tanh,tanhshrink,relu,relu6,selu
// ------------------------------------------------------------------------------------
static void noarg(S s,K x,J i) {if(!xnone(x,i))AT_ERROR("No arguments expected for ",s," module");}

using Ft = Tensor (*)(const Tensor&);

static K noarg(const char* s,Ft f, K x) {
 KTRY
  Tensor t; bool p=xten(x,t);
  return kresult(p, f(p ? t : kput(x)));
 KCATCH(s);
}

static Tensor relu6(     const Tensor &t) {return torch::hardtanh(t,0.0,6.0);}
static Tensor softsign(  const Tensor &t) {return t/(t.abs()+1);}
static Tensor tanhshrink(const Tensor &t) {return t-t.tanh();}

KAPI kgelu(K x)       {return noarg("gelu",       torch::gelu,        x);}
KAPI krelu(K x)       {return noarg("relu",       torch::relu,        x);}
KAPI krelu6(K x)      {return noarg("relu6",      relu6,              x);}
KAPI kselu(K x)       {return noarg("selu",       torch::selu,        x);}
KAPI klogsigmoid(K x) {return noarg("logsigmoid", torch::log_sigmoid, x);}
KAPI ksoftsign(K x)   {return noarg("softsign",   softsign,           x);}
KAPI ktanhshrink(K x) {return noarg("tanhshrink", tanhshrink,         x);}

// ------------------------------------------------------------------------------------
//     activation functions with a single numeric scalar option
// ------------------------------------------------------------------------------------
// default1 - set default scalar value given function enumeration
// arg1     - process x for scalar, e.g. (t;.5) or (t;(`alpha;.5))
// setting1 - given scalar, compare with default and set entry in k dictionary
// fn1      - call activation directly (no module)
// ------------------------------------------------------------------------------------
static void default1(Cast c,Setting &k,Scalar &v) {  // given module type, get setting & default value
 switch(c) {
  case Cast::glu:        k=Setting::dim;    v=GLUOptions().dim(); break;
  case Cast::elu:
  case Cast::celu:       k=Setting::alpha;  v=ExpOptions().alpha(); break;
  case Cast::leakyrelu:  k=Setting::slope;  v=LeakyOptions().slope(); break;
  case Cast::hardshrink:
  case Cast::softshrink: k=Setting::lambda; v=ShrinkOptions().lambda(); break;
  default: AT_ERROR("Unexpected module type: ",(I)c," unable to get default scalar setting");
 }
}

static void arg1(Cast c,const char* s,K x,J i,Scalar& v) { // check argument(s) for single numeric scalar or named, e.g. (`alpha;.01)
 Pairs p; Setting k; J n=xargc(x,i,p); default1(c,k,v);
 if(!(n || p.n)) return;
 if(!(n==0 || (n==1 && xnum(x,i,v))))
  AT_ERROR("Unrecognized argument(s) for ",s," module");
 while(xpair(p))
  if(mset(p.k) == k)
   pnum(p,v);
  else
   AT_ERROR("Unrecognized option: ",p.k,", ",s," module expects single scalar option: ",mset(k));
 if(c==Cast::glu && !v.isIntegral(false))
  AT_ERROR("Dimension for gated linear unit must be given as an integer");
}

static void setting1(bool a,Cast c,K x,const Scalar &w) {
 Setting k; Scalar v; default1(c,k,v);
 if(a || !match(v,w)) dictadd(x,mset(k),kscalar(w));
}

using Fts = Tensor (*)(const Tensor&, Scalar);

static K fn1(Cast c,const char* s,K x,Fts f) {
 KTRY
  Tensor t; Setting k; Scalar v;
  if(xten(x,t))
   return default1(c,k,v), kten(f(t,v));
  else if(xten(x,0,t) || xmixed(x,2))
   return arg1(c,s,x,1,v), t.defined() ? kten(f(t,v)) : kget(f(kput(x,0),v));
  else
   return default1(c,k,v), kget(f(kput(x),v));
 KCATCH(s);
}

static Tensor elu(const Tensor& t,Scalar v) {return torch::elu(t,v);}          // torch fn has two additional scalars
static Tensor glu(const Tensor& t,Scalar v) {return torch::glu(t,v.toLong());} // torch fn uses int64_t for dimension

KAPI kelu(K x)        {return fn1(Cast::elu,        "elu",        x, elu);}
KAPI kglu(K x)        {return fn1(Cast::glu,        "glu",        x, glu);}
KAPI kcelu(K x)       {return fn1(Cast::celu,       "celu",       x, torch::celu);}
KAPI kleakyrelu(K x)  {return fn1(Cast::leakyrelu,  "leakyrelu",  x, torch::leaky_relu);}
KAPI khardshrink(K x) {return fn1(Cast::hardshrink, "hardshrink", x, torch::hardshrink);}
KAPI ksoftshrink(K x) {return fn1(Cast::softshrink, "softshrink", x, torch::softshrink);}

// ------------------------------------------------------------------------------------
//      activation functions with up to two scalars, e.g. min/max, lower/upper
// ------------------------------------------------------------------------------------
// default2 - set default scalar values given function enumeration
// arg2 - process x for scalar(s), e.g. (t;-1;1) or (t;(`min,-1;`max,1))
//        2 versions, one for extra training flag arg for functional form of rrelu()
// setting2 - given scalars, compare with defaults and set entries in k dictionary
// fn2      - call activation function directly (no module)
// ------------------------------------------------------------------------------------
static void default2(Cast c,Setting& k1,Setting& k2,Scalar& v1,Scalar& v2) { //given cast, set keys & default values
 switch(c) {
  case Cast::prelu:
   k1=Setting::in;                    k2=Setting::init;
   v1=PReLUOptions().in();            v2=PReLUOptions().init(); break;
  case Cast::rrelu:
   k1=Setting::lower;                 k2=Setting::upper;
   v1=RReLUOptions().lower();         v2=RReLUOptions().upper(); break;
  case Cast::hardtanh:
   k1=Setting::min;                   k2=Setting::max;
   v1=HardtanhOptions().min();        v2=HardtanhOptions().max(); break;
  case Cast::softplus:
   k1=Setting::beta;                  k2=Setting::threshold;
   v1=SoftplusOptions().beta();       v2=SoftplusOptions().threshold(); break;
  case Cast::threshold:
   k1=Setting::threshold;             k2=Setting::value; 
   v1=ThresholdOptions().threshold(); v2=ThresholdOptions().value(); break;
  default: AT_ERROR("Unexpected module type: ",(I)c," unable to get default settings");
 }
}

static void arg2(bool r,Cast c,const char* s,K x,J i,bool &b,Scalar& v1,Scalar& v2) {
 // r:true for functional rrelu, b:training flag  (only used for functional rrelu)
 Pairs p; Setting e,k1,k2; J n=xargc(x,i,p); b=false; default2(c,k1,k2,v1,v2);
 if(!(n || p.n)) return;
 if(r && xbool(x,n+i-1,b)) n--;
 if(!(n==0 || (xnum(x,i,v1) && (n==1 || (n==2 && xnum(x,i+1,v2))))))
  AT_ERROR("Unrecognized argument(s) for ",s);
 while(xpair(p)) {
  e=mset(p.k);
  if(e==k1)
   pnum(p,v1);
  else if(e==k2)
   pnum(p,v2);
  else if(r && e==Setting::train)
   b=pbool(p);
  else
   AT_ERROR("Unrecognized option: ",p.k,", ",s," expects options: ",mset(k1),",",mset(k2));
 }
 if(c==Cast::prelu && !v1.isIntegral(false))
  AT_ERROR("Parameterized ReLU expects number of learnable parameters to be integer, either 1 or number of inputs");
}

// version for modules without special handling for function version of rrelu()
static void arg2(Cast c,const char* s,K x,J i,Scalar& v1,Scalar& v2) {
 bool b; arg2(false,c,s,x,i,b,v1,v2);
}

static void setting2(bool a,Cast c,K x,const Scalar &w1,const Scalar& w2) {
 Setting k1,k2; Scalar v1,v2; default2(c,k1,k2,v1,v2);
 if(a || !match(v1,w1)) dictadd(x,mset(k1),kscalar(w1));
 if(a || !match(v2,w2)) dictadd(x,mset(k2),kscalar(w2));
}

using Ftss = Tensor (*)(const Tensor&, Scalar, Scalar);

static K fn2(Cast c,const char* s,K x,Ftss f) {
 KTRY
  Tensor t; bool b; Setting k1,k2; Scalar v1,v2;
  if(xten(x,t)) {                                                     // tensor w'out any other args
   default2(c,k1,k2,v1,v2);                                           // set defaults for scalars
   return kten(f(t,v1,v2));                                           // return tensor ptr to result
  } else if(xten(x,0,t) || xmixed(x,4)) {                             // arg(s) detected
   bool p=t.defined(), r=c==Cast::rrelu;
   arg2(r,c,s,x,1,b,v1,v2);                                           // check that args are valid
   if(!p) t=kput(x,0);                                                // if no tensor ptr, tensor from array
   Tensor a = r ? torch::rrelu(t,v1,v2,b) : f(t,v1,v2);               // special handling for rrelu w'train=true
   return kresult(p,a);
  } else {
   default2(c,k1,k2,v1,v2);                                           // no tensor ptr, use defaults
   return kget(f(kput(x),v1,v2));                                     // return k value from given k input
  }
 KCATCH(s);
}

Tensor rrelu(const Tensor& t,Scalar a,Scalar b) {return torch::rrelu(t,a,b);}  // rrelu, train=false

KAPI krrelu(K x)     {return fn2(Cast::rrelu,     "rrelu",     x, rrelu);}
KAPI khardtanh(K x)  {return fn2(Cast::hardtanh,  "hardtanh",  x, torch::hardtanh);}
KAPI ksoftplus(K x)  {return fn2(Cast::softplus,  "softplus",  x, torch::softplus);}
KAPI kthreshold(K x) {return fn2(Cast::threshold, "threshold", x, torch::threshold);}

// parameterized relu as function requires weight directly rather than module's count & initial value
KAPI kprelu(K x) {
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
static FlattenOptions flatten(K x,J i) {
 FlattenOptions o; int64_t s=o.start_dim(),e=o.end_dim(); Pairs p; J n=xargc(x,i,p);
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

static void flatten(bool a,K x,const FlattenImpl* m) {
 FlattenOptions d,o=m->options;
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

  case Cast::dropout:      PUSH(q,n,torch::nn::Dropout(drop(s,x,i))); break;
  case Cast::fdropout:     PUSH(q,n,torch::nn::FeatureDropout(drop(s,x,i))); break;
  case Cast::adropout:     PUSH(q,n,AlphaDropout(drop(s,x,i))); break;
  case Cast::fadropout:    PUSH(q,n,FeatureAlphaDropout(drop(s,x,i))); break;

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

  case Cast::lppool1d:     PUSH(q,n,LPPool1d(lppool<1>(x,i,c))); break;
  case Cast::lppool2d:     PUSH(q,n,LPPool2d(lppool<2>(x,i,c))); break;

  case Cast::pad:          PUSH(q,n,(pad(x,i))); break;
  case Cast::reflect1d:    PUSH(q,n,ReflectionPad1d(rpad<1,torch::nn::ReflectionPad1dOptions>(x,i,c))); break;
  case Cast::reflect2d:    PUSH(q,n,ReflectionPad2d(rpad<2,torch::nn::ReflectionPad2dOptions>(x,i,c))); break;
  case Cast::replicate1d:  PUSH(q,n,ReplicationPad1d(rpad<1,torch::nn::ReplicationPad1dOptions>(x,i,c))); break;
  case Cast::replicate2d:  PUSH(q,n,ReplicationPad2d(rpad<2,torch::nn::ReplicationPad2dOptions>(x,i,c))); break;
  case Cast::replicate3d:  PUSH(q,n,ReplicationPad3d(rpad<3,torch::nn::ReplicationPad3dOptions>(x,i,c))); break;

  case Cast::rnn:          PUSH(q,n,(rnn<torch::nn::RNN, torch::nn::RNNOptions> (s,x,i))); break;
  case Cast::gru:          PUSH(q,n,(rnn<torch::nn::GRU, torch::nn::GRUOptions> (s,x,i))); break;
  case Cast::lstm:         PUSH(q,n,(rnn<torch::nn::LSTM,torch::nn::LSTMOptions>(s,x,i))); break;

  case Cast::logsigmoid:   noarg(s,x,i); PUSH(q,n,LogSigmoid()); break;
  case Cast::sigmoid:      noarg(s,x,i); PUSH(q,n,Sigmoid()); break;
  case Cast::softsign:     noarg(s,x,i); PUSH(q,n,Softsign()); break;
  case Cast::tanh:         noarg(s,x,i); PUSH(q,n,Tanh()); break;
  case Cast::tanhshrink:   noarg(s,x,i); PUSH(q,n,Tanhshrink()); break;
  case Cast::gelu:         noarg(s,x,i); PUSH(q,n,GELU()); break;
  case Cast::relu:         noarg(s,x,i); PUSH(q,n,ReLU()); break;
  case Cast::selu:         noarg(s,x,i); PUSH(q,n,SELU()); break;
  case Cast::relu6:        noarg(s,x,i); PUSH(q,n,ReLU6()); break;

  case Cast::softmax:      PUSH(q,n,(soft<Softmax>   (s,x,i))); break;
  case Cast::softmin:      PUSH(q,n,(soft<Softmin>   (s,x,i))); break;
  case Cast::logsoftmax:   PUSH(q,n,(soft<LogSoftmax>(s,x,i))); break;
  case Cast::flatten:      PUSH(q,n,Flatten(flatten(x,i))); break;
  case Cast::squeeze:      PUSH(q,n,Squeeze(squeeze(x,i,c))); break;
  case Cast::unsqueeze:    PUSH(q,n,Unsqueeze(squeeze(x,i,c))); break;
  case Cast::expand:       PUSH(q,n,Expand(getsize(x,i,c))); break;
  case Cast::reshape:      PUSH(q,n,Reshape(getsize(x,i,c))); break;

  case Cast::glu:          arg1(c,s,x,i,v); PUSH(q,n,GLU(v.toLong())); break;
  case Cast::elu:          arg1(c,s,x,i,v); PUSH(q,n,ELU(v)); break;
  case Cast::celu:         arg1(c,s,x,i,v); PUSH(q,n,CELU(v)); break;
  case Cast::leakyrelu:    arg1(c,s,x,i,v); PUSH(q,n,LeakyReLU(v)); break;
  case Cast::hardshrink:   arg1(c,s,x,i,v); PUSH(q,n,Hardshrink(v)); break;
  case Cast::softshrink:   arg1(c,s,x,i,v); PUSH(q,n,Softshrink(v)); break;
  
  case Cast::prelu:        arg2(c,s,x,i,v,w);  PUSH(q,n,PReLU(v.toLong(),w.toDouble())); break;
  case Cast::rrelu:        arg2(c,s,x,i,v,w);  PUSH(q,n,RReLU(v,w)); break;
  case Cast::hardtanh:     arg2(c,s,x,i,v,w);  PUSH(q,n,Hardtanh(v,w)); break;
  case Cast::softplus:     arg2(c,s,x,i,v,w);  PUSH(q,n,Softplus(v,w)); break;
  case Cast::threshold:    arg2(c,s,x,i,v,w);  PUSH(q,n,Threshold(v,w)); break;

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

 } else if(auto* m=g.as<torch::nn::Dropout>())        { c=Cast::dropout;   drop(a,x,m->options.p());
 } else if(auto* m=g.as<torch::nn::FeatureDropout>()) { c=Cast::fdropout;  drop(a,x,m->options.p());
 } else if(auto* m=g.as<AlphaDropout>())              { c=Cast::adropout;  drop(a,x,m->options.p());
 } else if(auto* m=g.as<FeatureAlphaDropout>())       { c=Cast::fadropout; drop(a,x,m->options.p());

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

 } else if(auto* m=g.as<LPPool1d>())         { c=Cast::lppool1d; lppool<1,LPPool1dImpl>(a,x,m);
 } else if(auto* m=g.as<LPPool2d>())         { c=Cast::lppool2d; lppool<2,LPPool2dImpl>(a,x,m);

 } else if(auto* m=g.as<Pad>())              { c=Cast::pad;         pad(a,x,m);
 } else if(auto* m=g.as<ReflectionPad1d>())  { c=Cast::reflect1d;   rpad(x,m);
 } else if(auto* m=g.as<ReflectionPad2d>())  { c=Cast::reflect2d;   rpad(x,m);
/*
 } else if(auto* m=g.as<torch::nn::ReflectionPad1d>())  { c=Cast::reflect1d;   rpad(x,m);
 } else if(auto* m=g.as<torch::nn::ReflectionPad2d>())  { c=Cast::reflect2d;   rpad(x,m);
*/
 } else if(auto* m=g.as<ReplicationPad1d>()) { c=Cast::replicate1d; rpad(x,m);
 } else if(auto* m=g.as<ReplicationPad2d>()) { c=Cast::replicate2d; rpad(x,m);
 } else if(auto* m=g.as<ReplicationPad3d>()) { c=Cast::replicate3d; rpad(x,m);

 } else if(auto* m=g.as<torch::nn::RNN>())   { c=Cast::rnn;  rnn<torch::nn::RNNImpl,  torch::nn::RNNOptions> (a,x,m);
 } else if(auto* m=g.as<torch::nn::GRU>())   { c=Cast::gru;  rnn<torch::nn::GRUImpl,  torch::nn::GRUOptions> (a,x,m);
 } else if(auto* m=g.as<torch::nn::LSTM>())  { c=Cast::lstm; rnn<torch::nn::LSTMImpl, torch::nn::LSTMOptions>(a,x,m);

 } else if(g.as<LogSigmoid>())         { c=Cast::logsigmoid;
 } else if(g.as<Sigmoid>())            { c=Cast::sigmoid;
 } else if(g.as<Softsign>())           { c=Cast::softsign;
 } else if(g.as<Tanh>())               { c=Cast::tanh;
 } else if(g.as<Tanhshrink>())         { c=Cast::tanhshrink;
 } else if(g.as<ReLU>())               { c=Cast::relu;
 } else if(g.as<SELU>())               { c=Cast::selu;
 } else if(g.as<ReLU6>())              { c=Cast::relu6;

 } else if(auto* m=g.as<Softmax>())    { c=Cast::softmax;    soft(a,x,m);
 } else if(auto* m=g.as<Softmin>())    { c=Cast::softmin;    soft(a,x,m);
 } else if(auto* m=g.as<LogSoftmax>()) { c=Cast::logsoftmax; soft(a,x,m);
 } else if(auto* m=g.as<Flatten>())    { c=Cast::flatten;    flatten(a,x,m);
 } else if(auto* m=g.as<Squeeze>())    { c=Cast::squeeze;    squeeze(a,x,m->options);
 } else if(auto* m=g.as<Unsqueeze>())  { c=Cast::unsqueeze;  squeeze(a,x,m->options);
 } else if(auto* m=g.as<Expand>())     { c=Cast::expand;     getsize(a,x,m->options);
 } else if(auto* m=g.as<Reshape>())    { c=Cast::reshape;    getsize(a,x,m->options);

 } else if(auto* m=g.as<GLU>())        { c=Cast::glu;        setting1(a,c,x,m->options.dim());
 } else if(auto* m=g.as<ELU>())        { c=Cast::elu;        setting1(a,c,x,m->options.alpha());
 } else if(auto* m=g.as<CELU>())       { c=Cast::celu;       setting1(a,c,x,m->options.alpha());
 } else if(auto* m=g.as<LeakyReLU>())  { c=Cast::leakyrelu;  setting1(a,c,x,m->options.slope());
 } else if(auto* m=g.as<Hardshrink>()) { c=Cast::hardshrink; setting1(a,c,x,m->options.lambda());
 } else if(auto* m=g.as<Softshrink>()) { c=Cast::softshrink; setting1(a,c,x,m->options.lambda());

 } else if(auto* m=g.as<PReLU>())      { c=Cast::prelu;      setting2(a,c,x,m->options.in(),        m->options.init());
 } else if(auto* m=g.as<RReLU>())      { c=Cast::rrelu;      setting2(a,c,x,m->options.lower(),     m->options.upper());
 } else if(auto* m=g.as<Hardtanh>())   { c=Cast::hardtanh;   setting2(a,c,x,m->options.min(),       m->options.max());
 } else if(auto* m=g.as<Softplus>())   { c=Cast::softplus;   setting2(a,c,x,m->options.beta(),      m->options.threshold());
 } else if(auto* m=g.as<Threshold>())  { c=Cast::threshold;  setting2(a,c,x,m->options.threshold(), m->options.value());

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
 fn(x, "seq",        KFN(seq), 1);           // api function for module create/query

 fn(x, "adaptavg1d", KFN(adaptavg1d), 1);    // functional form of modules/activations
 fn(x, "adaptavg2d", KFN(adaptavg2d), 1);
 fn(x, "adaptavg3d", KFN(adaptavg3d), 1);
 fn(x, "adaptmax1d", KFN(adaptmax1d), 1);
 fn(x, "adaptmax2d", KFN(adaptmax2d), 1);
 fn(x, "adaptmax3d", KFN(adaptmax3d), 1);
 fn(x, "avgpool1d",  KFN(avgpool1d), 1);
 fn(x, "avgpool2d",  KFN(avgpool2d), 1);
 fn(x, "avgpool3d",  KFN(avgpool3d), 1);
 fn(x, "celu",       KFN(kcelu), 1);
 fn(x, "elu",        KFN(kelu), 1);
 fn(x, "flatten",    KFN(kflatten), 1);
 fn(x, "glu",        KFN(kglu), 1);
 fn(x, "hardshrink", KFN(khardshrink), 1);
 fn(x, "hardtanh",   KFN(khardtanh), 1);
 fn(x, "leakyrelu",  KFN(kleakyrelu), 1);
 fn(x, "linear",     KFN(klinear), 1);
 fn(x, "logsigmoid", KFN(klogsigmoid), 1);
 fn(x, "logsoftmax", KFN(klogsoftmax), 1);
 fn(x, "maxpool1d",  KFN(maxpool1d), 1);
 fn(x, "maxpool2d",  KFN(maxpool2d), 1);
 fn(x, "maxpool3d",  KFN(maxpool3d), 1);
 fn(x, "prelu",      KFN(kprelu), 1);
 fn(x, "gelu",       KFN(kgelu), 1);
 fn(x, "relu",       KFN(krelu), 1);
 fn(x, "relu6",      KFN(krelu6), 1);
 fn(x, "rrelu",      KFN(krrelu), 1);
 fn(x, "selu",       KFN(kselu), 1);
 fn(x, "softmax",    KFN(ksoftmax), 1);
 fn(x, "softmin",    KFN(ksoftmin), 1);
 fn(x, "softplus",   KFN(ksoftplus), 1);
 fn(x, "softsign",   KFN(ksoftsign), 1);
 fn(x, "softshrink", KFN(ksoftshrink), 1);
 fn(x, "tanhshrink", KFN(ktanhshrink), 1);
 fn(x, "threshold",  KFN(kthreshold), 1);
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
  CELU(1.0),
  torch::nn::Conv1d(1,2,3),
  torch::nn::Conv2d(1,2,3),
  torch::nn::Conv3d(1,2,3),
  torch::nn::Dropout(.5),
  torch::nn::FeatureDropout(.5),
  torch::nn::FeatureDropout(),
  torch::nn::FeatureDropout(.2),
  AlphaDropout(.5),
  AlphaDropout(),
  AlphaDropout(.2),
  FeatureAlphaDropout(.5),
  FeatureAlphaDropout(),
  FeatureAlphaDropout(.25),
  ELU(),
  torch::nn::Embedding(4,10),
  torch::nn::FeatureDropout(.5),
  FractionalMaxPool2d(FractionalMaxPoolOptions<2>(3).ratio(.5)),
  FractionalMaxPool3d(FractionalMaxPoolOptions<3>(3).ratio(.5)),
  GLU(),
  torch::nn::GRU(4,5),
  Hardshrink(.5),
  Hardtanh(-1,1),
  torch::nn::LSTM(4,5),
  LeakyReLU(.1),
  torch::nn::Linear(3,4),
  LogSigmoid(),
  LogSoftmax(1,torch::kDouble),
  LPPool1d(2,3),
  LPPool2d(2,3),
  torch::nn::MaxPool1d(2),
  torch::nn::MaxPool2d(2),
  torch::nn::MaxPool3d(2),
  PReLU(1),
  Pad(LongVector{1,1}),
  torch::nn::RNN(4,5),
  RReLU(.125,.333),
  GELU(),
  ReLU(),
  ReLU6(),
  ReflectionPad1d(2),
  ReflectionPad2d(2),
  ReplicationPad1d(2),
  ReplicationPad2d(2),
  ReplicationPad3d(2),
  SELU(),
  Sigmoid(),
  Softsign(),
  Softmax(-1),
  Softmin(1),
  Softplus(1,20),
  Softshrink(.5),
  Tanh(),
  Tanhshrink(),
  Threshold(.1,20),
  Flatten(),
  Flatten(1),
  Flatten(1,-1),
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
