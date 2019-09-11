#pragma once

#ifdef __clang__
# pragma clang diagnostic push
# pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"                   // k.h warning
# pragma GCC diagnostic ignored "-Wnested-anon-types"                      // k.h warning
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // ATen.h VA_ARG warning
#elif defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wpedantic"
#endif

#define KXVER 3
#include "k.h"
#undef R
#undef xs

#include "torch/torch.h"

#ifdef __clang__
# pragma clang diagnostic pop
#elif defined __GNUC__
# pragma GCC diagnostic pop
#endif

#define KFN(f) reinterpret_cast<V *>(f)
#define KERR(e) krr((S)e)

#define KTRY \
 try {
#define KCATCH(x)                                                           \
 } catch (const c10::Error &e) {                                            \
  return KERR(krrbuf(env().frame ? e.what() : e.what_without_backtrace())); \
 } catch (const std::exception &e) {                                        \
  return KERR(krrbuf(e.what()));                                            \
 } catch (...) {                                                            \
  return KERR(x);                                                           \
 }

#ifdef __cplusplus
# define KEXT extern "C"
#else
# define KEXT
#endif

#ifdef _WIN32
# define KAPI KEXT __declspec(dllexport) K
#else
# define KAPI KEXT K
#endif

#define Ksize torch::SmallVector<int64_t,8>
#define cs(x) ss((S)x)

using A=signed char;
using B=bool;
using cS=const char*;

using Tensor=torch::Tensor;
using Scalar=torch::Scalar;
using JRef=torch::IntArrayRef;
template<size_t D,typename T=int64_t> using Expand=torch::ExpandingArray<D,T>;
using ScalarType=torch::ScalarType;
using TypeMeta=caffe2::TypeMeta;
using TensorOptions=torch::TensorOptions;
using Module=torch::nn::Module;
using Sequential=torch::nn::Sequential;
using OptimizerBase=torch::optim::detail::OptimizerBase;
using Optptr=std::shared_ptr<OptimizerBase>;
using Optimizer=torch::optim::Optimizer;
using LossClosureOptimizer=torch::optim::LossClosureOptimizer;
using TensorDict = torch::OrderedDict<std::string, torch::Tensor>;
class TORCH_API Loss;
using Lossptr=std::shared_ptr<Loss>;

typedef struct {
 A a = 0;  // type: 1-dict, 2-list of pairs, 3-general list, 4-sym list
 A t = 0;  // type of value in last pair processed
 H i = 0;  // next pair to process
 H n = 0;  // count of pairs
 S k = 0;  // name of an evaluated name,value pair
 K x = 0;  // k value with dict/pairs/list
 union {
  B b;  // boolean value from last evaluated pair
  J j;  // long value
  F f;  // double value
  S s;  // symbol value
  K v;  // value (isn't sym or numeric scalar)
 };
} Pairs;

enum class Class:char {
 undefined=0,
 tensor,
 vector,
 sequential,
 loss,
 optimizer,
 model
};

enum class Cast:char {
 undefined=0, 
 tensor,      sequential,  model,           //basic structures

 adaptavg1d,  adaptavg2d,  adaptavg3d,      //modules
 adaptmax1d,  adaptmax2d,  adaptmax3d,
 adropout,    avgpool1d,   avgpool2d,
 avgpool3d,   batchnorm,   celu,
 conv1d,      conv2d,      conv3d,
 dropout,     elu,         embed,
 fadropout,   fdropout,
 fmaxpool2d,  fmaxpool3d,

 gelu,        glu,         gru,
 hardshrink,  hardtanh,    leakyrelu,
 linear,      logsigmoid,  logsoftmax,
 lppool1d,    lppool2d,    lstm,
 maxpool1d,   maxpool2d,   maxpool3d,
 pad,         prelu,       reflect1d,
 reflect2d,   relu,        relu6,
 replicate1d, replicate2d, replicate3d,
 rnn,         rrelu,       selu,
 sigmoid,     softmax,     softmin,
 softplus,    softshrink,  softsign,
 tanh,        tanhshrink,  threshold,

 bce,         bcelogits,   bcelogitw,      //loss fns
 ce,          cosineloss,  ctc,
 hinge,       kl,          l1,
 margin,      mse,         multilabel,
 multimargin, multisoft,   nll,
 poissonloss, smoothl1,    softmargin,
 triplet,    

 adagrad,     adam,        lbfgs,          //optimizers
 rmsprop,     sgd
};

enum class Tensormode:char {   // tensor creation modes
 undefined,
 arange,   empty,    eye,      full,     linspace, logspace, ones,
 rand,     randint,  randn,    randperm, range,    zeros
};

enum class Setting:char {
 undefined,
 affine,     alpha,      amsgrad,    batchfirst, beta,       beta1,      beta2,
 bi,         bias,       blank,      ceiling,    centered,   changetol,  cols,
 countpad,   dampening,  decay,      dilate,     dim,        drop,       eps,
 eval,       fn,         full,       gradtol,    groups,     hidden,     history,
 ignore,     in,         indices,    init,       inplace,    iter,       lambda,
 layers,     log,        lower,      lr,         lrdecay,    margin,     max,
 min,        momentum,   nesterov,   out,        outpad,     outsize,    p,
 pad,        power,      ratio,      reduce,     rows,       size,       slope,
 stride,     swap,       threshold,  track,      train,      transpose,  type,
 upper,      value,      weight,     zeroinf
};

enum class State:char {
 Class,module,name,options,parms,buffers
};

struct TORCH_API Ktag {
 Class a = Class::undefined;
 Cast  c = Cast::undefined;
 virtual ~Ktag() = default;
};

struct TORCH_API Kten : public Ktag {
 Tensor t;
 Kten(const Tensor& x) : t(std::move(x)) {a=Class::tensor; c=Cast::tensor;}
};

struct TORCH_API Kvec : public Ktag {
 std::vector<Tensor> v;
 Kvec(const std::vector<Tensor>& x) : v(std::move(x)) {a=Class::vector; c=Cast::tensor;}
};

struct TORCH_API Kseq : public Ktag {
 Sequential q;
 Kseq(const Sequential& x) : q(std::move(x)) {a=Class::sequential; c=Cast::sequential;}
};

struct TORCH_API Kloss : public Ktag {
 Lossptr l;
 Kloss(Cast x,const Lossptr& y) : l(std::move(y)) {a=Class::loss; c=x;}
 bool is_empty() const noexcept {return l == nullptr;}
 Loss* get() {TORCH_CHECK(!is_empty(), "Undefined loss function"); return l.get();}
};

struct TORCH_API Kopt : public Ktag {
 Optptr o;
 Kopt(Cast x,const Optptr& y) : o(std::move(y)) {a=Class::optimizer; c=x;}
 bool is_empty() const noexcept {return o == nullptr;}
 OptimizerBase* get() {TORCH_CHECK(!is_empty(), "Undefined optimizer"); return o.get();}
};

struct TORCH_API Kmodel : public Ktag {
 Cast lc;          // loss fn
 Cast oc;          // optimizer
 Sequential q;     // sequential module
 Lossptr l;        // shared ptr to loss module
 Optptr o;         // shared ptr to optimizer
 Kmodel(Kseq x,Kloss y,Kopt z) : lc(y.c),oc(z.c),q(x.q),l(y.l),o(z.o) {a=Class::model; c=Cast::model;}
 Kmodel(Kseq *x,Kloss *y,Kopt *z) : lc(y->c),oc(z->c),q(x->q),l(y->l),o(z->o) {a=Class::model; c=Cast::model;}
};

S krrbuf(const char *);
V dictadd(K,S,K);
V dictadd(K,cS,K);
B xind(K,J);
K kptr(V*);
B xptr(K);
B xptr(K,J);
Ktag* xtag(K);
Ktag* xtag(K,J);

B match(const Scalar&,const Scalar&);
K kscalar(const Scalar&);
J xlen(K);
J xlen(K,J);
cS kname(A);
J ksizeof(A);
A maptype(TypeMeta);
TypeMeta maptype(A);
S mapclass(Class);

S statekey(State);
K statekeys();
J statefind(State,K);
S statesym(State e,K x,J j=-1);
K statedict(State e,K x,J j=-1);
V stateparms(S,Module&,K,B);

B xnull(K);
B xnull(K,J);
B xempty(K);
B xempty(K,J);
B xmixed(K,J);
B xsym(K,S&);
B xsym(K,J,S&);
B xsyms(K,S&);
B xdev(K,torch::Device&);
B xdev(K,J,torch::Device&);

B xint64(K,int64_t&);
B xint64(K,J,int64_t&);
B xlong(K,J&);
B xlong(K,J,J&);
B xlong(K,J&,J*&);
B xlong(K,J,J&,J*&);
B xdouble(K,F&);
B xdouble(K,J,F&);
B xdict(K);
B xdict(K,J);
B xstate(K);
B xstate(K,J);

B xsize(K,JRef&);
B xsize(K,J,JRef&);
B xsize(K,J,int64_t*);
B xsize(K,J,F*);
B xsize(K,J,J,int64_t*);
B xsize(K,J,J,F*);

B xten(K,Tensor&);
B xten(K,J,Tensor&);
Tensor* xten(K);
Tensor* xten(K,J);
B xtenpair(K,Tensor&,Tensor&);
B xtenpair(K,J,Tensor&,Tensor&);
B xten3(K,Tensor&,Tensor&,Tensor&);
B xten3(K,J,Tensor&,Tensor&,Tensor&);
B xtenarg(K,J,Tensor&,Tensor&);
B xtenarg(K,J,Tensor&,Tensor&,Tensor&);
B xtenarg(K,Tensor&,Tensor&);
B xtenarg(K,Tensor&,Tensor&,Tensor&);
B xseq(K,Sequential&);
B xseq(K,J,Sequential&);
Sequential* xseq(K);
Sequential* xseq(K,J);
Kloss* xloss(K);
Kloss* xloss(K,J);
Kopt* xoptim(K);
Kopt* xoptim(K,J);
Kmodel* xmodel(K);
Kmodel* xmodel(K,J);
std::vector<Tensor>* xvec(K);
std::vector<Tensor>* xvec(K,J);

B xnum(K,F&);
B xnum(K,J,F&);
B xnum(K,Scalar&);
B xnum(K,J,Scalar&);
B xnumn(K,c10::optional<Scalar>&);
B xnumn(K,J,c10::optional<Scalar>&);
B xnumt(K,Scalar&);
B xnumt(K,J,Scalar&);
B xnumlist(K,J,Scalar&);
B xbyte(K,Scalar&);
B xbyte(K,J,Scalar&);
B xscalar(K,Scalar&);
B xscalar(K,J,Scalar&);

B xbool(K,B&);
B xbool(K,J,B&);
B xlevel(K,I&);
TypeMeta mtype(S);
S mtype(TypeMeta);
ScalarType stype(S);
S stype(ScalarType);
S stype(c10::optional<ScalarType>);
B xtype(K,ScalarType&);
B xtype(K,J,ScalarType&);
B xtype(K,c10::optional<ScalarType>&);
B xtype(K,J,c10::optional<ScalarType>&);
B xtype(K,TypeMeta&);
B xtype(K,J,TypeMeta&);
B xopt(S,TensorOptions&);
B xopt(K,TensorOptions&);
B xopt(K,J,TensorOptions&);
B xto(S,TensorOptions&);
B xto(K,TensorOptions&);
B xto(K,J,TensorOptions&);
B xmode(K,S&,Tensormode&);
B xmode(K,J,S&,Tensormode&);
B xbacksym(K,B&,B&);
B xbacksym(K,J,B&,B&);

B xpairs(K,Pairs&);
B xpairs(K,J,Pairs&);
B xpair(Pairs&);
J xargc(K,J,Pairs&);
B xnone(K,J);

S psym(const Pairs&);
ScalarType ptype(const Pairs&);
V perr(const Pairs&,cS);
B pbool(const Pairs&);
J plong(const Pairs&);
F pdouble(const Pairs&);
V pnum(const Pairs&,Scalar&);
V psize(const Pairs&,JRef&,J n=-1);
V psize(const Pairs&,J,int64_t*);
V psize(const Pairs&,J,F*);
V pten(const Pairs&,Tensor&);

S& optsym(const torch::Device&);
S& optsym(const TypeMeta&);
S& optsym(const torch::Layout&);
S& optsym(const bool&);
K optkey();
K optval(const TensorOptions &o,K x,J i=-1);
K optmap(const TensorOptions&);
K kcast(A,K);
K kbool(K);
K kdict(const TensorDict&);
J kfind(K,const std::string&);
K klist(J,const int64_t*);
K klist(J,const F*);
K kexpand(J,const int64_t*);
K kexpand(J,const F*);
#define KEX(x) kexpand(x.size(),(*x).data())  // k list from ExpandingArray
V fn(K,cS,V*,I);

V randomfn(K);
V mathfn(K);

// tensor routines:
K kget(const Tensor&);
K kget(const std::vector<int64_t>&);
K kget(const std::vector<Tensor>&);
K kget(const std::deque<Tensor>&);
Tensor kput(K);
Tensor kput(K,J);
K kten(const Tensor&);
K kvec(const std::vector<Tensor>&);
K tento(Kten*,const TensorOptions&,B,B);
K vecto(Kvec*,const TensorOptions&,B);

K ktenpair(B,Tensor&,Tensor&);
K kten3(B,Tensor&,Tensor&,Tensor&);
K tensordetail(const Tensor&,I);
V tensorcopy(Tensor&,const Tensor&,B async=false);
V tensorfn(K);

// module routines:
//K kseq(const Sequential&,Cast c=Cast::sequential);
K kseq(const Sequential&);
K seqto(Kseq*,const TensorOptions&,B);
V modfn(K);
K mstate(K);

// loss functions:
K lossdict(Ktag*,K);
K lossto(Kloss*,const TensorOptions&,B);
V lossfn(K);

// optimization functions:
K optstate(Ktag*,K);
V optfn(K);

// global environment
typedef struct {
 I cuda;             // number of CUDA devices
 B frame=false;      // if true, error message returns stack frame
 B alloptions=true;  // if true, queries return all settings, else only those not matching defaults
 S help=cs("help");

 std::vector<std::tuple<S,torch::Device>> device;

 std::array<std::tuple<A,TypeMeta>,8> ktype = {{               //k type -> torch type
  std::make_tuple(KE, at::scalarTypeToTypeMeta(at::kFloat)),
  std::make_tuple(KF, at::scalarTypeToTypeMeta(at::kDouble)),
  std::make_tuple(KJ, at::scalarTypeToTypeMeta(at::kLong)),
  std::make_tuple(KI, at::scalarTypeToTypeMeta(at::kInt)),
  std::make_tuple(KH, at::scalarTypeToTypeMeta(at::kShort)),
  std::make_tuple(KB, at::scalarTypeToTypeMeta(at::kBool)),
  std::make_tuple(KG, at::scalarTypeToTypeMeta(at::kByte)),
  std::make_tuple(KC, at::scalarTypeToTypeMeta(at::kChar))
 }};

 std::array<std::tuple<S,TypeMeta,A>,9> dtype = {{                            //sym -> torch type -> k type
  std::make_tuple(cs("float"),  at::scalarTypeToTypeMeta(at::kFloat),  KE),
  std::make_tuple(cs("double"), at::scalarTypeToTypeMeta(at::kDouble), KF),
  std::make_tuple(cs("half"),   at::scalarTypeToTypeMeta(at::kHalf),   KE),
  std::make_tuple(cs("bool"),   at::scalarTypeToTypeMeta(at::kBool),   KB),
  std::make_tuple(cs("byte"),   at::scalarTypeToTypeMeta(at::kByte),   KG),
  std::make_tuple(cs("char"),   at::scalarTypeToTypeMeta(at::kChar),   KC),
  std::make_tuple(cs("long"),   at::scalarTypeToTypeMeta(at::kLong),   KJ),
  std::make_tuple(cs("int"),    at::scalarTypeToTypeMeta(at::kInt),    KI),
  std::make_tuple(cs("short"),  at::scalarTypeToTypeMeta(at::kShort),  KH)
 }};

 std::array<std::tuple<S,torch::Layout>,2> layout = {{
  std::make_tuple(cs("strided"),torch::kStrided),          
  std::make_tuple(cs("sparse"), torch::kSparse)
 }};

 std::array<std::tuple<S,B>,2> gradient = {{
  std::make_tuple(cs("grad"),   true),          
  std::make_tuple(cs("nograd"), false)
 }};

/*
 std::array<std::tuple<S,B>,2> async = {{
  std::make_tuple(cs("async"),   true),          
  std::make_tuple(cs("sync"),   false)
 }};
*/

 std::array<std::tuple<S,Class>,6> kclass = {{
  std::make_tuple(cs("tensor"),     Class::tensor),          
  std::make_tuple(cs("vector"),     Class::vector),
  std::make_tuple(cs("sequential"), Class::sequential),
  std::make_tuple(cs("loss"),       Class::loss),
  std::make_tuple(cs("optimizer"),  Class::optimizer),
  std::make_tuple(cs("model"),      Class::model)
 }};

 std::array<std::tuple<S,Tensormode>,13> tensormode = {{    //tensor creation mode: map symbol -> enum
  std::make_tuple(cs("empty"),    Tensormode::empty),
  std::make_tuple(cs("full"),     Tensormode::full),
  std::make_tuple(cs("eye"),      Tensormode::eye),
  std::make_tuple(cs("ones"),     Tensormode::ones),
  std::make_tuple(cs("zeros"),    Tensormode::zeros),
  std::make_tuple(cs("rand"),     Tensormode::rand),
  std::make_tuple(cs("randn"),    Tensormode::randn),
  std::make_tuple(cs("randint"),  Tensormode::randint),
  std::make_tuple(cs("randperm"), Tensormode::randperm),
  std::make_tuple(cs("range"),    Tensormode::range),
  std::make_tuple(cs("arange"),   Tensormode::arange),
  std::make_tuple(cs("linspace"), Tensormode::linspace),
  std::make_tuple(cs("logspace"), Tensormode::logspace)
 }};

 std::array<std::tuple<S,torch::nn::RNNActivation>,2> rnnfn = {{
  std::make_tuple(cs("relu"),torch::nn::RNNActivation::ReLU),
  std::make_tuple(cs("tanh"),torch::nn::RNNActivation::Tanh)
 }};

 std::array<std::tuple<S,Cast>,59> module = {{  // module sym -> enum
  std::make_tuple(cs("adaptavg1d"),      Cast::adaptavg1d),
  std::make_tuple(cs("adaptavg2d"),      Cast::adaptavg2d),
  std::make_tuple(cs("adaptavg3d"),      Cast::adaptavg3d),
  std::make_tuple(cs("adaptmax1d"),      Cast::adaptmax1d),
  std::make_tuple(cs("adaptmax2d"),      Cast::adaptmax2d),
  std::make_tuple(cs("adaptmax3d"),      Cast::adaptmax3d),
  std::make_tuple(cs("adropout"),        Cast::adropout),
  std::make_tuple(cs("avgpool1d"),       Cast::avgpool1d),
  std::make_tuple(cs("avgpool2d"),       Cast::avgpool2d),
  std::make_tuple(cs("avgpool3d"),       Cast::avgpool3d),
  std::make_tuple(cs("batchnorm"),       Cast::batchnorm),
  std::make_tuple(cs("celu"),            Cast::celu),
  std::make_tuple(cs("conv1d"),          Cast::conv1d),
  std::make_tuple(cs("conv2d"),          Cast::conv2d),
  std::make_tuple(cs("conv3d"),          Cast::conv3d),
  std::make_tuple(cs("dropout"),         Cast::dropout),
  std::make_tuple(cs("elu"),             Cast::elu),
  std::make_tuple(cs("embed"),           Cast::embed),
  std::make_tuple(cs("fdropout"),        Cast::fdropout),
  std::make_tuple(cs("fadrop"),          Cast::fadropout),
  std::make_tuple(cs("fmaxpool2d"),      Cast::fmaxpool2d),
  std::make_tuple(cs("fmaxpool3d"),      Cast::fmaxpool3d),
  std::make_tuple(cs("gelu"),            Cast::gelu),
  std::make_tuple(cs("glu"),             Cast::glu),
  std::make_tuple(cs("gru"),             Cast::gru),
  std::make_tuple(cs("hardshrink"),      Cast::hardshrink),
  std::make_tuple(cs("hardtanh"),        Cast::hardtanh),
  std::make_tuple(cs("leakyrelu"),       Cast::leakyrelu),
  std::make_tuple(cs("linear"),          Cast::linear),
  std::make_tuple(cs("logsigmoid"),      Cast::logsigmoid),
  std::make_tuple(cs("logsoftmax"),      Cast::logsoftmax),
  std::make_tuple(cs("lppool1d"),        Cast::lppool1d),
  std::make_tuple(cs("lppool2d"),        Cast::lppool2d),
  std::make_tuple(cs("lstm"),            Cast::lstm),
  std::make_tuple(cs("maxpool1d"),       Cast::maxpool1d),
  std::make_tuple(cs("maxpool2d"),       Cast::maxpool2d),
  std::make_tuple(cs("maxpool3d"),       Cast::maxpool3d),
  std::make_tuple(cs("pad"),             Cast::pad),
  std::make_tuple(cs("prelu"),           Cast::prelu),
  std::make_tuple(cs("reflect1d"),       Cast::reflect1d),
  std::make_tuple(cs("reflect2d"),       Cast::reflect2d),
  std::make_tuple(cs("relu"),            Cast::relu),
  std::make_tuple(cs("relu6"),           Cast::relu6),
  std::make_tuple(cs("replicate1d"),     Cast::replicate1d),
  std::make_tuple(cs("replicate2d"),     Cast::replicate2d),
  std::make_tuple(cs("replicate3d"),     Cast::replicate3d),
  std::make_tuple(cs("rnn"),             Cast::rnn),
  std::make_tuple(cs("rrelu"),           Cast::rrelu),
  std::make_tuple(cs("selu"),            Cast::selu),
  std::make_tuple(cs("sequential"),      Cast::sequential),
  std::make_tuple(cs("sigmoid"),         Cast::sigmoid),
  std::make_tuple(cs("softmax"),         Cast::softmax),
  std::make_tuple(cs("softmin"),         Cast::softmin),
  std::make_tuple(cs("softplus"),        Cast::softplus),
  std::make_tuple(cs("softshrink"),      Cast::softshrink),
  std::make_tuple(cs("softsign"),        Cast::softsign),
  std::make_tuple(cs("tanh"),            Cast::tanh),
  std::make_tuple(cs("tanhshrink"),      Cast::tanhshrink),
  std::make_tuple(cs("threshold"),       Cast::threshold)
 }};

 std::array<std::tuple<S,Setting>,43> mset = {{      // module option sym -> enum
  std::make_tuple(cs("affine"),     Setting::affine),
  std::make_tuple(cs("alpha"),      Setting::alpha),
  std::make_tuple(cs("batchfirst"), Setting::batchfirst),
  std::make_tuple(cs("beta"),       Setting::beta),
  std::make_tuple(cs("bi"),         Setting::bi),
  std::make_tuple(cs("bias"),       Setting::bias),
  std::make_tuple(cs("ceiling"),    Setting::ceiling),
  std::make_tuple(cs("cols"),       Setting::cols),
  std::make_tuple(cs("countpad"),   Setting::countpad),
  std::make_tuple(cs("dilate"),     Setting::dilate),
  std::make_tuple(cs("dim"),        Setting::dim),
  std::make_tuple(cs("drop"),       Setting::drop),
  std::make_tuple(cs("eps"),        Setting::eps),
  std::make_tuple(cs("fn"),         Setting::fn),
  std::make_tuple(cs("groups"),     Setting::groups),
  std::make_tuple(cs("hidden"),     Setting::hidden),
  std::make_tuple(cs("in"),         Setting::in),
  std::make_tuple(cs("indices"),    Setting::indices),
  std::make_tuple(cs("init"),       Setting::init),
  std::make_tuple(cs("inplace"),    Setting::inplace),
  std::make_tuple(cs("lambda"),     Setting::lambda),
  std::make_tuple(cs("layers"),     Setting::layers),
  std::make_tuple(cs("lower"),      Setting::lower),
  std::make_tuple(cs("max"),        Setting::max),
  std::make_tuple(cs("min"),        Setting::min),
  std::make_tuple(cs("momentum"),   Setting::momentum),
  std::make_tuple(cs("out"),        Setting::out),
  std::make_tuple(cs("outpad"),     Setting::outpad),
  std::make_tuple(cs("outsize"),    Setting::outsize),
  std::make_tuple(cs("pad"),        Setting::pad),
  std::make_tuple(cs("power"),      Setting::power),
  std::make_tuple(cs("ratio"),      Setting::ratio),
  std::make_tuple(cs("rows"),       Setting::rows),
  std::make_tuple(cs("size"),       Setting::size),
  std::make_tuple(cs("slope"),      Setting::slope),
  std::make_tuple(cs("stride"),     Setting::stride),
  std::make_tuple(cs("threshold"),  Setting::threshold),
  std::make_tuple(cs("track"),      Setting::track),
  std::make_tuple(cs("train"),      Setting::train),
  std::make_tuple(cs("transpose"),  Setting::transpose),
  std::make_tuple(cs("type"),       Setting::type),
  std::make_tuple(cs("upper"),      Setting::upper),
  std::make_tuple(cs("value"),      Setting::value)
 }};

 std::array<std::tuple<S,State>,6> state = {{         //state dictionary keys: map symbol -> enum
  std::make_tuple(cs("class"),   State::Class),
  std::make_tuple(cs("module"),  State::module),
  std::make_tuple(cs("name"),    State::name),
  std::make_tuple(cs("options"), State::options),
  std::make_tuple(cs("parms"),   State::parms),
  std::make_tuple(cs("buffers"), State::buffers)
 }};

 std::array<std::tuple<S,Cast>,19> loss = {{             // loss: map symbol -> enum
  std::make_tuple(cs("bce"),          Cast::bce),
  std::make_tuple(cs("bcelogits"),    Cast::bcelogits),
  std::make_tuple(cs("bcelogitw"),    Cast::bcelogitw),
  std::make_tuple(cs("ce"),           Cast::ce),
  std::make_tuple(cs("cosineloss"),   Cast::cosineloss),
  std::make_tuple(cs("ctc"),          Cast::ctc),
  std::make_tuple(cs("hinge"),        Cast::hinge),
  std::make_tuple(cs("kl"),           Cast::kl),
  std::make_tuple(cs("l1"),           Cast::l1),
  std::make_tuple(cs("margin"),       Cast::margin),
  std::make_tuple(cs("mse"),          Cast::mse),
  std::make_tuple(cs("multilabel"),   Cast::multilabel),
  std::make_tuple(cs("multimargin"),  Cast::multimargin),
  std::make_tuple(cs("multisoft"),    Cast::multisoft),
  std::make_tuple(cs("nll"),          Cast::nll),
  std::make_tuple(cs("poissonloss"),  Cast::poissonloss),
  std::make_tuple(cs("smoothl1"),     Cast::smoothl1),
  std::make_tuple(cs("softmargin"),   Cast::softmargin),
  std::make_tuple(cs("triplet"),      Cast::triplet)
 }};

 std::array<std::tuple<S,Setting>,11> lset = {{          // loss option sym -> enum
  std::make_tuple(cs("blank"),     Setting::blank),
  std::make_tuple(cs("eps"),       Setting::eps),
  std::make_tuple(cs("full"),      Setting::full),
  std::make_tuple(cs("ignore"),    Setting::ignore),
  std::make_tuple(cs("log"),       Setting::log),
  std::make_tuple(cs("margin"),    Setting::margin),
  std::make_tuple(cs("p"),         Setting::p),
  std::make_tuple(cs("reduce"),    Setting::reduce),
  std::make_tuple(cs("swap"),      Setting::swap),
  std::make_tuple(cs("weight"),    Setting::weight),
  std::make_tuple(cs("zeroinf"),   Setting::zeroinf)
 }};

 std::array<std::tuple<S,Cast,double>,5> opt = {{        //optimizer: map symbol -> enum, default learning rate
  std::make_tuple(cs("adagrad"), Cast::adagrad, 0.010),
  std::make_tuple(cs("adam"),    Cast::adam,    0.001),
  std::make_tuple(cs("lbfgs"),   Cast::lbfgs,   1.000),
  std::make_tuple(cs("rmsprop"), Cast::rmsprop, 0.010),
  std::make_tuple(cs("sgd"),     Cast::sgd,     0.010)
 }};

 std::array<std::tuple<S,Setting>,17> oset = {{         //optimizer setting: map symbol -> enum
  std::make_tuple(cs("lr"),         Setting::lr),
  std::make_tuple(cs("lrdecay"),    Setting::lrdecay),
  std::make_tuple(cs("decay"),      Setting::decay),
  std::make_tuple(cs("beta1"),      Setting::beta1),
  std::make_tuple(cs("beta2"),      Setting::beta2),
  std::make_tuple(cs("eps"),        Setting::eps),
  std::make_tuple(cs("amsgrad"),    Setting::amsgrad),
  std::make_tuple(cs("iter"),       Setting::iter),
  std::make_tuple(cs("eval"),       Setting::eval),
  std::make_tuple(cs("gradtol"),    Setting::gradtol),
  std::make_tuple(cs("changetol"),  Setting::changetol),
  std::make_tuple(cs("history"),    Setting::history),
  std::make_tuple(cs("alpha"),      Setting::alpha),
  std::make_tuple(cs("momentum"),   Setting::momentum),
  std::make_tuple(cs("centered"),   Setting::centered),
  std::make_tuple(cs("dampening"),  Setting::dampening),
  std::make_tuple(cs("nesterov"),   Setting::nesterov)
 }};

 std::array<std::tuple<S,bool,bool>,4> backsym = {{     //map sym to booleans for retain_graph & create_graph
  std::make_tuple(cs("free"),       false, false),
  std::make_tuple(cs("retain"),     true,  false),
  std::make_tuple(cs("create"),     true,  true),
  std::make_tuple(cs("createfree"), false, true)
 }};

 std::array<std::tuple<S,int64_t>,3> reduce = {{
  std::make_tuple(cs("none"), Reduction::None),
  std::make_tuple(cs("mean"), Reduction::Mean),
  std::make_tuple(cs("sum"),  Reduction::Sum)
 }};
} Env;

Env& env();
