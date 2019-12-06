#pragma once
// ---------------------------------------------------------------------
// Adaptive Avg Pool pending, should be released in 1.4
// Options for a `D`-dimensional adaptive avgpool functional and module
// ---------------------------------------------------------------------
/*
namespace torch { namespace nn {
template <size_t D> struct AdaptiveAvgPoolOptions {
  AdaptiveAvgPoolOptions(ExpandingArray<D> output_size) : output_size_(output_size) {}
  TORCH_ARG(ExpandingArray<D>, output_size);
};

using AdaptiveAvgPool1dOptions = AdaptiveAvgPoolOptions<1>;
using AdaptiveAvgPool2dOptions = AdaptiveAvgPoolOptions<2>;
using AdaptiveAvgPool3dOptions = AdaptiveAvgPoolOptions<3>;

namespace functional {
inline Tensor adaptive_avg_pool1d(const Tensor& input, const AdaptiveAvgPool1dOptions& options) {
   return torch::adaptive_avg_pool1d(input, options.output_size());
}

inline Tensor adaptive_avg_pool2d(const Tensor& input, const AdaptiveAvgPool2dOptions& options) {
   return torch::adaptive_avg_pool2d(input, options.output_size());
}

inline Tensor adaptive_avg_pool3d(const Tensor& input, const AdaptiveAvgPool3dOptions& options) {
   return torch::adaptive_avg_pool3d(input, options.output_size());
}
}

template <size_t D, typename Derived>
class TORCH_API AdaptiveAvgPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  AdaptiveAvgPoolImpl(ExpandingArray<D> output_size)
      : AdaptiveAvgPoolImpl(AdaptiveAvgPoolOptions<D>(output_size)) {}
  explicit AdaptiveAvgPoolImpl(const AdaptiveAvgPoolOptions<D>& options_) : options(options_) {}
  void reset() override {}
  AdaptiveAvgPoolOptions<D> options;
};

class TORCH_API AdaptiveAvgPool1dImpl :
  public AdaptiveAvgPoolImpl<1, AdaptiveAvgPool1dImpl> {
 public:
  using AdaptiveAvgPoolImpl<1, AdaptiveAvgPool1dImpl>::AdaptiveAvgPoolImpl;
  Tensor forward(const Tensor& input) {return torch::nn::functional::adaptive_avg_pool1d(input, options);}
};

class TORCH_API AdaptiveAvgPool2dImpl :
  public AdaptiveAvgPoolImpl<2, AdaptiveAvgPool2dImpl> {
 public:
  using AdaptiveAvgPoolImpl<2, AdaptiveAvgPool2dImpl>::AdaptiveAvgPoolImpl;
  Tensor forward(const Tensor& input) {return torch::nn::functional::adaptive_avg_pool2d(input, options);}
};

class TORCH_API AdaptiveAvgPool3dImpl :
  public AdaptiveAvgPoolImpl<3, AdaptiveAvgPool3dImpl> {
 public:
  using AdaptiveAvgPoolImpl<3, AdaptiveAvgPool3dImpl>::AdaptiveAvgPoolImpl;
  Tensor forward(const Tensor& input) {return torch::nn::functional::adaptive_avg_pool3d(input, options);}
};

TORCH_MODULE(AdaptiveAvgPool1d);
TORCH_MODULE(AdaptiveAvgPool2d);
TORCH_MODULE(AdaptiveAvgPool3d);
}}
*/

// ------------------------------------------
// fractional max pool 2,3d
// ------------------------------------------
template <size_t D>
struct TORCH_API FractionalMaxPoolOptions {
 using Ex=torch::ExpandingArray<D>;
 using Ef=torch::ExpandingArray<D,double>;
 FractionalMaxPoolOptions(Ex s) : size_(s) {}
 TORCH_ARG(Ex,                size);
 TORCH_ARG(c10::optional<Ex>, outsize);
 TORCH_ARG(c10::optional<Ef>, ratio);
 TORCH_ARG(bool,              indices)=false;
};

template <size_t D, typename Derived>
class FractionalMaxPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  FractionalMaxPoolImpl(torch::ExpandingArray<D> size) : FractionalMaxPoolImpl(FractionalMaxPoolOptions<D>(size)) {}
  explicit FractionalMaxPoolImpl(const FractionalMaxPoolOptions<D>& o) : options(o) {reset();}

  void reset() override {
   TORCH_CHECK(  options.outsize() || options.ratio(),  "no output size or ratio");
   TORCH_CHECK(!(options.outsize() && options.ratio()), "both output size and output ratio defined");
   if(options.ratio().has_value()) {
    for(auto i:*(options.ratio().value()))
     TORCH_CHECK(0<i && i<1, "output ratios must be positive and less than 1");
   }
   if(options.indices()) indices=torch::nn::Module::register_buffer("indices",indices);
  }

  void setup(const torch::Tensor& t,torch::Tensor &s,std::array<int64_t,D>& sz) {
   TORCH_CHECK(t.dim()==D+2, D+2,"-dimensional input expected, ",t.dim()," dimension(s) supplied");
   s=torch::rand({t.size(0),t.size(1),D}, torch::dtype(t.dtype()).device(t.device()));
   if(!options.outsize())
    for(size_t i=0;i<D;++i) sz[i]=t.size(i+2)*(*options.ratio().value())[i];
  }

  FractionalMaxPoolOptions<D> options;
  torch::Tensor indices;
};

class TORCH_API FractionalMaxPool2dImpl : public FractionalMaxPoolImpl<2, FractionalMaxPool2dImpl> {
 public:
  using FractionalMaxPoolImpl<2, FractionalMaxPool2dImpl>::FractionalMaxPoolImpl;
  torch::Tensor forward(const torch::Tensor& t) {
   torch::Tensor i,r,s; std::array<int64_t,2> sz; setup(t,s,sz);
   std::tie(r,i)=torch::fractional_max_pool2d(t, options.size(), options.outsize() ? options.outsize().value() : sz, s);
   if(options.indices()) indices=i;
   return r;
 }
};

class TORCH_API FractionalMaxPool3dImpl : public FractionalMaxPoolImpl<3, FractionalMaxPool3dImpl> {
 public:
  using FractionalMaxPoolImpl<3, FractionalMaxPool3dImpl>::FractionalMaxPoolImpl;
  torch::Tensor forward(const torch::Tensor& t) {
   torch::Tensor i,r,s; std::array<int64_t,3> sz; setup(t,s,sz);
   std::tie(r,i)=torch::fractional_max_pool3d(t, options.size(), options.outsize() ? options.outsize().value() : sz, s);
   if(options.indices()) indices=i;
   return r;
 }
};

TORCH_MODULE(FractionalMaxPool2d);
TORCH_MODULE(FractionalMaxPool3d);

// ------------------------------------------
// lp pool 1d & 2d 
// ------------------------------------------
template <size_t D>
struct TORCH_API LPPoolOptions {
 using Ex=torch::ExpandingArray<D>;
 LPPoolOptions(double p,Ex s) : power_(p),kernel_size_(s),stride_(s) {}
 TORCH_ARG(double, power);
 TORCH_ARG(Ex,     kernel_size);
 TORCH_ARG(Ex,     stride);
 TORCH_ARG(bool,   ceil_mode)=false;
};

template <size_t D, typename Derived>
class LPPoolImpl : public torch::nn::Cloneable<Derived> {
 public:
  LPPoolImpl(double p,torch::ExpandingArray<D> s) : LPPoolImpl(LPPoolOptions<D>(p,s)) {}
  //explicit LPPoolImpl(LPPoolOptions<D> o) : options(std::move(o)) {reset();}
  explicit LPPoolImpl(const LPPoolOptions<D>& o) : options(o) {};
  void reset() override {}
  LPPoolOptions<D> options;
};

class TORCH_API LPPool1dImpl : public LPPoolImpl<1, LPPool1dImpl> {
 public:
  using LPPoolImpl<1, LPPool1dImpl>::LPPoolImpl;
  torch::Tensor forward(const torch::Tensor& t) {
   auto r=torch::avg_pool1d(t.pow(options.power()),options.kernel_size(),options.stride(),0,options.ceil_mode());
   return r.mul((*options.kernel_size())[0]).pow(1.0/options.power());
  }
};

class TORCH_API LPPool2dImpl : public LPPoolImpl<2, LPPool2dImpl> {
 public:
  using LPPoolImpl<2, LPPool2dImpl>::LPPoolImpl;
  torch::Tensor forward(const torch::Tensor& t) {
   auto r=torch::avg_pool2d(t.pow(options.power()),options.kernel_size(),options.stride(),0,options.ceil_mode());
   return (torch::sign(r) * torch::relu(torch::abs(r))).mul((*options.kernel_size())[0]*(*options.kernel_size())[1]).pow(1.0/options.power());
  }
};

TORCH_MODULE(LPPool1d);
TORCH_MODULE(LPPool2d);

// -------------------------------------------------------------------------------
// flexible/fixed-dim padding options for constant/reflect/replicate padding
// -------------------------------------------------------------------------------
struct TORCH_API PadOptions {
 PadOptions(std::vector<int64_t> p) : pad_(std::move(p)) {}
 PadOptions() {}
 TORCH_ARG(std::vector<int64_t>, pad);
 TORCH_ARG(torch::Scalar, value)=0;
};

/*
namespace torch { namespace nn {
template <size_t D>
struct TORCH_API ReflectionPadOptions {
  ReflectionPadOptions(ExpandingArray<D*2> padding) : padding_(padding) {}
  TORCH_ARG(ExpandingArray<D*2>, padding);
};
using ReflectionPad1dOptions = ReflectionPadOptions<1>;
using ReflectionPad2dOptions = ReflectionPadOptions<2>;

template <size_t D>
struct TORCH_API ReplicationPadOptions {
  ReplicationPadOptions(ExpandingArray<D*2> padding) : padding_(padding) {}
  TORCH_ARG(ExpandingArray<D*2>, padding);
};

using ReplicationPad1dOptions = ReplicationPadOptions<1>;
using ReplicationPad2dOptions = ReplicationPadOptions<2>;
using ReplicationPad3dOptions = ReplicationPadOptions<3>;
}}
*/

// ------------------------------------------
// constant pad n-dim
// ------------------------------------------
class TORCH_API PadImpl : public torch::nn::Cloneable<PadImpl> {
 public:
  PadImpl(std::vector<int64_t> p) : PadImpl(PadOptions(p)) {}
  explicit PadImpl(PadOptions o) : options(std::move(o)) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& input) {
   return torch::constant_pad_nd(input,options.pad(),options.value());
  }
  PadOptions options;
};

TORCH_MODULE(Pad);

// ------------------------------------------
// reflection pad 1,2d
// ------------------------------------------
template <size_t D, typename Derived>
class TORCH_API ReflectionPadImpl : public torch::nn::Cloneable<Derived> {
 public:
  ReflectionPadImpl(torch::ExpandingArray<D*2> padding)
      : ReflectionPadImpl(torch::nn::ReflectionPadOptions<D>(padding)) {}
  explicit ReflectionPadImpl(const torch::nn::ReflectionPadOptions<D>& options_) : options(options_) {}
  void reset() override {}
  torch::nn::ReflectionPadOptions<D> options;
};

class TORCH_API ReflectionPad1dImpl : public ReflectionPadImpl<1, ReflectionPad1dImpl> {
 public:
  using ReflectionPadImpl<1, ReflectionPad1dImpl>::ReflectionPadImpl;
  torch::Tensor forward(const torch::Tensor& input) {
   return torch::reflection_pad1d(input,options.padding());
  }
};

class TORCH_API ReflectionPad2dImpl : public ReflectionPadImpl<2, ReflectionPad2dImpl> {
 public:
  using ReflectionPadImpl<2, ReflectionPad2dImpl>::ReflectionPadImpl;
  torch::Tensor forward(const torch::Tensor& input) {
   return torch::reflection_pad2d(input,options.padding());
  }
};

TORCH_MODULE(ReflectionPad1d);
TORCH_MODULE(ReflectionPad2d);

// ------------------------------------------
// replication pad 1,2,3d
// ------------------------------------------
template <size_t D, typename Derived>
class TORCH_API ReplicationPadImpl : public torch::nn::Cloneable<Derived> {
 public:
  ReplicationPadImpl(torch::ExpandingArray<D*2> p) : ReplicationPadImpl(torch::nn::ReplicationPadOptions<D>(p)) {}
  explicit ReplicationPadImpl(const torch::nn::ReplicationPadOptions<D> o) : options(o) {}
  void reset() override {}
  torch::nn::ReplicationPadOptions<D> options;
};

class TORCH_API ReplicationPad1dImpl : public ReplicationPadImpl<1, ReplicationPad1dImpl> {
 public:
  using ReplicationPadImpl<1, ReplicationPad1dImpl>::ReplicationPadImpl;
  torch::Tensor forward(const torch::Tensor& input) {
   return torch::replication_pad1d(input,options.padding());
  }
};

class TORCH_API ReplicationPad2dImpl : public ReplicationPadImpl<2, ReplicationPad2dImpl> {
 public:
  using ReplicationPadImpl<2, ReplicationPad2dImpl>::ReplicationPadImpl;
  torch::Tensor forward(const torch::Tensor& input) {
   return torch::replication_pad2d(input,options.padding());
  }
};

class TORCH_API ReplicationPad3dImpl : public ReplicationPadImpl<3, ReplicationPad3dImpl> {
 public:
  using ReplicationPadImpl<3, ReplicationPad3dImpl>::ReplicationPadImpl;
  torch::Tensor forward(const torch::Tensor& input) {
   return torch::replication_pad3d(input,options.padding());
  }
};

TORCH_MODULE(ReplicationPad1d);
TORCH_MODULE(ReplicationPad2d);
TORCH_MODULE(ReplicationPad3d);

// ------------------------------------------------------------------------------
// fns without args: logsigmoid,tanhshrink,softsign,tanh,sigmoid,gelu
// also fns w'inplace as only arg: relu,relu6,selu
// (inplace=true doesn't seem to work with Sequential->forward() )
// ------------------------------------------------------------------------------
class TORCH_API LogSigmoidImpl : public torch::nn::Cloneable<LogSigmoidImpl> {
 public:
  LogSigmoidImpl() = default;
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& input) {return torch::log_sigmoid(input);}
};
TORCH_MODULE(LogSigmoid);

class TORCH_API TanhshrinkImpl : public torch::nn::Cloneable<TanhshrinkImpl> {
 public:
  TanhshrinkImpl() = default;
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& input) {return input - input.tanh();}
};
TORCH_MODULE(Tanhshrink);

class TORCH_API SoftsignImpl : public torch::nn::Cloneable<SoftsignImpl> {
 public:
  SoftsignImpl() = default;
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& input) {return input / (input.abs() + 1);}
};
TORCH_MODULE(Softsign);

class TORCH_API TanhImpl : public torch::nn::Cloneable<TanhImpl> {
 public:
  TanhImpl() = default;
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& input) {return input.tanh();}
};
TORCH_MODULE(Tanh);

class TORCH_API SigmoidImpl : public torch::nn::Cloneable<SigmoidImpl> {
 public:
  SigmoidImpl() = default;
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& input) {return input.sigmoid();}
};
TORCH_MODULE(Sigmoid);

class TORCH_API GELUImpl : public torch::nn::Cloneable<GELUImpl> {
 public:
  GELUImpl() = default;
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {return torch::gelu(t);}
};
TORCH_MODULE(GELU);

class TORCH_API ReLUImpl : public torch::nn::Cloneable<ReLUImpl> {
 public:
  ReLUImpl() = default;
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {return t.relu();}
};
TORCH_MODULE(ReLU);

class TORCH_API SELUImpl : public torch::nn::Cloneable<SELUImpl> {
 public:
  SELUImpl() = default;
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {return torch::selu(t);}
};
TORCH_MODULE(SELU);

class TORCH_API ReLU6Impl : public torch::nn::Cloneable<ReLU6Impl> {
 public:
  ReLU6Impl() = default;
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {return torch::hardtanh(t,0.0,6.0);}
};
TORCH_MODULE(ReLU6);

// -------------------------------------------------------------
// softmax, softmin & logsoftmax activation layers
// -------------------------------------------------------------
struct TORCH_API SoftOptions {
 SoftOptions(int64_t d) : dim_(d) {}
 SoftOptions(int64_t d,c10::optional<torch::ScalarType> t) : dim_(d),dtype_(t) {}
 SoftOptions() {}
 TORCH_ARG(int64_t, dim);
 TORCH_ARG(c10::optional<torch::ScalarType>, dtype)=c10::nullopt;
};

template <typename Derived>
class SoftImpl : public torch::nn::Cloneable<Derived> {
 public:
  SoftImpl(int64_t d) : SoftImpl(SoftOptions(d)) {}
  SoftImpl(int64_t d,c10::optional<torch::ScalarType> t) : SoftImpl(SoftOptions(d,t)) {}
  explicit SoftImpl(SoftOptions o) : options(std::move(o)) {reset();}
  void reset() override {}
  SoftOptions options;
};

class TORCH_API SoftmaxImpl : public SoftImpl<SoftmaxImpl> {
 public:
  using SoftImpl<SoftmaxImpl>::SoftImpl;
  torch::Tensor forward(const torch::Tensor& input) {
   return input.softmax(options.dim(),options.dtype());
  }
};
TORCH_MODULE(Softmax);

class TORCH_API SoftminImpl : public SoftImpl<SoftminImpl> {
 public:
  using SoftImpl<SoftminImpl>::SoftImpl;
  torch::Tensor forward(const torch::Tensor& input) {
   return (-input).softmax(options.dim(),options.dtype());
  }
};
TORCH_MODULE(Softmin);

class TORCH_API LogSoftmaxImpl : public SoftImpl<LogSoftmaxImpl> {
 public:
  using SoftImpl<LogSoftmaxImpl>::SoftImpl;
  torch::Tensor forward(const torch::Tensor& input) {
   return input.log_softmax(options.dim(),options.dtype());
  }
};
TORCH_MODULE(LogSoftmax);

// -------------------------------------------------------------
//  prelu - parametric rectified linear unit
// -------------------------------------------------------------
struct TORCH_API PReLUOptions {
 PReLUOptions(int64_t n) : in_(n) {}
 PReLUOptions(int64_t n,double w) : in_(n),init_(w) {}
 PReLUOptions() {}
 TORCH_ARG(int64_t, in)=1;
 TORCH_ARG(double, init)=.25;
};

class TORCH_API PReLUImpl : public torch::nn::Cloneable<PReLUImpl> {
 public:
  PReLUImpl(int64_t n) : PReLUImpl(PReLUOptions(n)) {}
  PReLUImpl(int64_t n,double w) : PReLUImpl(PReLUOptions(n,w)) {}
  explicit PReLUImpl(PReLUOptions o) : options(std::move(o)) {reset();}
  void reset() override {
   weight=torch::nn::Module::register_parameter("weight",torch::empty({options.in()}).fill_(options.init()));
  }
  torch::Tensor forward(const torch::Tensor& t) {
   return torch::prelu(t,weight.to(t.dtype()));
  }
  PReLUOptions options;
  torch::Tensor weight;
};
TORCH_MODULE(PReLU);

// -----------------------------------------------------------------
//  elu,celu - exponential & continuously differentiable linear unit
// -----------------------------------------------------------------
struct TORCH_API ExpOptions {
 ExpOptions(torch::Scalar a) : alpha_(a) {}
 ExpOptions() {}
 TORCH_ARG(torch::Scalar, alpha)=1.0;
};

template <typename Derived>
class ExpImpl : public torch::nn::Cloneable<Derived> {
 public:
  ExpImpl(torch::Scalar a) : ExpImpl(ExpOptions(a)) {}
  explicit ExpImpl(ExpOptions o) : options(std::move(o)) {reset();}
  ExpImpl() = default;
  void reset() override {}
  ExpOptions options;
};

class TORCH_API ELUImpl : public ExpImpl<ELUImpl> {
 public:
  using ExpImpl<ELUImpl>::ExpImpl;
  torch::Tensor forward(const torch::Tensor& t) {
   return torch::elu(t,options.alpha());
  }
};
TORCH_MODULE(ELU);

class TORCH_API CELUImpl : public ExpImpl<CELUImpl> {
 public:
  using ExpImpl<CELUImpl>::ExpImpl;
  torch::Tensor forward(const torch::Tensor& t) {
   return torch::celu(t,options.alpha());
  }
};
TORCH_MODULE(CELU);

// -----------------------------------------------------------------
// leakyrelu - allow a small positive gradient(slope) when x<0
// -----------------------------------------------------------------
struct TORCH_API LeakyOptions {
 LeakyOptions(torch::Scalar s) : slope_(s) {}
 LeakyOptions() {}
 TORCH_ARG(torch::Scalar, slope)=0.01;
};

class TORCH_API LeakyReLUImpl : public torch::nn::Cloneable<LeakyReLUImpl> {
 public:
  LeakyReLUImpl(torch::Scalar s) : LeakyReLUImpl(LeakyOptions(s)) {}
  explicit LeakyReLUImpl(LeakyOptions o) : options(std::move(o)) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {
   return torch::leaky_relu(t,options.slope());
  }
 LeakyOptions options;
};
TORCH_MODULE(LeakyReLU);

// -----------------------------------------------------------------------------
// rrelu - randomized leakyrelu w'uniform random slope within given lo,hi bounds
// -----------------------------------------------------------------------------
struct TORCH_API RReLUOptions {
 RReLUOptions(torch::Scalar l,torch::Scalar u) : lower_(l),upper_(u) {}
 RReLUOptions() {}
 TORCH_ARG(torch::Scalar, lower)=1.0 / 8.0;
 TORCH_ARG(torch::Scalar, upper)=1.0 / 3.0;
};

class TORCH_API RReLUImpl : public torch::nn::Cloneable<RReLUImpl> {
 public:
  RReLUImpl(torch::Scalar l,torch::Scalar u) : RReLUImpl(RReLUOptions(l,u)) {}
  explicit RReLUImpl(RReLUOptions o) : options(std::move(o)) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {
   return torch::rrelu(t,options.lower(),options.upper(),this->is_training());
  }
 RReLUOptions options;
};
TORCH_MODULE(RReLU);

// -----------------------------------------------------------------------------
// glu - gated linear unit (splitting input along selected dimension)
// -----------------------------------------------------------------------------
struct TORCH_API GLUOptions {
 GLUOptions(int64_t d) : dim_(d) {}
 GLUOptions() {}
 TORCH_ARG(int64_t, dim)=-1;
};

class TORCH_API GLUImpl : public torch::nn::Cloneable<GLUImpl> {
 public:
  GLUImpl() = default;
  GLUImpl(int64_t d) : GLUImpl(GLUOptions(d)) {}
  explicit GLUImpl(GLUOptions o) : options(std::move(o)) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {return torch::glu(t,options.dim());}
  GLUOptions options;
};
TORCH_MODULE(GLU);

// -----------------------------------------------------------------------------
// threshold - thresholds each element of input tensor
// -----------------------------------------------------------------------------
struct TORCH_API ThresholdOptions {
 ThresholdOptions(torch::Scalar t,torch::Scalar v) : threshold_(t),value_(v) {}
 ThresholdOptions() {}
 TORCH_ARG(torch::Scalar, threshold)=0;
 TORCH_ARG(torch::Scalar, value)=0;
};

class TORCH_API ThresholdImpl : public torch::nn::Cloneable<ThresholdImpl> {
 public:
  ThresholdImpl(torch::Scalar t,torch::Scalar v) : ThresholdImpl(ThresholdOptions(t,v)) {}
  explicit ThresholdImpl(ThresholdOptions o) : options(std::move(o)) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {
   return torch::threshold(t,options.threshold(),options.value());
  }
  ThresholdOptions options;
};
TORCH_MODULE(Threshold);

// -----------------------------------------------------------------------------
// softplus - smooth approximation to relu, can constrain to always be positive
// -----------------------------------------------------------------------------
struct TORCH_API SoftplusOptions {
 SoftplusOptions(torch::Scalar b) : beta_(b) {}
 SoftplusOptions(torch::Scalar b,torch::Scalar t) : beta_(b),threshold_(t) {}
 SoftplusOptions() {}
 TORCH_ARG(torch::Scalar, beta)=1;
 TORCH_ARG(torch::Scalar, threshold)=20;
};

class TORCH_API SoftplusImpl : public torch::nn::Cloneable<SoftplusImpl> {
 public:
  SoftplusImpl(torch::Scalar b) : SoftplusImpl(SoftplusOptions(b)) {}
  SoftplusImpl(torch::Scalar b,torch::Scalar t) : SoftplusImpl(SoftplusOptions(b,t)) {}
  explicit SoftplusImpl(SoftplusOptions o) : options(std::move(o)) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {
   return torch::softplus(t,options.beta(),options.threshold());
  }
  SoftplusOptions options;
};
TORCH_MODULE(Softplus);

// -----------------------------------------------------------------------------
// hardtanh - computationally cheaper version of tanh, straight line at min,max
// -----------------------------------------------------------------------------
struct TORCH_API HardtanhOptions {
 HardtanhOptions(torch::Scalar a,torch::Scalar b) : min_(a),max_(b) {}
 HardtanhOptions() {}
 TORCH_ARG(torch::Scalar, min)=-1;
 TORCH_ARG(torch::Scalar, max)=1;
};

class TORCH_API HardtanhImpl : public torch::nn::Cloneable<HardtanhImpl> {
 public:
  HardtanhImpl(torch::Scalar a,torch::Scalar b) : HardtanhImpl(HardtanhOptions(a,b)) {}
  explicit HardtanhImpl(HardtanhOptions o) : options(std::move(o)) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {
   return torch::hardtanh(t,options.min(),options.max());
  }
  HardtanhOptions options;
};
TORCH_MODULE(Hardtanh);

// -------------------------------------------------------------
//  hardshrink, softshrink
// -------------------------------------------------------------
struct TORCH_API ShrinkOptions {
 ShrinkOptions(torch::Scalar a) : lambda_(a) {}
 ShrinkOptions() {}
 TORCH_ARG(torch::Scalar, lambda)=0.5;
};

template <typename Derived>
class ShrinkImpl : public torch::nn::Cloneable<Derived> {
 public:
  ShrinkImpl(torch::Scalar a) : ShrinkImpl(ShrinkOptions(a)) {}
  explicit ShrinkImpl(ShrinkOptions o) : options(std::move(o)) {reset();}
  void reset() override {}
  ShrinkOptions options;
};

class TORCH_API HardshrinkImpl : public ShrinkImpl<HardshrinkImpl> {
 public:
  using ShrinkImpl<HardshrinkImpl>::ShrinkImpl;
  torch::Tensor forward(const torch::Tensor& t) {return t.hardshrink(options.lambda());}
};
TORCH_MODULE(Hardshrink);

class TORCH_API SoftshrinkImpl : public ShrinkImpl<SoftshrinkImpl> {
 public:
  using ShrinkImpl<SoftshrinkImpl>::ShrinkImpl;
  torch::Tensor forward(const torch::Tensor& t) {return torch::softshrink(t,options.lambda());}
};
TORCH_MODULE(Softshrink);

// ------------------------------------------------------------------------------------------
// unable to use torch base class for dropout, add one here for alpha & feature alpha dropout
// (using torch::nn::detail::DropoutImplBase compiles, but linker errors on reset member fn)
// ------------------------------------------------------------------------------------------
template <typename Derived> class DropoutImplBase : public torch::nn::Cloneable<Derived> {
 public:
  DropoutImplBase() {}
  explicit DropoutImplBase(torch::nn::DropoutOptions o) : options(o) {
   TORCH_CHECK(options.p() >= 0, "Dropout rate must not be less than zero");
   TORCH_CHECK(options.p() <= 1, "Dropout rate must not be greater than one");
  }
  void reset() {}
  torch::nn::DropoutOptions options;
};

class TORCH_API AlphaDropoutImpl : public DropoutImplBase<AlphaDropoutImpl> {
 public:
  using DropoutImplBase<AlphaDropoutImpl>::DropoutImplBase;
  torch::Tensor forward(const torch::Tensor& t) {return torch::alpha_dropout(t,options.p(),this->is_training());}
};
TORCH_MODULE(AlphaDropout);

class TORCH_API FeatureAlphaDropoutImpl : public DropoutImplBase<FeatureAlphaDropoutImpl> {
 public:
  using DropoutImplBase<FeatureAlphaDropoutImpl>::DropoutImplBase;
  torch::Tensor forward(const torch::Tensor& t) {return torch::feature_alpha_dropout(t,options.p(),this->is_training());}
};
TORCH_MODULE(FeatureAlphaDropout);

// -------------------------------------------------------------
//  flatten - typically from 2nd dimension on
// -------------------------------------------------------------
struct TORCH_API FlattenOptions {
 FlattenOptions(int64_t s=1,int64_t e=-1) : start_dim_(s),end_dim_(e) {}
 TORCH_ARG(int64_t, start_dim);
 TORCH_ARG(int64_t,   end_dim);
};

class FlattenImpl : public torch::nn::Cloneable<FlattenImpl> {
 public:
  FlattenImpl(int64_t s=1,int64_t e=-1) : FlattenImpl(FlattenOptions(s,e)) {}
  explicit FlattenImpl(const FlattenOptions& o) : options(o) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {
   return torch::flatten(t,options.start_dim(),options.end_dim());
  };
  FlattenOptions options;
};
TORCH_MODULE(Flatten);

// -------------------------------------------------------------
//  squeeze - remove dimension(s) from tensor
//  unsqueeze - add dimension to tensor
// -------------------------------------------------------------
struct TORCH_API SqueezeOptions {
 SqueezeOptions(int64_t d,bool b=false) : dim_(d),inplace_(b) {}
 SqueezeOptions() {}
 TORCH_ARG(c10::optional<int64_t>, dim) = c10::nullopt;
 TORCH_ARG(bool, inplace) = false;
};

class SqueezeImpl : public torch::nn::Cloneable<SqueezeImpl> {
 public:
  SqueezeImpl(int64_t d,bool b=false) : SqueezeImpl(SqueezeOptions(d,b)) {}
  SqueezeImpl() : SqueezeImpl(SqueezeOptions()) {}
  explicit SqueezeImpl(const SqueezeOptions& o) : options(o) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& t) {
   if(options.dim().has_value()) {
    if(options.inplace())
     return t.squeeze_(options.dim().value());
    else
     return t.squeeze(options.dim().value());
   } else {
    if(options.inplace())
     return t.squeeze_();
    else
     return t.squeeze();
   }
  };
  SqueezeOptions options;
};
TORCH_MODULE(Squeeze);

class UnsqueezeImpl : public torch::nn::Cloneable<UnsqueezeImpl> {
 public:
  UnsqueezeImpl(int64_t d,bool b=false) : UnsqueezeImpl(SqueezeOptions(d,b)) {}
  explicit UnsqueezeImpl(const SqueezeOptions& o) : options(o) {reset();}
  void reset() override {TORCH_CHECK(options.dim().has_value(),"unsqueeze: no dimension given");}
  torch::Tensor forward(const torch::Tensor& t) {
   if(options.inplace())
    return t.unsqueeze_(options.dim().value());
   else
    return t.unsqueeze(options.dim().value());
  };
  SqueezeOptions options;
};
TORCH_MODULE(Unsqueeze);

// -------------------------------------------------------------
// expand & reshape - modules with size options
// -------------------------------------------------------------
struct TORCH_API SizeOptions {
 SizeOptions(std::vector<int64_t> s) : size_(std::move(s)) {}
 TORCH_ARG(std::vector<int64_t>, size);
};

class ExpandImpl : public torch::nn::Cloneable<ExpandImpl> {
 public:
 ExpandImpl(std::vector<int64_t> s) : ExpandImpl(SizeOptions(s)) {}
 explicit ExpandImpl(const SizeOptions& o) : options(o) {reset();}
 void reset() override {}
 torch::Tensor forward(const torch::Tensor& t) { return t.expand(options.size());}
 SizeOptions options;
};
TORCH_MODULE(Expand);

class ReshapeImpl : public torch::nn::Cloneable<ReshapeImpl> {
 public:
 ReshapeImpl(std::vector<int64_t> s) : ReshapeImpl(SizeOptions(s)) {}
 explicit ReshapeImpl(const SizeOptions& o) : options(o) {reset();}
 void reset() override {}
 torch::Tensor forward(const torch::Tensor& t) { return t.reshape(options.size());}
 SizeOptions options;
};
TORCH_MODULE(Reshape);
