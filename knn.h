#pragma once
// -------------------------------------------------------------------------------------------
// fractional max pool 2,3d
// defined here due to bug with 1.4 versions, output ratio defined as integer instead of float
// -------------------------------------------------------------------------------------------
template <size_t D>
struct FractionalMaxPoolOptions {
  FractionalMaxPoolOptions(ExpandingArray<D> kernel_size) : kernel_size_(kernel_size) {}
  TORCH_ARG(ExpandingArray<D>, kernel_size);
  TORCH_ARG(c10::optional<ExpandingArray<D>>, output_size) = c10::nullopt;
  using ExpandingArrayDouble=torch::ExpandingArray<D,double>;
  TORCH_ARG(c10::optional<ExpandingArrayDouble>, output_ratio) = c10::nullopt;
  TORCH_ARG(torch::Tensor, _random_samples) = Tensor();
};

using FractionalMaxPool2dOptions = FractionalMaxPoolOptions<2>;
using FractionalMaxPool3dOptions = FractionalMaxPoolOptions<3>;
TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(FractionalMaxPool2d, FractionalMaxPool2dFuncOptions)
TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(FractionalMaxPool3d, FractionalMaxPool3dFuncOptions)

namespace functional {
namespace detail {
 inline std::tuple<Tensor, Tensor> fractional_max_pool2d_with_indices(
    const Tensor& input,
    const ExpandingArray<2>& kernel_size,
    const c10::optional<ExpandingArray<2>>& output_size,
    const c10::optional<ExpandingArray<2,double>>& output_ratio,
    const Tensor& _random_samples) {
   if (output_size == c10::nullopt && output_ratio == c10::nullopt) {
     TORCH_CHECK(false, "fractional_max_pool2d requires specifying either an output_size or an output_ratio");
   }

   c10::optional<ExpandingArray<2>> output_size_ = output_size;
   if (output_size_ == c10::nullopt) {
    TORCH_INTERNAL_ASSERT(output_ratio != c10::nullopt,"no output ratios defined");
    output_size_ = {(int64_t)(input.sizes()[2] * (*output_ratio.value())[0]),
                    (int64_t)(input.sizes()[3] * (*output_ratio.value())[1])};
   }

   Tensor _random_samples_ = _random_samples;
   if (!_random_samples_.defined()) {
     _random_samples_ = torch::rand({input.sizes()[0], input.sizes()[1], 2}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));
   }
   return torch::fractional_max_pool2d(input, kernel_size, *output_size_, _random_samples_);
 }
 
 inline Tensor fractional_max_pool2d(const Tensor& input,
                                    ExpandingArray<2> kernel_size,
                                    c10::optional<ExpandingArray<2>> output_size,
                                    c10::optional<ExpandingArray<2,double>> output_ratio,
                                    const Tensor& _random_samples) {
   return std::get<0>(fractional_max_pool2d_with_indices(input, kernel_size, output_size, output_ratio, _random_samples));
 }
}
 inline std::tuple<Tensor, Tensor> fractional_max_pool2d_with_indices(const Tensor& input, const FractionalMaxPool2dFuncOptions& options) {
  return detail::fractional_max_pool2d_with_indices(input,options.kernel_size(),options.output_size(),options.output_ratio(),options._random_samples());
 }

 inline Tensor fractional_max_pool2d(const Tensor& input, const FractionalMaxPool2dFuncOptions& options) {
  return detail::fractional_max_pool2d(input,options.kernel_size(),options.output_size(),options.output_ratio(),options._random_samples());
 }
}

class TORCH_API FractionalMaxPool2dImpl : public torch::nn::Cloneable<FractionalMaxPool2dImpl> {
 public:
  FractionalMaxPool2dImpl(ExpandingArray<2> kernel_size) : FractionalMaxPool2dImpl(FractionalMaxPool2dOptions(kernel_size)) {}
  explicit FractionalMaxPool2dImpl(const FractionalMaxPool2dOptions& options_) : options(options_) {reset();}

  void reset() override {
   _random_samples = register_buffer("_random_samples", options._random_samples());
   if (options.output_size() == c10::nullopt && options.output_ratio() == c10::nullopt) {
    TORCH_CHECK(false, "FractionalMaxPool2d requires specifying either an output size, or a pooling ratio");
   }
   if (options.output_size() != c10::nullopt && options.output_ratio() != c10::nullopt) {
    TORCH_CHECK(false, "only one of output_size and output_ratio may be specified");
   }
   if (options.output_ratio() != c10::nullopt) {
    at::ArrayRef<double> output_ratio = at::ArrayRef<double>(options.output_ratio().value());
    if (!(0 < output_ratio[0] && output_ratio[0] < 1 &&
          0 < output_ratio[1] && output_ratio[1] < 1)) {
      TORCH_CHECK(false, "output_ratio must be between 0 and 1 (got ", output_ratio, ")");
    }
   }
  }

  void pretty_print(std::ostream& stream) const override {stream << "torch::nn::FractionalMaxPool2d()";}

  Tensor forward(const Tensor& input) {
   return functional::detail::fractional_max_pool2d(
           input, options.kernel_size(), options.output_size(), options.output_ratio(), _random_samples);
  }

  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input) {
   return functional::detail::fractional_max_pool2d_with_indices(
           input, options.kernel_size(), options.output_size(), options.output_ratio(), _random_samples);
  }

  FractionalMaxPool2dOptions options;
  Tensor _random_samples;
};
TORCH_MODULE(FractionalMaxPool2d);

namespace functional {
namespace detail {
 inline std::tuple<Tensor, Tensor> fractional_max_pool3d_with_indices(
    const Tensor& input,
    const ExpandingArray<3>& kernel_size,
    const c10::optional<ExpandingArray<3>>& output_size,
    const c10::optional<ExpandingArray<3,double>>& output_ratio,
    const Tensor& _random_samples) {
   if (output_size == c10::nullopt && output_ratio == c10::nullopt) {
     TORCH_CHECK(false,"fractional_max_pool3d requires specifying either an output_size or an output_ratio");
   }

   c10::optional<ExpandingArray<3>> output_size_ = output_size;
   if (output_size_ == c10::nullopt) {
     TORCH_INTERNAL_ASSERT(output_ratio != c10::nullopt,"no output ratios defined");
     output_size_ = {(int64_t)(input.sizes()[2] * (*output_ratio.value())[0]),
                     (int64_t)(input.sizes()[3] * (*output_ratio.value())[1]),
                     (int64_t)(input.sizes()[4] * (*output_ratio.value())[2])};
   }

   Tensor _random_samples_ = _random_samples;
   if (!_random_samples_.defined()) {
    _random_samples_ = torch::rand({input.size(0), input.size(1), 3}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));
   }
   return torch::fractional_max_pool3d(input, kernel_size, *output_size, _random_samples_);
 }

 inline Tensor fractional_max_pool3d(const Tensor& input,
                                    ExpandingArray<3> kernel_size,
                                    c10::optional<ExpandingArray<3>> output_size,
                                    c10::optional<ExpandingArray<3,double>> output_ratio,
                                    const Tensor& _random_samples) {
  return std::get<0>(fractional_max_pool3d_with_indices(input, kernel_size, output_size, output_ratio, _random_samples));
 }
}

inline std::tuple<Tensor, Tensor> fractional_max_pool3d_with_indices(const Tensor& input, const FractionalMaxPool3dFuncOptions& options) {
  return detail::fractional_max_pool3d_with_indices(input, options.kernel_size(), options.output_size(), options.output_ratio(), options._random_samples());
 }

inline Tensor fractional_max_pool3d(const Tensor& input, const FractionalMaxPool3dFuncOptions& options) {
  return detail::fractional_max_pool3d(input, options.kernel_size(), options.output_size(), options.output_ratio(), options._random_samples());
 }
}

class TORCH_API FractionalMaxPool3dImpl : public torch::nn::Cloneable<FractionalMaxPool3dImpl> {
 public:
  FractionalMaxPool3dImpl(ExpandingArray<3> kernel_size) : FractionalMaxPool3dImpl(FractionalMaxPool3dOptions(kernel_size)) {}
  explicit FractionalMaxPool3dImpl(const FractionalMaxPool3dOptions& options_) : options(options_) {reset();}

  void reset() override {
   _random_samples = register_buffer("_random_samples", options._random_samples());
   if (options.output_size() == c10::nullopt && options.output_ratio() == c10::nullopt) {
    TORCH_CHECK(false,"FractionalMaxPool3d requires specifying either an output size, or a pooling ratio");
   }
   if (options.output_size() != c10::nullopt && options.output_ratio() != c10::nullopt) {
     TORCH_CHECK(false, "only one of output_size and output_ratio may be specified");
   }
   if (options.output_ratio() != c10::nullopt) {
    at::ArrayRef<double> output_ratio = at::ArrayRef<double>(options.output_ratio().value());
    if (!(0 < output_ratio[0] && output_ratio[0] < 1 &&
          0 < output_ratio[1] && output_ratio[1] < 1 &&
          0 < output_ratio[2] && output_ratio[2] < 1)) {
      TORCH_CHECK(false, "output_ratio must be between 0 and 1 (got ", output_ratio, ")");
    }
   }
  }

  void pretty_print(std::ostream& stream) const override {stream << "torch::nn::FractionalMaxPool3d()";}

  Tensor forward(const Tensor& input) {
   return functional::detail::fractional_max_pool3d(input, options.kernel_size(), options.output_size(), options.output_ratio(),_random_samples);
  }

  std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input) {
  return functional::detail::fractional_max_pool3d_with_indices(input,options.kernel_size(),options.output_size(),options.output_ratio(),_random_samples);
  }

  FractionalMaxPool3dOptions options;
  Tensor _random_samples;
};
TORCH_MODULE(FractionalMaxPool3d);

// -----------------------------------------------------------------------------------------
//  transposed convolutions forward fn requires 2 args: input & optional output size
//   as of version 1.4, a Sequential module can only handle forward fn w'single arg
//  modules below use same forward call without defining output size (default behaviour)
//  this allows transposed convolutions to be used in Sequential modules
// -----------------------------------------------------------------------------------------
class TORCH_API ConvTranspose1dImpl : public torch::nn::ConvTranspose1dImpl {
 public:
 ConvTranspose1dImpl(torch::nn::ConvTranspose1dOptions o) : torch::nn::ConvTranspose1dImpl(o) {}
 torch::Tensor forward(const torch::Tensor& input) {
  return torch::nn::ConvTranspose1dImpl::forward(input,c10::nullopt);
 }
};

class TORCH_API ConvTranspose2dImpl : public torch::nn::ConvTranspose2dImpl {
 public:
 ConvTranspose2dImpl(torch::nn::ConvTranspose2dOptions o) : torch::nn::ConvTranspose2dImpl(o) {}
 torch::Tensor forward(const torch::Tensor& input) {
  return torch::nn::ConvTranspose2dImpl::forward(input,c10::nullopt);
 }
};

class TORCH_API ConvTranspose3dImpl : public torch::nn::ConvTranspose3dImpl {
 public:
 ConvTranspose3dImpl(torch::nn::ConvTranspose3dOptions o) : torch::nn::ConvTranspose3dImpl(o) {}
 torch::Tensor forward(const torch::Tensor& input) {
  return torch::nn::ConvTranspose3dImpl::forward(input,c10::nullopt);
 }
};

TORCH_MODULE(ConvTranspose1d);
TORCH_MODULE(ConvTranspose2d);
TORCH_MODULE(ConvTranspose3d);

// --------------------------------------------------------------------------
// general pad: create module to match functional call with size, mode, value
// --------------------------------------------------------------------------
class TORCH_API PadImpl : public torch::nn::Cloneable<PadImpl> {
 public:
  PadImpl(std::vector<int64_t> p) : PadImpl(torch::nn::functional::PadFuncOptions(p)) {}
  explicit PadImpl(const torch::nn::functional::PadFuncOptions& o) : options(o) {reset();}
  void reset() override {}
  torch::Tensor forward(const torch::Tensor& input) {
   return torch::nn::functional::pad(input,options);
  }
  torch::nn::functional::PadFuncOptions options;
};

TORCH_MODULE(Pad);

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

  void pretty_print(std::ostream& s) const override {
   s << "Squeeze(dim="; options.dim() ? s << options.dim().value() : s << "None";
   s << ", inplace=" << options.inplace() <<")";
  }

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

  void pretty_print(std::ostream& s) const override {
   s << "Unsqueeze(dim="; options.dim() ? s << options.dim().value() : s << "None";
   s << ", inplace=" << options.inplace() <<")";
  }

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
 void pretty_print(std::ostream& s) const override {s << "Expand(size=" << options.size() << ")";}
 torch::Tensor forward(const torch::Tensor& t) { return t.expand(options.size());}
 SizeOptions options;
};
TORCH_MODULE(Expand);

class ReshapeImpl : public torch::nn::Cloneable<ReshapeImpl> {
 public:
 ReshapeImpl(std::vector<int64_t> s) : ReshapeImpl(SizeOptions(s)) {}
 explicit ReshapeImpl(const SizeOptions& o) : options(o) {reset();}
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Reshape(size=" << options.size() << ")";}
 torch::Tensor forward(const torch::Tensor& t) { return t.reshape(options.size());}
 SizeOptions options;
};
TORCH_MODULE(Reshape);

// ----------------------------------------------------------------------------------------------------
// cat - add convenience module for cat(tensors,dim)
// ----------------------------------------------------------------------------------------------------
struct TORCH_API CatOptions {
 CatOptions(int64_t d=0) : dim_(d) {}
 TORCH_ARG(int64_t, dim) = 0;
};

class CatImpl : public torch::nn::Cloneable<CatImpl> {
 public:
 CatImpl(int64_t d) {CatImpl(CatOptions(d));}
 explicit CatImpl(const CatOptions& o={}) : options(std::move(o)) {}
 void reset() override {}
 void pretty_print(std::ostream& s) const override {s << "Cat(dim=" << options.dim() << ")";}
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::cat({x,y},options.dim());
 }
 CatOptions options;
};
TORCH_MODULE(Cat);


// ----------------------------------------------------------------------------------------------------
// Join - define sequential modules for inputs x & y, and a layer for joining the output of each module
// ----------------------------------------------------------------------------------------------------
class TORCH_API JoinImpl : public torch::nn::Cloneable<JoinImpl> {
 public:
 JoinImpl(const Sequential& x,const Sequential& y,const AnyModule& z) : qx(std::move(x)),qy(std::move(y)),join(std::move(z)) {
  register_module("qx", qx);
  register_module("qy", qy);
  register_module("join", join.ptr());
  reset();
 }
 void reset() override {}

 void pretty_print(std::ostream& s) const override {s << "Join";}

 Tensor forward(const Tensor& x,const Tensor& y) {
  if(qx->size() && qy->size())
   return join.forward(qx->forward(x),qy->forward(y));
  else if(qx->size())
   return join.forward(qx->forward(x),y);
  else if(qy->size())
   return join.forward(x,qy->forward(y));
  else
   return join.forward(x,y);
 }
 Sequential qx;
 Sequential qy;
 AnyModule  join;
};
TORCH_MODULE(Join);

// ----------------------------------------------------------------------------------------------------
// Sequence - rework Sequential to accept nested sequentionals, also make push_back of AnyModule public
// ----------------------------------------------------------------------------------------------------
struct TORCH_API SequenceImpl : public torch::nn::SequentialImpl {
  using SequentialImpl::SequentialImpl;

  torch::Tensor forward(torch::Tensor x) {
    return SequentialImpl::forward(x);
  }
};
TORCH_MODULE(Sequence);
