#pragma once
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
