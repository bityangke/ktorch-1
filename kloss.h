#pragma once

class TORCH_API Loss {
 public:
  virtual ~Loss() = default;
  virtual torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& y) {
   AT_ERROR("Loss function of 2 tensors, e.g. input & target, not implemented");
  }
  virtual torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z) {
   AT_ERROR("Loss function of 3 tensors, e.g. input,target,weights or input1,input2,target, is not implemented");
  }
  virtual torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y,const torch::Tensor& nx,const torch::Tensor& ny) {
   AT_ERROR("Loss function of 4 tensors, e.g. input,target,input lengths,target lengths, is not implemented");
  }
  virtual torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y,torch::IntArrayRef nx,torch::IntArrayRef ny) {
   AT_ERROR("Loss function of 4 args, e.g. input,target,array ref of input lengths,target lengths, is not implemented");
  }
  virtual void to(torch::Device device,torch::Dtype dtype,bool non_blocking = false) {}
  virtual void to(torch::Dtype dtype, bool non_blocking = false) {}
  virtual void to(torch::Device device, bool non_blocking = false) {}
};

static inline void Reduce(int64_t r) {
 TORCH_CHECK(r==Reduction::None || r==Reduction::Mean || r==Reduction::Sum,
  "Reduction is 0,1,2 for none,mean,sum, unrecognized setting: ",r);
}

// ----------------------------------------------------------------------
// loss functions w'single option of reduction: none,mean,sum <-> 0,1,2
// ----------------------------------------------------------------------
struct TORCH_API LossOptions {
 LossOptions(int64_t r=Reduction::Mean) : reduce_(r) {}
 TORCH_ARG(int64_t, reduce);
};

class TORCH_API BasicLoss : public Loss {
 public:
  BasicLoss(LossOptions o=LossOptions()) : options(o) {Reduce(options.reduce_);} 
  BasicLoss(int64_t r) : BasicLoss(LossOptions(r)) {}
  LossOptions options;
};

class TORCH_API BCELoss : public BasicLoss {
 public:
 using BasicLoss::BasicLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y,const torch::Tensor& w={}) {
  return torch::binary_cross_entropy(x,y,w,options.reduce_);
 }
};

class TORCH_API KLDivLoss : public BasicLoss {
 public:
 using BasicLoss::BasicLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::kl_div(x,y,options.reduce_);
 }
};

class TORCH_API L1Loss : public BasicLoss {
 public:
 using BasicLoss::BasicLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::l1_loss(x,y,options.reduce_);
 }
};

class TORCH_API MSELoss : public BasicLoss {
 public:
 using BasicLoss::BasicLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::mse_loss(x,y,options.reduce_);
 }
};

class TORCH_API MultiLabelMarginLoss : public BasicLoss {
 public:
 using BasicLoss::BasicLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::multilabel_margin_loss(x,y,options.reduce_);
 }
};

class TORCH_API SmoothL1Loss : public BasicLoss {
 public:
 using BasicLoss::BasicLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::smooth_l1_loss(x,y,options.reduce_);
 }
};

class TORCH_API SoftMarginLoss : public BasicLoss {
 public:
 using BasicLoss::BasicLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::soft_margin_loss(x,y,options.reduce_);
 }
};

// --------------------------------------------------------------------------
// loss functions w'option of class weights along w'reduction: none,mean,sum
// binary cross entropy w'logits & multi-label soft margin loss
// --------------------------------------------------------------------------
struct TORCH_API WeightedLossOptions {
 WeightedLossOptions(const torch::Tensor& w,int64_t r=Reduction::Mean) : weight_(std::move(w)), reduce_(r) {}
 WeightedLossOptions(int64_t r=Reduction::Mean) : reduce_(r) {}
 TORCH_ARG(torch::Tensor, weight)={};
 TORCH_ARG(int64_t, reduce);
};

class TORCH_API WeightedLoss : public Loss {
 public:
  WeightedLoss(WeightedLossOptions o=WeightedLossOptions()) : options(std::move(o)) {Reduce(options.reduce_);} 
  WeightedLoss(int64_t r) : WeightedLoss(WeightedLossOptions(r)) {}
  WeightedLoss(const torch::Tensor& w,int64_t r) : WeightedLoss(WeightedLossOptions(w,r)) {}
  WeightedLossOptions options;
  void to(torch::Device device,torch::Dtype dtype,bool non_blocking = false) {options.weight_.to(device,dtype,non_blocking);}
  void to(torch::Dtype dtype, bool non_blocking = false) {options.weight_.to(dtype,non_blocking);}
  void to(torch::Device device, bool non_blocking = false) {options.weight_.to(device,non_blocking);}
};

class TORCH_API BCEWithLogitsLoss : public WeightedLoss {
 public:
 using WeightedLoss::WeightedLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y,const torch::Tensor& w={}) {
  return torch::binary_cross_entropy_with_logits(x,y,w,options.weight_,options.reduce_);
 }
};

torch::Tensor multilabel_soft_margin_loss(const torch::Tensor& x,const torch::Tensor& y,const torch::Tensor& w={},int64_t r=Reduction::Mean);

torch::Tensor multilabel_soft_margin_loss(const torch::Tensor& x,const torch::Tensor& y,const torch::Tensor& w,int64_t r) {
 auto l = -(y * torch::log_sigmoid(x) + (1 - y) * torch::log_sigmoid(-x));
 if(w.defined()) l *= w;
 l = l.sum(1) / x.size(1); // only return n=batch size loss values
 switch(r) {
  case Reduction::None: return l;
  case Reduction::Mean: return l.mean();
  case Reduction::Sum:  return l.sum();
  default: AT_ERROR("Unrecognized reduction: ",r);
 }
 // unable to use torch::apply_loss_reduction(l,r), in anonymous namespace in ATen/native/Loss.cpp
}

class TORCH_API MultiLabelSoftMarginLoss : public WeightedLoss {
 public:
 using WeightedLoss::WeightedLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return multilabel_soft_margin_loss(x,y,options.weight_,options.reduce_);
 }
};

// ------------------------------------------------------------------------------------
// loss functions w'option of class weights, index to ignore & reduction: none,mean,sum
// negative log likelihood & cross entropy (includes log softmax of inputs)
// ------------------------------------------------------------------------------------
struct TORCH_API LogLossOptions {
 LogLossOptions(const torch::Tensor& w,int64_t i=-100,int64_t r=Reduction::Mean) : weight_(std::move(w)), ignore_(i), reduce_(r) {}
 LogLossOptions(int64_t r=Reduction::Mean) : reduce_(r) {}
 TORCH_ARG(torch::Tensor, weight)={};
 TORCH_ARG(int64_t, ignore)=-100;
 TORCH_ARG(int64_t, reduce);
};

class TORCH_API LogLoss : public Loss {
 public:
  LogLoss(LogLossOptions o=LogLossOptions()) : options(std::move(o)) {Reduce(options.reduce_);} 
  LogLoss(int64_t r) : LogLoss(LogLossOptions(r)) {}
  LogLoss(const torch::Tensor& w,int64_t i,int64_t r) : LogLoss(LogLossOptions(w,i,r)) {}
  LogLossOptions options;
  void to(torch::Device device,torch::Dtype dtype,bool non_blocking = false) {options.weight_.to(device,dtype,non_blocking);}
  void to(torch::Dtype dtype, bool non_blocking = false) {options.weight_.to(dtype,non_blocking);}
  void to(torch::Device device, bool non_blocking = false) {options.weight_.to(device,non_blocking);}
};

class TORCH_API NLLLoss : public LogLoss {
 public:
 using LogLoss::LogLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::nll_loss(x,y,options.weight_,options.reduce_,options.ignore_);
 }
};

class TORCH_API CrossEntropyLoss : public LogLoss {
 public:
 using LogLoss::LogLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::nll_loss(torch::log_softmax(x,1),y,options.weight_,options.reduce_,options.ignore_);
 }
};

// ------------------------------------------------------------------------------------
// loss functions w'options consisting of margin & reduction method:
// hinge embedding loss, cosine embedding loss & margin ranking loss
// ------------------------------------------------------------------------------------
struct TORCH_API MarginLossOptions {
 MarginLossOptions(int64_t r) : reduce_(r) {}
 MarginLossOptions(double m=0.0, int64_t r=Reduction::Mean) : margin_(m), reduce_(r) {}
 TORCH_ARG(double,  margin)=0.0;
 TORCH_ARG(int64_t, reduce)=Reduction::Mean;
};

class TORCH_API MarginLoss : public Loss {
 public:
  MarginLoss(MarginLossOptions o=MarginLossOptions()) : options(std::move(o)) {Reduce(options.reduce_);} 
  MarginLoss(int64_t r) : MarginLoss(MarginLossOptions(r)) {}
  MarginLoss(double m,int64_t r) : MarginLoss(MarginLossOptions(m,r)) {}
  MarginLossOptions options;
};

class TORCH_API HingeEmbeddingLoss : public MarginLoss {
 public:
 using MarginLoss::MarginLoss;
 torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
  return torch::hinge_embedding_loss(x,y,options.margin_,options.reduce_);
 }
};

class TORCH_API CosineEmbeddingLoss : public MarginLoss {
 public:
 using MarginLoss::MarginLoss;
 torch::Tensor forward(const torch::Tensor& x1,const torch::Tensor& x2,const torch::Tensor& y) {
  return torch::cosine_embedding_loss(x1,x2,y,options.margin_,options.reduce_);
 }
};

class TORCH_API MarginRankingLoss : public MarginLoss {
 public:
 using MarginLoss::MarginLoss;
 torch::Tensor forward(const torch::Tensor& x1,const torch::Tensor& x2,const torch::Tensor& y) {
  return torch::margin_ranking_loss(x1,x2,y,options.margin_,options.reduce_);
 }
};

// ------------------------------------------------------------------------------------
//  multi margin loss w'options for power,margin,weights & reduction method
// ------------------------------------------------------------------------------------
struct TORCH_API MultiMarginLossOptions {
 MultiMarginLossOptions(Scalar p=1,Scalar m=1.0,const torch::Tensor& w={},int64_t r=Reduction::Mean) : p_(p),margin_(m),weight_(std::move(w)),reduce_(r) {}
 MultiMarginLossOptions(const torch::Tensor& w,int64_t r=Reduction::Mean) : weight_(std::move(w)), reduce_(r) {}
 MultiMarginLossOptions(int64_t r) : reduce_(r) {}
 TORCH_ARG(Scalar,        p)=1;
 TORCH_ARG(Scalar,        margin)=1.0;
 TORCH_ARG(torch::Tensor, weight)={};
 TORCH_ARG(int64_t,       reduce)=Reduction::Mean;
};

class TORCH_API MultiMarginLoss : public Loss {
 public:
  MultiMarginLoss(MultiMarginLossOptions o=MultiMarginLossOptions()) : options(std::move(o)) {Reduce(options.reduce_);} 
  MultiMarginLoss(int64_t r) : MultiMarginLoss(MultiMarginLossOptions(r)) {}
  MultiMarginLoss(const torch::Tensor& w,int64_t r) : MultiMarginLoss(MultiMarginLossOptions(w,r)) {}
  MultiMarginLoss(Scalar p,Scalar m,const torch::Tensor& w,int64_t r) : MultiMarginLoss(MultiMarginLossOptions(p,m,w,r)) {}
  MultiMarginLossOptions options;
  torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
   return torch::multi_margin_loss(x,y,options.p_,options.margin_,options.weight_,options.reduce_);
  }
  void to(torch::Device device,torch::Dtype dtype,bool non_blocking = false) {options.weight_.to(device,dtype,non_blocking);}
  void to(torch::Dtype dtype, bool non_blocking = false) {options.weight_.to(dtype,non_blocking);}
  void to(torch::Device device, bool non_blocking = false) {options.weight_.to(device,non_blocking);}
};

// ------------------------------------------------------------------------------------
//  triplet margin loss w'options for margin,p,eps,swap & reduction method
// ------------------------------------------------------------------------------------
struct TORCH_API TripletLossOptions {
 TripletLossOptions(double m=1.0,double p=2.0,double e=1e-06,bool s=false,int64_t r=Reduction::Mean) : margin_(m),p_(p),eps_(e),swap_(s),reduce_(r) {}
 TripletLossOptions(int64_t r) : reduce_(r) {}
 TORCH_ARG(double,  margin)=1.0;
 TORCH_ARG(double,  p)=2.0;
 TORCH_ARG(double,  eps)=1e-06;
 TORCH_ARG(bool,    swap)=false;
 TORCH_ARG(int64_t, reduce)=Reduction::Mean;
};

class TORCH_API TripletMarginLoss : public Loss {
 public:
  TripletMarginLoss(TripletLossOptions o=TripletLossOptions()) : options(std::move(o)) {Reduce(options.reduce_);} 
  TripletMarginLoss(int64_t r) : TripletMarginLoss(TripletLossOptions(r)) {}
  TripletMarginLoss(double m,double p,double e,bool s,int64_t r) : TripletMarginLoss(TripletLossOptions(m,p,e,s,r)) {}
  TripletLossOptions options;
  torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y,const torch::Tensor& z) {
   return torch::triplet_margin_loss(x,y,z,options.margin_,options.p_,options.eps_,options.swap_,options.reduce_);
  }
};


// ------------------------------------------------------------------------------------------------------
// poisson nll loss w'options for log input,full loss w'stirling approximation,epsilon & reduction method
// ------------------------------------------------------------------------------------------------------
struct TORCH_API PoissonLossOptions {
 PoissonLossOptions(bool l=true,bool f=false,double e=1e-08,int64_t r=Reduction::Mean) : log_(l),full_(f),eps_(e),reduce_(r) {}
 PoissonLossOptions(int64_t r) : reduce_(r) {}
 TORCH_ARG(bool,    log)=true;
 TORCH_ARG(bool,    full)=false;
 TORCH_ARG(double,  eps)=1e-08;
 TORCH_ARG(int64_t, reduce)=Reduction::Mean;
};

class TORCH_API PoissonNLLLoss : public Loss {
 public:
  PoissonNLLLoss(PoissonLossOptions o=PoissonLossOptions()) : options(std::move(o)) {Reduce(options.reduce_);} 
  PoissonNLLLoss(int64_t r) : PoissonNLLLoss(PoissonLossOptions(r)) {}
  PoissonNLLLoss(bool l,bool f,double e,int64_t r) : PoissonNLLLoss(PoissonLossOptions(l,f,e,r)) {}
  PoissonLossOptions options;
  torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y) {
   return torch::poisson_nll_loss(x,y,options.log_,options.full_,options.eps_,options.reduce_);
  }
};

// ------------------------------------------------------------------------------------------------------
// ctc - connectionist temporal classification loss between continuous time series and a target sequence
// ------------------------------------------------------------------------------------------------------
struct TORCH_API CTCLossOptions {
 CTCLossOptions(int64_t b=0,bool z=false,int64_t r=Reduction::Mean) : blank_(b),zeroinf_(z),reduce_(r) {}
 TORCH_ARG(int64_t, blank)=0;
 TORCH_ARG(bool,    zeroinf)=false;
 TORCH_ARG(int64_t, reduce)=Reduction::Mean;
};

class TORCH_API CTCLoss : public Loss {
 public:
  CTCLoss(CTCLossOptions o=CTCLossOptions()) : options(std::move(o)) {Reduce(options.reduce_);} 
  CTCLoss(int64_t b,bool z,int64_t r) : CTCLoss(CTCLossOptions(b,z,r)) {}
  CTCLossOptions options;
  torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y,const torch::Tensor& nx,const torch::Tensor& ny) {
   return torch::ctc_loss(x,y,nx,ny,options.blank_,options.reduce_,options.zeroinf_);
  }
  torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y,torch::IntArrayRef nx,torch::IntArrayRef ny) {
   return torch::ctc_loss(x,y,nx,ny,options.blank_,options.reduce_,options.zeroinf_);
  }
};
