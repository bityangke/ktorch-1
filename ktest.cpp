#include "ktorch.h"
#include "knn.h"
#include "private.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif

ACCESS_PRIVATE_FIELD(torch::nn::SequentialImpl, std::vector<AnyModule>, modules_)
// PATCH ACCESS_PRIVATE_FIELD(torch::optim::SGD, int64_t, iteration_)

#ifdef __clang__
# pragma clang diagnostic pop
#elif defined __GNUC__
# pragma GCC diagnostic pop
#endif

#define OPTION(x,k,v) dictadd(x, lset(Setting::k), v)

/*
//#include  <c10/cuda/CUDAUtils.h>
KAPI dtest(K x) {
 auto d=c10::cuda::current_device();
 std::cerr << d << "\n";
 return(K)0;
}
*/

/* PATCH
KAPI sgdtest(K x) {
 auto a=torch::optim::SGDOptions(.01);
 TensorVector v;
 auto o=torch::optim::SGD(v,a);
 std::cerr << o.iteration() << "\n";
 std::cerr << access_private::iteration_(o) << "\n";
 access_private::iteration_(o)=2;
 std::cerr << o.iteration() << "\n";
 std::cerr << access_private::iteration_(o) << "\n";
 return (K)0;
}
*/

auto childcount(const Module& m) {return m.children().size();}
auto childcount(Module* m) {std::cerr << "module ptr:\n"; return m->children().size();}
KAPI testcontainer(K x) {
 torch::nn::Linear m(1,2);
 std::cerr << childcount(*m) << "\n";
 std::cerr << childcount(m.get()) << "\n";
 return (K)0;
}

void testprint(int64_t d,const std::string s,const Module& m) {
  std::cerr << "  depth: " << d << ",";
  std::cerr << "   name: " << (s.size() ? s :  m.name());
  if(m.children().size()) {
   std::cerr << " " << m.name() << " (\n";
   for(auto& a:m.named_children())
    testprint(d+1,a.key(),*a.value());
   std::cerr << ")\n";
  } else {
  std::cerr << "\t" << m << "\n";
  }
}

KAPI testjoin(K x) {
 KTRY
  Sequential q1=Sequential(torch::nn::Embedding(10,50), torch::nn::Linear(50,784), Reshape(std::vector<int64_t>{-1,1,28,28}));
  Sequential q2;
  Cat c(1);
  Join m=Join(q1,q2,AnyModule(c));
  std::cerr << m << "\n";
  testprint(0,"",*m);
  return (K)0;
 KCATCH("testjoin");
}

void margs(Sequential& q,K x,J i);

KAPI kjoin(K x) {
 KTRY
  Sequential qx,qy,qc;
  TORCH_CHECK(!x->t, "not implemented for ",kname(x));
  TORCH_CHECK(x->n==3, "3 args expected, (sequential x, sequential y, catenating layer)");
  if(!xseq(x,0,qx)) margs(qx,kK(x)[0],0);
  if(!xseq(x,1,qy)) margs(qy,kK(x)[1],0);
  margs(qc,x,2);
  auto& m=access_private::modules_(*qc);
  auto a=Join(qx,qy,m[0]);
  auto mx=qx->modules();
  std::cerr << "qx modules:\n";
  for(auto &i:mx) std::cerr << *i << "\n";
  auto my=qy->modules();
  std::cerr << "qy modules:\n";
  for(auto &i:my) std::cerr << *i << "\n";
  std::cerr << "main x<->y module:\n" << a << "\n";
  return (K)0;
 KCATCH("join");
}

KAPI stest(K x) {
 auto a=AnyModule(torch::nn::Linear(2,3));
 Sequential q;
 auto& m=access_private::modules_(*q);
 std::cerr << m.size() << "\n";
 m.push_back(a);
 std::cerr << m.size() << "\n";
 (*q).register_module("test",m[0].ptr());
 std::cerr << q << "\n";
 return(K)0;
}

static K metrics(K x) {return (x->t==KS || x->t==-KS) ? x : nullptr;}

KAPI metrictest(K x) {
 KTRY
  TORCH_CHECK(!x->t && x->n>2, "bad arg(s)");
  K z=metrics(kK(x)[x->n-1]);
  if(z) {
   for(J i=0,n=z->t==KS ? z->n : 1; i<n; ++i) {
    S s=z->t==KS ? kS(z)[i] : z->s;
    //std::cerr << s << " -> " (I)metric(s) << "\n";
    std::cerr << s << "\n";
   }
  } else {
  }
  return (K)0;
 KCATCH("metrics");
}

KAPI ksizes(K x) {
 std::cerr << "Class:   " << sizeof(Class) << "\n";
 std::cerr << "Cast:    " << sizeof(Cast) << "\n";
 std::cerr << "Ktag:    " << sizeof(Ktag) << "\n";
 std::cerr << "Kten:    " << sizeof(Kten) << "\n";
 std::cerr << "Kvec:    " << sizeof(Kvec) << "\n";
 std::cerr << "Kmodule: " << sizeof(Kmodule) << "\n";
 std::cerr << "Kopt:    " << sizeof(Kopt) << "\n";
 std::cerr << "Kseq:    " << sizeof(Kseq) << "\n";
 std::cerr << "Kmodel:  " << sizeof(Kmodel) << "\n";
 return (K)0;
}

KAPI randint_type(K x) {
 std::cerr << torch::randint(10,{3}) << "\n";
 return (K)0;
}

KAPI cudaloss(K x) {
 auto* l=xloss(x);
 auto& m=l->m;
 auto p=m.ptr();
 std::cerr << *p << "\n";
 p->to(torch::kCUDA);
 std::cerr << *p << "\n";
 if(l->c == Cast::nll) {
  auto g=m.get<torch::nn::NLLLoss>();
  std::cerr << g->options.weight() << "\n";
  std::cerr << g->weight << "\n";
  std::cerr << g->weight.is_same(g->options.weight()) << "\n";
 }
 return(K)0;
}

KAPI wt(K x,K y,K w) {
 KTRY
  torch::nn::BCELoss m;
  Tensor X,Y,W;
  if(xten(x,X) && xten(y,Y) && xten(w,W)) {
   m->options.weight(W);
   Tensor r=m->forward(X,Y);
   m->options.weight({});
   return kget(r);
  } else {
   return(K)0;
  }
 KCATCH("bce wt");
}

KAPI memtest(K x) {
 std::vector<torch::Tensor> v;
 v.reserve(10000000);
 for(size_t i=0; i<10000000; ++i) v.emplace_back(torch::arange(10));
 return kj(v.size());
}

//#include <c10/cuda/CUDAMacros.h>
//#include <c10/cuda/CUDACachingAllocator.h>

// check for cuda via USE_CUDA
// #ifdef USE_CUDA
//  ..
// #endif
/*
namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {
C10_CUDA_API void emptyCache();
C10_CUDA_API uint64_t currentMemoryAllocated(int device);
C10_CUDA_API uint64_t maxMemoryAllocated(int device);
C10_CUDA_API void     resetMaxMemoryAllocated(int device);
C10_CUDA_API uint64_t currentMemoryCached(int device);
C10_CUDA_API uint64_t maxMemoryCached(int device);
C10_CUDA_API void     resetMaxMemoryCached(int device);
}}}
*/

/*
cache      
memory     e.g. memory() or memory`cuda or memory 0
maxcache   
maxmemory  
emptycache
resetcache 
resetmemory
*/
KAPI cudamem(K x) {
 KTRY
  // if sym, get device no
  // if int, verify -1<n< env.cuda
  //auto n=c10::cuda::CUDACachingAllocator::currentMemoryAllocated(x->j);
  //return kj(n);
  return kj(nj);
 KCATCH("cuda memory");
}

using Reduce1=c10::variant<torch::enumtype::kNone, torch::enumtype::kMean, torch::enumtype::kSum>;
using Reduce2=c10::variant<torch::enumtype::kNone, torch::enumtype::kBatchMean, torch::enumtype::kSum, torch::enumtype::kMean>;

static void reduce(Reduce1& r,Cast c,S s) {
 switch(emap(s)) {
  case Enum::none: r=torch::kNone; break;
  case Enum::mean: r=torch::kMean; break;
  case Enum::sum:  r=torch::kSum; break;
  default: AT_ERROR("not one of none,mean,sum");
 }
}

static void reduce(Reduce2& r,Cast c,S s) {
 switch(emap(s)) {
  case Enum::none:      r=torch::kNone; break;
  case Enum::batchmean: r=torch::kBatchMean; break;
  case Enum::mean:      r=torch::kMean; break;
  case Enum::sum:       r=torch::kSum; break;
  default: AT_ERROR("not one of none,batchmean,mean,sum");
 }
}

KAPI losstest(K x) {
 KTRY
 torch::nn::L1LossOptions o1;
 torch::nn::KLDivLossOptions o2;
 reduce(o1.reduction(),Cast::l1,x->s);
 std::cerr << torch::enumtype::get_enum_name(o1.reduction()) << "\n";
 reduce(o2.reduction(),Cast::kl,x->s);
 std::cerr << torch::enumtype::get_enum_name(o2.reduction()) << "\n";
 return (K)0;
 KCATCH("loss reduction");
}

#define ENUMTEST(name) \
{ \
  v = torch::k##name; \
  std::cerr << torch::enumtype::get_enum_name(v) << " " << ESYM(v) << "\n"; \
}

KAPI enumtest(K x) {
  c10::variant<
    torch::enumtype::kLinear,
    torch::enumtype::kConv1D,
    torch::enumtype::kConv2D,
    torch::enumtype::kConv3D,
    torch::enumtype::kConvTranspose1D,
    torch::enumtype::kConvTranspose2D,
    torch::enumtype::kConvTranspose3D,
    torch::enumtype::kSigmoid,
    torch::enumtype::kTanh,
    torch::enumtype::kReLU,
    torch::enumtype::kLeakyReLU,
    torch::enumtype::kFanIn,
    torch::enumtype::kFanOut,
    torch::enumtype::kConstant,
    torch::enumtype::kReflect,
    torch::enumtype::kReplicate,
    torch::enumtype::kCircular,
    torch::enumtype::kNearest,
    torch::enumtype::kBilinear,
    torch::enumtype::kBicubic,
    torch::enumtype::kTrilinear,
    torch::enumtype::kArea,
    torch::enumtype::kSum,
    torch::enumtype::kMean,
    torch::enumtype::kMax,
    torch::enumtype::kNone,
    torch::enumtype::kBatchMean,
    torch::enumtype::kZeros,
    torch::enumtype::kBorder,
    torch::enumtype::kReflection
  > v;

  ENUMTEST(Linear)
  ENUMTEST(Conv1D)
  ENUMTEST(Conv2D)
  ENUMTEST(Conv3D)
  ENUMTEST(ConvTranspose1D)
  ENUMTEST(ConvTranspose2D)
  ENUMTEST(ConvTranspose3D)
  ENUMTEST(Sigmoid)
  ENUMTEST(Tanh)
  ENUMTEST(ReLU)
  ENUMTEST(LeakyReLU)
  ENUMTEST(FanIn)
  ENUMTEST(FanOut)
  ENUMTEST(Constant)
  ENUMTEST(Reflect)
  ENUMTEST(Replicate)
  ENUMTEST(Circular)
  ENUMTEST(Nearest)
  ENUMTEST(Bilinear)
  ENUMTEST(Bicubic)
  ENUMTEST(Trilinear)
  ENUMTEST(Area)
  ENUMTEST(Sum)
  ENUMTEST(Mean)
  ENUMTEST(Max)
  ENUMTEST(None)
  ENUMTEST(BatchMean)
  ENUMTEST(Zeros)
  ENUMTEST(Border)
  ENUMTEST(Reflection)
 return (K)0;
}

KAPI kdata(K x,K y) {
 KTRY
  int64_t i=0;
  auto dataset = torch::data::datasets::MNIST(x->s)
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader(std::move(dataset));
  for (torch::data::Example<>& batch : *data_loader) {
    if(i==y->j) {
     std::cout << batch.target << "\n";
     std::cout << batch.data   << "\n ";
     return kten(batch.data);
    }
  }
  return (K)0;
 KCATCH("mnist test");
}

void f(int64_t n) {
 n*=1000000;
 auto t=torch::rand(n);
 double d=0; float f=0,*p=t.data_ptr<float>();
 for(int64_t i=0; i<n; ++i) d+=p[i], f+=p[i];
 std::cerr << "double  sum: " <<   d << "\n";
 std::cerr << "float   sum: " <<   f << "\n\n";
 std::cerr << "double mean: " << d/n << "\n";
 std::cerr << "float  mean: " << f/n << "\n";
 std::cerr << "torch  mean: " << t.mean().item().toFloat() << "\n";
}

KAPI ftest(K x) {
 f(x->j);
 return (K)0;
}

KAPI hashtest(K x) {
 std::unordered_set<J> Ptrs;
 Ptrs.insert(10);
 Ptrs.insert(20);
 Ptrs.insert(2);
 Ptrs.insert(20);

 std::cerr << "size: " << Ptrs.size() << "\n";
 for(auto j:Ptrs)
  std::cerr << j << "\n";

 //std::cerr << " find: " << Ptrs.find(20) << "\n";
 std::cerr << "count: " << Ptrs.count(20) << "\n";
 Ptrs.erase(20);
 std::cerr << "size: " << Ptrs.size() << "\n";
 //std::cerr << " find: " << Ptrs.find(20) << "\n";
 std::cerr << "count: " << Ptrs.count(20) << "\n";
 return (K)0;
}

void errfail() {
 if(true) {
  AT_ERROR("err");
 } else {
  AT_ERROR("false");
 }
}

KAPI testcount(K x,K y) {
 KTRY
 if(y->t != -KJ) return KERR("2nd arg must be offset");
 Pairs p; J i=y->j; J n=xargc(x,i,p);
 std::cerr << "arg count: " << n << ", pair count: " << p.n << "\n";
 return kb(xnone(x,i));
 KCATCH("test count");
}

KAPI testptr(K x) {
 Tensor t;
 if(xten(x,t))
  std::cerr<<"tensor\n";
 else if(xten(x,0,t) && x->n==1)
  std::cerr<<"enlisted tensor\n";
 else
  std::cerr<<"something else\n";
 return(K)0;
}

#define ASSERT_THROWS_WITH(statement, substring)                        \
  {                                                                     \
    std::string assert_throws_with_error_message;                       \
    try {                                                               \
      (void)statement;                                                  \
      std::cerr << "Expected statement `" #statement                       \
                "` to throw an exception, but it did not";              \
    } catch (const c10::Error& e) {                                     \
      assert_throws_with_error_message = e.what_without_backtrace();    \
    } catch (const std::exception& e) {                                 \
      assert_throws_with_error_message = e.what();                      \
    }                                                                   \
    if (assert_throws_with_error_message.find(substring) ==             \
        std::string::npos) {                                            \
      std::cerr << "Error message \"" << assert_throws_with_error_message  \
             << "\" did not contain expected substring \"" << substring \
             << "\"";                                                   \
    }                                                                   \
  }

KAPI namecheck(K x) {
 torch::nn::Sequential s; //size_t n=0;
 std::cout << "initial size: " << s->size() << "\n";
 ASSERT_THROWS_WITH(
      s->push_back("name.with.dot", torch::nn::Linear(3, 4)),
      "Submodule name must not contain a dot (got 'name.with.dot')");
  ASSERT_THROWS_WITH(
      s->push_back("", torch::nn::Linear(3, 4)),
      "Submodule name must not be empty");
  std::cout << "size after name errors: " << s->size() << "\n";
  //for(auto&c:s->named_children()) n++;
  std::cout << "size of modules: "        << s->modules(false).size() << "\n";
  std::cout << "size of named children: " << s->named_children().size() << "\n";

  return(K)0;
}

KAPI dupname(K x) {
KTRY
 Sequential q(
  {{"A", torch::nn::Linear(1,2)},
   {"B", torch::nn::Conv2d(3,4,5)}});
 return (K)0;
KCATCH("duplicate names");
}

KAPI kdict(K x) {
 return kb(xdict(x));
}

static K ksub(K x,const char* e) {
 KTRY
 std::cerr << "in ksub " << (!x ? "null" : "with args")<< "\n";
 Tensor t;
 if(!x || (x->t==-KS && x->s==cs("help"))) {
  std::cerr << " still in ksub " << (!x ? "null" : "with args")<< "\n";
  AT_ERROR(e," help here..");
 }

 if(xten(x,t)) {
  return kten(torch::max(t));
 } else {
  return ksub(nullptr,e);
 }
 KCATCH(e);
}

KAPI ktest(K x) {return ksub(x,"ktest()");}

KAPI mixtest(K x) {return kb(xmixed(x,4));}

K help(bool b,const char* s) {return b ? KERR(s) : (fprintf(stderr,"%s\n",s), (K)0);}

#define KHELP(cond, ...)  \
  if((cond))              \
   AT_WARN(__VA_ARGS__);  \
  else                    \
   AT_ERROR(__VA_ARGS__); \

/*
KAPI helptest(K x) {
KTRY
 const char* a="some FN";
// if(!x || xhelp(x)) {
 if(!x || xempty(x)) {
  KHELP(x,"This is one part,",a,"\n"
          " another part.."
          " more parts.\n"
          " last part\n")
  return(K)0;
  } else {
   return helptest(nullptr);
  }
KCATCH("help test");
}
*/

typedef struct {
 std::array<std::tuple<S,Cast,std::function<Tensor(Tensor)>>,2> fn = {{
 }};
} Testenv;

Testenv& testenv() {static Testenv e; return e;}

KAPI pairtest(K x) {
 KTRY
 Pairs p;
 if(xpairs(x,p) || xpairs(x,x->n-1,p)) {
  switch(p.a) {
   case 1: std::cout << "Dictionary["; break;
   case 2: std::cout << "Pairs["; break;
   case 3: std::cout << "List["; break;
   case 4: std::cout << "Symbol list["; break;
   default: std::cout << "Unknown name,value structure["; break;
  }
  std::cout << p.n << "]\n";
  while(xpair(p)) {
   switch(p.t) {
    case -KB: std::cout << "Boolean: " << p.k << " -> " << p.b << "\n"; break;
    case -KS: std::cout << " Symbol: " << p.k << " -> " << p.s << "\n"; break;
    case -KJ: std::cout << "Integer: " << p.k << " -> " << p.j << "\n"; break;
    case -KF: std::cout << " Double: " << p.k << " -> " << p.f << "\n"; break;
    default:  std::cout << "  Other: " << p.k << " -> " << kname(p.t) << "\n"; break;
   }
  }
  return kb(true);
 } else {
  return kb(false);
 }
 KCATCH("pairs test..");
}

KAPI lbfgs(K x) {
    int i, n=x->j;
    auto t=torch::randn({n});

    TensorVector v = {torch::randn({n}, torch::requires_grad())};
    //torch::optim::SGD o(v, /*lr=*/0.01);
    torch::optim::LBFGS o(v, 1);

    auto cost = [&](){
        o.zero_grad();
        auto d = torch::pow(v[0] - t, 2).sum();
        std::cerr << i << ") " << d.item().toDouble() << "\n";
        d.backward();
        return d;
    };

    for (i = 0; i < 5; ++i){
        o.step(cost);//for LBFGS
        //cost(); o.step(); // for SGD
    }
    return kget(torch::stack({t,v[0]}));
}

KAPI learn(K x) {
 KTRY
  Scalar s; Tensor t;
  if(xten(x,0,t) && xnum(x,1,s)) {
   if(t.grad().defined()) {
    torch::NoGradGuard g;
    //t.add_(-s.toDouble()*t.grad());
    t.add_(-s*t.grad());
    t.grad().zero_();
    return (K)0;
   } else {
    return KERR("No gradient defined");
   }
  } else {
   return KERR("Unrecognized arg(s), expecting (tensor;learning rate)");
  }
 KCATCH("Error applying learning rate and gradient to tensor");
}

bool xindex(K x,Tensor &t) {J n,*j; return xten(x,t) ? true : (xlong(x,n,j) ? t=kput(x),true : false);}
bool xindex(K x,J i,Tensor &t) { return xind(x,i) && xindex(kK(x)[i],t);}

KAPI kindex(K x) {
 J d; Tensor r,t,i;
 KTRY
  if(xten(x,0,t) && xlong(x,1,d) && xindex(x,2,i)) {
    if(x->n==3)
     return kten(torch::index_select(t,d,i));
    else if(xten(x,3,r) && x->n==4)
     return torch::index_select_out(r,t,d,i), (K)0;
  }
  return KERR("Unrecognized arg(s), expected (tensor;dim;indices;optional output tensor)");
 KCATCH("index");
}

KAPI opttest(K x) {
 TensorOptions o;
 //o.is_variable(true);
 std::cout << "dtype:       " << o.dtype() << "\n";
 std::cout << "device:      " << o.device() << "\n";
 std::cout << "layout:      " << o.layout() << "\n";
 std::cout << "gradient:    " << o.requires_grad() << "\n";
 //std::cout << "variable:    " << o.is_variable() << "\n";
 std::cout << "has dtype:   " << o.has_dtype()  << "\n";
 std::cout << "has device:  " << o.has_device() << "\n";
 std::cout << "has layout:  " << o.has_layout() << "\n";
 //std::cout << "has variable:" << o.has_is_variable() << "\n";
 std::cout << "has gradient:" << o.has_requires_grad() << "\n";
 return (K)0;
}

// tensor(`sparse;array)
// tensor(`sparse;array;mask)
// tensor(`sparse;size) -> tensor(`empty;3 4;`sparse)

/*
KAPI sparse(K x) {
 Tensor a,b; TensorOptions o;
 KTRY
  if(xten(x,a)) {
  } else if(xten(x,0,a) && xten(x,1,b) {
   if(x->n==2)
   else if(x->n==3)
   
    // no size
    // size
    // w'options
  } else if(x->n = {
  }
 
  return(K)0;
 KCATCH("Sparse tensor error");
}
*/

KAPI to_sparse(K x) {
 if(auto* t=xten(x))
  return kten(t->to_sparse());
 else
  AT_ERROR("to_sparse not implemented for ",kname(x->t));
}


KAPI sparse1(K x) {
 auto m=kput(kK(x)[0]),t=kput(kK(x)[1]),v=torch::masked_select(t,m),i=torch::nonzero(m);
 //return kten(torch::sparse_coo_tensor(i.t(),v));
 return kten(torch::sparse_coo_tensor(i.t(),v,m.sizes()));
}


KAPI gan(K x) {
 KTRY
  const int64_t kNoiseSize = 100;
  const int64_t kBatchSize = 60;
  const int64_t kNumberOfEpochs = 30;
  const char*   kDataFolder = "/home/t/data/mnist";
  const int64_t kCheckpointEvery = 1000;
  const int64_t kNumberOfSamplesPerCheckpoint = 10;
  const int64_t kLogInterval = 10;

  using namespace torch;
  torch::manual_seed(1);
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

  struct DCGANGeneratorImpl : nn::Module {
    DCGANGeneratorImpl(int kNoiseSize)
      : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 256, 4).bias(false)),
        conv2(nn::ConvTranspose2dOptions(256, 128, 3).stride(2).padding(1).bias(false)),
        conv3(nn::ConvTranspose2dOptions(128,  64, 4).stride(2).padding(1).bias(false)),
        conv4(nn::ConvTranspose2dOptions( 64,   1, 4).stride(2).padding(1).bias(false)),
        batch_norm1(256),
        batch_norm2(128),
        batch_norm3(64) {
     register_module("conv1", conv1);
     register_module("conv2", conv2);
     register_module("conv3", conv3);
     register_module("conv4", conv4);
     register_module("batch_norm1", batch_norm1);
     register_module("batch_norm2", batch_norm2);
     register_module("batch_norm3", batch_norm3);
   }

   torch::Tensor forward(torch::Tensor x) {
      x = torch::relu(batch_norm1(conv1(x)));
      x = torch::relu(batch_norm2(conv2(x)));
      x = torch::relu(batch_norm3(conv3(x)));
      x = torch::tanh(conv4(x));
      return x;
    }

    nn::ConvTranspose2d conv1, conv2, conv3, conv4;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
  };

  TORCH_MODULE(DCGANGenerator);

  DCGANGenerator generator(kNoiseSize);
  generator->to(device);

  nn::Sequential discriminator(
      nn::Conv2d(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      nn::Conv2d(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
      nn::BatchNorm2d(128),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      nn::Conv2d(nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
      nn::BatchNorm2d(256),
      nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
      nn::Conv2d(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
      nn::Sigmoid());
  discriminator->to(device);

  // Assume the MNIST dataset is available under `kDataFolder`;
  auto dataset = torch::data::datasets::MNIST(kDataFolder)
                     .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                     .map(torch::data::transforms::Stack<>());
  const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

  auto data_loader = torch::data::make_data_loader(std::move(dataset),torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

  torch::optim::Adam generator_optimizer(generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
  torch::optim::Adam discriminator_optimizer(discriminator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));

  int64_t checkpoint_counter = 1;
  for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      // Train discriminator with real images.
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();
      std::cerr << " label size: " << real_labels.sizes() << "\n";
      std::cerr << " image size: " << real_images.sizes() << "\n";
      std::cerr << "output size: " << real_output.sizes() << "\n";
      return (K)0;

      // Train discriminator with fake images.
      torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();
      batch_index++;
      if (batch_index % kLogInterval == 0) {
        std::printf("\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f\n", epoch, kNumberOfEpochs, batch_index, batches_per_epoch, d_loss.item<float>(), g_loss.item<float>());
      }

      if (batch_index % kCheckpointEvery == 0) {
        // Checkpoint the model and optimizer state.
        torch::save(generator, "generator-checkpoint.pt");
        torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
        torch::save(discriminator, "discriminator-checkpoint.pt");
        torch::save(
            discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
        // Sample the generator and save the images.
        torch::Tensor samples = generator->forward(torch::randn(
            {kNumberOfSamplesPerCheckpoint, kNoiseSize, 1, 1}, device));
        torch::save(
            (samples + 1.0) / 2.0,
            torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
        std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
      }
    }
  }
  return(K)0;
 KCATCH("gan");
}

/*
KAPI gan(K x) {
 const int64_t kNoiseSize = 100;
 const int64_t kBatchSize = 60;
 const int64_t kNumberOfEpochs = 30;
 const char*   kDataFolder = "/home/t/data/mnist";
 const int64_t kLogInterval = 1000;

 torch::manual_seed(1);
 using namespace torch;
 torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

  nn::Sequential generator(
   nn::Conv2d(nn::Conv2dOptions(kNoiseSize, 256, 4).bias(false).transposed(true)),
      nn::BatchNorm(256),
      nn::Functional(torch::relu),
   nn::Conv2d(nn::Conv2dOptions(256, 128, 3).stride(2).padding(1).bias(false).transposed(true)),
      nn::BatchNorm(128),
      nn::Functional(torch::relu),
   nn::Conv2d(nn::Conv2dOptions(128, 64, 4).stride(2).padding(1).bias(false).transposed(true)),
      nn::BatchNorm(64),
      nn::Functional(torch::relu),
   nn::Conv2d(nn::Conv2dOptions(64, 1, 4).stride(2).padding(1).bias(false).transposed(true)),
   nn::Functional(torch::tanh));
  generator->to(device);

  nn::Sequential discriminator(
      nn::Conv2d(nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
      nn::Functional(torch::leaky_relu, 0.2),
      nn::Conv2d(nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
      nn::BatchNorm(128),
      nn::Functional(torch::leaky_relu, 0.2),
      nn::Conv2d(nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
      nn::BatchNorm(256),
      nn::Functional(torch::leaky_relu, 0.2),
      nn::Conv2d(nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(false)),
      nn::Functional(torch::sigmoid));
  discriminator->to(device);
  //torch::Tensor z = torch::randn({kBatchSize, kNoiseSize, 1, 1}, device);
  //return kten(generator->forward(z));

  auto dataset = torch::data::datasets::MNIST(kDataFolder).map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());
  const int64_t batches_per_epoch = std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));
  auto data_loader = torch::data::make_data_loader(std::move(dataset),torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));
//auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset),torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));
  torch::optim::Adam generator_optimizer    (    generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
  torch::optim::Adam discriminator_optimizer(discriminator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
  auto losses=torch::zeros(kNumberOfEpochs*batches_per_epoch*2);
  auto lossptr=losses.data_ptr<float>();

  for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    int64_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      //return kget(batch.data);
      // Train discriminator with real images.
      std::cerr << "discriminator\n";
      discriminator->zero_grad();
      torch::Tensor real_images = batch.data.to(device);
      torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
      torch::Tensor real_output = discriminator->forward(real_images);
      torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
      d_loss_real.backward();

      // Train discriminator with fake images.
      torch::Tensor noise = torch::randn({batch.data.size(0), kNoiseSize, 1, 1}, device);
      std::cerr << "generator\n";
      torch::Tensor fake_images = generator->forward(noise);
      torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
      torch::Tensor fake_output = discriminator->forward(fake_images.detach());
      torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
      d_loss_fake.backward();

      torch::Tensor d_loss = d_loss_real + d_loss_fake;
      discriminator_optimizer.step();

      // Train generator.
      generator->zero_grad();
      fake_labels.fill_(1);
      fake_output = discriminator->forward(fake_images);
      torch::Tensor g_loss =
          torch::binary_cross_entropy(fake_output, fake_labels);
      g_loss.backward();
      generator_optimizer.step();
      batch_index++;
      *lossptr++ = d_loss.item<float>();
      *lossptr++ = g_loss.item<float>();
      if (batch_index % kLogInterval == 0) {
        std::printf("\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f\n",
            epoch, kNumberOfEpochs, batch_index, batches_per_epoch, d_loss.item<float>(), g_loss.item<float>());
      }
    }
  }
  torch::Tensor samples = generator->forward(torch::randn({10, kNoiseSize, 1, 1}, device));
  torch::save((samples + 1.0) / 2.0, torch::str("sample.pt"));
  samples = generator->forward(torch::randn({100, kNoiseSize, 1, 1}, device));
  return kten(samples);
 //return kget(losses.reshape({kNumberOfEpochs,batches_per_epoch,2}));
}

KAPI gentest(K x) {
 torch::nn::Sequential generator(
    torch::nn::ConvTranspose2d(torch::nn::Conv2dOptions(100, 256, 4).bias(false).transposed(true)),
    torch::nn::BatchNorm(256),
    torch::nn::Functional(torch::relu),
    torch::nn::ConvTranspose2d(torch::nn::Conv2dOptions(256, 128, 3).stride(2).padding(1).bias(false).transposed(true)),
    torch::nn::BatchNorm(128),
    torch::nn::Functional(torch::relu),
    torch::nn::ConvTranspose2d(torch::nn::Conv2dOptions(128, 64, 4).stride(2).padding(1).bias(false).transposed(true)),
    torch::nn::BatchNorm(64),
    torch::nn::Functional(torch::relu),
    torch::nn::ConvTranspose2d(torch::nn::Conv2dOptions(64, 1, 4).stride(2).padding(1).bias(false).transposed(true)),
    torch::nn::Functional(torch::tanh));
  //generator(torch::randn({64,100,1,1}),256);
  return (K)0;
}
*/
