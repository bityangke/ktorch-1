#include "ktorch.h"
#include "knn.h"
#include "kloss.h"
#include <c10/cuda/CUDAMacros.h>
//#include <c10/cuda/CUDACachingAllocator.h>

// check for cuda via USE_CUDA
// #ifdef USE_CUDA
//  ..
// #endif
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
 Sequential q(torch::nn::modules_ordered_dict(
  {{"A", torch::nn::Linear(1,2)},
   {"B", torch::nn::Conv2d(3,4,5)}}));
 return (K)0;
KCATCH("duplicate names");
}

KAPI kdict(K x) {
 return kb(xdict(x));
}

static K ksub(K x,cS e) {
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

K help(B b,cS s) {return b ? KERR(s) : (fprintf(stderr,"%s\n",s), (K)0);}

#define KHELP(cond, ...)  \
  if((cond))              \
   AT_WARN(__VA_ARGS__);  \
  else                    \
   AT_ERROR(__VA_ARGS__); \

/*
KAPI helptest(K x) {
KTRY
 cS a="some FN";
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

/*
K model(Sequential& q,Loss *l,Optimizer *o,LossClosureOptimizer *oc) {
    
 auto cost=[&]() {
  o->zero_grad();
  auto z=l->forward(q->forward(v[0]
  z.backward();
  return d;
 };
}
*/

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

B xindex(K x,Tensor &t) {J n,*j; return xten(x,t) ? true : (xlong(x,n,j) ? t=kput(x),true : false);}
B xindex(K x,J i,Tensor &t) { return xind(x,i) && xindex(kK(x)[i],t);}

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
 std::cout << "variable:    " << o.is_variable() << "\n";
 std::cout << "has dtype:   " << o.has_dtype()  << "\n";
 std::cout << "has device:  " << o.has_device() << "\n";
 std::cout << "has layout:  " << o.has_layout() << "\n";
 std::cout << "has variable:" << o.has_is_variable() << "\n";
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
