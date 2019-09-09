#include "ktorch.h"
#include "kmodule.h"

ZV modelpart(K x,J i,Kseq*& q,Kloss*& l,Kopt*& o) {
 for(;i<x->n;++i) {
  auto* g=xtag(x,i);
  switch(g ? g->a : Class::undefined) {
   case Class::sequential: q=(Kseq*)g;  break;
   case Class::loss:       l=(Kloss*)g; break;
   case Class::optimizer:  o=(Kopt*)g;  break;
   default: AT_ERROR("model arg[",i,"] unrecognized: ",
                    (g ? mapclass(g->a) : kname(kK(x)[i]->t))); break;
  }
 }
}

K modelstate(B a,B b,Kmodel *m) {
  std::cerr << "model state..\n";
 return (K)0;
}

KAPI model(K x) {
 KTRY
  B a=env().alloptions;
  Kseq *q=nullptr; Kloss *l=nullptr; Kopt *o=nullptr; Kmodel *m=nullptr;
  TORCH_CHECK(!x->t, "model not implemented for ",kname(x->t));
  if((m=xmodel(x)) || (x->n==2 && xbool(x,1,a) && (m=xmodel(x,0)))) {
   return modelstate(a,false,m);
  } else {
   m=xmodel(x,0); modelpart(x,m ? 1 : 0,q,l,o);
   if(m) {
    if(q) m->q=q->q;              // reassign model's sequential module
    if(l) m->lc=l->c, m->l=l->l;  // loss function
    if(o) m->oc=o->c, m->o=o->o;  // optimizer
    return (K)0;
   } else {
    TORCH_CHECK(q && l && o, (q ? (l ? "optimizer" : "loss") : "sequential module")," not specified");
    return kptr(new Kmodel(q,l,o));
   }
  }
 KCATCH("model");
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

ZK ksub(K x,cS e) {
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

Testenv& testenv(V) {static Testenv e; return e;}

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

KAPI modptr(K x) {
 auto p=torch::make_unique<torch::nn::LinearImpl>(10,7);
 std::cout << "ref count: "<< "ref count: "<< p->weight.use_count() << "\n";
 std::cout << p->weight << "\n";
 return kptr(p.release());
}

V shuffle(std::vector<Tensor>& v) {
 size_t i=0; Tensor p;
 for(auto& t:v) {
  if(!p.defined())
   p=torch::randperm(t.size(0),torch::dtype(torch::kLong).device(t.device()));
  else if(t.size(0) != p.size(0))
   AT_ERROR("Size mismatch: tensor[", i, "] length ",t.size(0), ", but permutation is ", p.size(0));
  else if (t.device() != p.device())
    AT_ERROR("Device mismatch: tensor[", i, "] is on ", t.device(), ", but permutation indices are on ", p.device());
  t=t.index_select(0,p); ++i;
 }
}

KAPI kshuffle(K x) {
 KTRY
  if(auto* v=xvec(x))
   shuffle(*v);
  else
   AT_ERROR("shuffle expects vector of tensors, input is ",kname(x->t));
  return (K)0;
 KCATCH("shuffle");
}

/*
KAPI shuffle(K x) {
 KTRY
  if(x->t<0) {
   AT_ERROR("shuffle not implemented for ",kname(x->t));
  } else if(x->t) {
   auto t=kput(x);
   auto n=t.size(0);
   auto p=

 KCATCH("shuffle");
}
*/

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

    std::vector<torch::Tensor> v = {torch::randn({n}, torch::requires_grad())};
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

V subset(Tensor& t,int64_t i,int64_t n) {
 auto s=(int64_t*)t.sizes().data(); s[0]=n;
 t.set_(t.storage(),i*t.stride(0),t.sizes(),t.strides());
}

/*
i=t.storage_offset()+t.numel;
if(i < t.storage().size()-t.numel())
 return true;
else if
*/

V resize(Tensor& t,int64_t d) {
 int64_t i,n=1;
 for(i=1; i<t.dim(); ++i) n*=t.size(i);
 subset(t,0,t.storage().size()/n);
}

V subset(std::vector<Tensor>& v,int64_t i,int64_t n) {
 size_t j=0;
 for(auto& t:v) {
  /*
  TORCH_CHECK(t.dim(), "dim");
  TORCH_CHECK(i<t.storage().size()/t.stride(d), "offset");
  */
  subset(t,i,n); j++;
 }
}

KAPI narrow(K x) {
 KTRY
  J d,i,n; Tensor t;
  if(xlong(x,1,d) && xlong(x,2,i) && xlong(x,3,n) && x->n==4)
   return xten(x,0,t) ? kten(t.narrow(d,i,n)) : kget(kput(x,0).narrow(d,i,n));
  else
   return KERR("Unrecognized arg(s) for narrow, expecting (input;dim;offset;size)");
 KCATCH("narrow");
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

KAPI seedtest(K x) {
 torch::manual_seed(x->j);
 return(K)0;
}

KAPI gradask(K x) {
 auto t=torch::tensor({1.0});
 std::cout << t << "\n";
 std::cout << t.options().requires_grad() << "\n";
 Tensor v=torch::tensor({1.0}, torch::requires_grad());
 v.requires_grad();
 v.options().requires_grad();
 std::cout << v.requires_grad() << "\n";
 std::cout << v.options().requires_grad() << "\n";
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

KAPI sparse1(K x) {
 auto m=kput(kK(x)[0]),t=kput(kK(x)[1]),v=torch::masked_select(t,m),i=torch::nonzero(m);
 //return kten(torch::sparse_coo_tensor(i.t(),v));
 return kten(torch::sparse_coo_tensor(i.t(),v,m.sizes()));
}
