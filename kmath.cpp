#include "ktorch.h"

// -----------------------------------------------------------------------------------------
// define function pointers, e.g. Ftt for function(tensor,tensor), G w'output
// -----------------------------------------------------------------------------------------
using Ft      = Tensor  (*)(const Tensor&);
using Gt      = Tensor& (*)(Tensor&, const Tensor&);
using Ftb     = Tensor  (*)(const Tensor&, bool);
using Gtb     = Tensor& (*)(Tensor&, const Tensor&, bool);
using Fti     = Tensor  (*)(const Tensor&, int64_t);
using Gti     = Tensor& (*)(Tensor&, const Tensor&, int64_t);
using Fts     = Tensor  (*)(const Tensor&, Scalar);
using Gts     = Tensor& (*)(Tensor&, const Tensor&, Scalar);
using Ftt     = Tensor  (*)(const Tensor&, const Tensor&);
using Gtt     = Tensor& (*)(Tensor&, const Tensor&, const Tensor&);
using Fttts   = Tensor  (*)(const Tensor&, const Tensor&, const Tensor&, Scalar);
using Gttts   = Tensor& (*)(Tensor&, const Tensor&, const Tensor&, const Tensor&, Scalar);
using Ftuple1 = std::tuple<Tensor,Tensor>   (*)(const Tensor&);
using Ftuple2 = std::tuple<Tensor,Tensor>   (*)(const Tensor&, const Tensor&);
using Gtuple1 = std::tuple<Tensor&,Tensor&> (*)(Tensor&, Tensor&, const Tensor&);
using Gtuple2 = std::tuple<Tensor&,Tensor&> (*)(Tensor&, Tensor&, const Tensor&, const Tensor&);
using Fmm     = Tensor  (*)(const Tensor&, const Tensor&, const Tensor&, Scalar, Scalar);
using Gmm     = Tensor& (*)(Tensor&, const Tensor&, const Tensor&, const Tensor&, Scalar, Scalar);

// -----------------------------------------------------------------------------------------
// point-wise & other math fns with arg(s) of tensor, (tensor;output tensor) or k array
// -----------------------------------------------------------------------------------------
ZK math1(K x, Ft f, Gt g, cS e) {
 KTRY
  Tensor t,r;
  if(xten(x,t))                                     // tensor
   return kten(f(t));
  else if(xten(x,1,r) && x->n==2)                   // (array/tensor;output tensor)
   return g(r,xten(x,0,t) ? t : kput(x,0)), (K)0;
  else                                              // k array
   return kget(f(kput(x)));
 KCATCH(e);
}

KAPI Abs(K x)        {return math1(x, torch::abs,        torch::abs_out,        "absolute value");}
KAPI Acos(K x)       {return math1(x, torch::acos,       torch::acos_out,       "arccosine");}
KAPI Asin(K x)       {return math1(x, torch::asin,       torch::asin_out,       "arcsine");}
KAPI Atan(K x)       {return math1(x, torch::atan,       torch::atan_out,       "arctangent");}
KAPI Ceil(K x)       {return math1(x, torch::ceil,       torch::ceil_out,       "ceiling");}
KAPI Cos(K x)        {return math1(x, torch::cos,        torch::cos_out,        "cosine");}
KAPI Cosh(K x)       {return math1(x, torch::cosh,       torch::cosh_out,       "hyperbolic cosine");}
KAPI Digamma(K x)    {return math1(x, torch::digamma,    torch::digamma_out,    "log derivative of gamma");}
KAPI Erf(K x)        {return math1(x, torch::erf,        torch::erf_out,        "error function");}
KAPI Erfc(K x)       {return math1(x, torch::erfc,       torch::erfc_out,       "complimentary error function");}
KAPI Erfinv(K x)     {return math1(x, torch::erfinv,     torch::erfinv_out,     "inverse error function");}
KAPI Exp(K x)        {return math1(x, torch::exp,        torch::exp_out,        "exponential");}
KAPI Expm1(K x)      {return math1(x, torch::expm1,      torch::expm1_out,      "exponential minus 1");}
KAPI Floor(K x)      {return math1(x, torch::floor,      torch::floor_out,      "floor");}
KAPI Frac(K x)       {return math1(x, torch::frac,       torch::frac_out,       "fractional");}
KAPI Inverse(K x)    {return math1(x, torch::inverse,    torch::inverse_out,    "matrix inverse");}
KAPI Log(K x)        {return math1(x, torch::log,        torch::log_out,        "log");}
KAPI Log10(K x)      {return math1(x, torch::log10,      torch::log10_out,      "log10");}
KAPI Log1p(K x)      {return math1(x, torch::log1p,      torch::log1p_out,      "log1p");}
KAPI Log2(K x)       {return math1(x, torch::log2,       torch::log2_out,       "log2");}
KAPI Neg(K x)        {return math1(x, torch::neg,        torch::neg_out,        "negative");}
KAPI Reciprocal(K x) {return math1(x, torch::reciprocal, torch::reciprocal_out, "reciprocal");}
KAPI Round(K x)      {return math1(x, torch::round,      torch::round_out,      "round");}
KAPI Rsqrt(K x)      {return math1(x, torch::rsqrt,      torch::rsqrt_out,      "reciprocal square root");}
KAPI Ksigmoid(K x)   {return math1(x, torch::sigmoid,    torch::sigmoid_out,    "sigmoid");}
KAPI Sign(K x)       {return math1(x, torch::sign,       torch::sign_out,       "sign");}
KAPI Sin(K x)        {return math1(x, torch::sin,        torch::sin_out,        "sine");}
KAPI Sinh(K x)       {return math1(x, torch::sinh,       torch::sinh_out,       "hyperbolic sine");}
KAPI Sqrt(K x)       {return math1(x, torch::sqrt,       torch::sqrt_out,       "square root");}
KAPI Tan(K x)        {return math1(x, torch::tan,        torch::tan_out,        "tangent");}
KAPI Ktanh(K x)      {return math1(x, torch::tanh,       torch::tanh_out,       "hyperbolic tangent");}
KAPI Trunc(K x)      {return math1(x, torch::trunc,      torch::trunc_out,      "truncate");}

// ---------------------------------------------------------------------------------------------
// point-wise functions with arg of (input1;input2;optional output tensor), input2 may be scalar
// ---------------------------------------------------------------------------------------------
Z Tensor atan2err(const Tensor& a,Scalar s) {
 AT_ERROR("atan2 with 2nd scalar argument not implemented in pytorch");
}

ZK math2(K x,Ftt f,Fts fn,Gtt g,cS e) {
 KTRY
  B p; Scalar n; Tensor a,b,r;
  if(2 == (xten(x,2,r) ? x->n-1 : xlen(x))) {
   if(xnum(x,1,n)) {
    if(!(p=xten(x,0,a))) a=kput(x,0);
   } else {
    p=xtenarg(x,a,b);
   }
   if(r.defined())
    return g(r,a,b.defined() ? b : torch::scalar_to_tensor(n)), (K)0;
   else
    return r=b.defined() ? f(a,b) : fn(a,n), (p ? kten(r) : kget(r));
  } else {
   AT_ERROR(e,": expects args of(input1;input2;optional output tensor), input1 is array or tensor, input2 may also be a number");
   return KERR(e);
  }
 KCATCH(e);
}

KAPI Atan2(K x)     {return math2(x, torch::atan2,     atan2err,         torch::atan2_out,     "arctangent 2");}
KAPI Div(K x)       {return math2(x, torch::div,       torch::div,       torch::div_out,       "divide");}
KAPI Fmod(K x)      {return math2(x, torch::fmod,      torch::fmod,      torch::fmod_out,      "floating point remainder(fmod)");}
KAPI Mul(K x)       {return math2(x, torch::mul,       torch::mul,       torch::mul_out,       "multiply");}
KAPI Remainder(K x) {return math2(x, torch::remainder, torch::remainder, torch::remainder_out, "remainder");}

// --------------------------------------------------------------------------------------------
// add - handle ambiguity of syntax w'3 args (a;s;b) vs (a;s;output) using (a;s;();output)
// addcmul/addcdiv - add tensor to product or quotient of two other tensors: t+v*a*b or t+v*a/b
// --------------------------------------------------------------------------------------------
KAPI Add(K x) {
 KTRY
  Scalar m=1; Tensor a,b,r; B s=xnum(x,1,m); I p=2;
  if(x->t) {
   AT_ERROR("add not implemented for ",kname(x->t));
  } else if(x->n<2 || x->n>4) {
   AT_ERROR("add expects 2-4 args, received ",x->n);
  } else if(x->n==2) {
   if(s && !(p=xten(x,0,a))) 
    a=kput(x,0);
   else
    p=xtenarg(x,a,b);
  } else if(x->n==3 && s) {
   if(!xten(x,0,a)) p--,a=kput(x,0);
   if(!xten(x,2,b)) p--,b=kput(x,2);
  } else if(x->n==3 && xten(x,2,r)) {
   p=xtenarg(x,a,b);
  } else if(x->n==4 && s && xten(x,3,r)) {
   if(!xten(x,0,a)) p--,a=kput(x,0);
   if(xempty(x,2))       p--;
   else if(!xten(x,2,b)) p--, b=kput(x,2);
  } else {
   AT_ERROR("add expects arrays/tensors a,b and scalar s in form: (a;b), (a;s) or (a;s;b)\n",
            "with optional output tensor r use: (a;b;r), (a;s;();r) or (a;s;b;r)");
   return KERR("add");
  }
  if(r.defined())
   return torch::add_out(r,a,b.defined() ? b : torch::ones({},a.dtype()),m), (K)0;
  else
   return r=(s && x->n==2) ? torch::add(a,m) : torch::add(a,b,m), (p ? kten(r) :  kget(r));
 KCATCH("add");
}

ZK addc(K x,Fttts f,Gttts g,cS e) {
 KTRY
  I p=3; Scalar m=1; Tensor a,b,c,r;
  B s=xnum(x,1,m); //s:true if scalar multiplier supplied as 2nd arg
  if(x->t) {
   AT_ERROR(e, " not implemented for ",kname(x->t));
  } else if(x->n<3 || x->n>5) {
   AT_ERROR("expected 3-5 args for ",e,", received ",x->n);
  }
  if ((x->n==3 || (!s && x->n==4 && xten(x,3,r))) || (s && (x->n==4 || (x->n==5 && xten(x,4,r))))) {
   if(!xten(x,0,a)) p--,a=kput(x,0);
   if(!xten(x,1+s,b)) p--,b=kput(x,1+s);
   if(!xten(x,2+s,c)) p--,c=kput(x,2+s);
   if(r.defined())
    return g(r,a,b,c,m), (K)0;
   else
    return r=f(a,b,c,m), (p ? kten(r) : kget(r));
  } else {
    AT_ERROR(e," expects tensor/array a,b,c & multiplier m, (a;b;c), (a;m;b;c), (a;b;c;output), or (a;m;b;c;output)");
    return KERR(e);
  }
 KCATCH(e);
}

KAPI Addcmul(K x) {return addc(x, torch::addcmul, torch::addcmul_out, "addcmul");}
KAPI Addcdiv(K x) {return addc(x, torch::addcdiv, torch::addcdiv_out, "addcdiv");}

// --------------------------------------------------------------------------------------------
// prod,sum - handle multiple signatures (input;type) (input;dim(s);type) etc.
// cumprod,cumsum - handle multiple signatures (input;type) & (input;dim;type)
// logsumexp - return log of summed exponentials of specified dimension of input tensor/array
// --------------------------------------------------------------------------------------------
ZK prodsum(K x,B b,cS e) { // b:true -> prod, false -> sum
 KTRY
  B p,k=false; JRef d; Tensor r,t; c10::optional<ScalarType> s=c10::nullopt;
  J n=xten(x,x->n-1,r) ? x->n-1 : xlen(x); //optional output tensor at end, decrement arg count
  if(xten(x,t) || (r.defined() && n==1 && xten(x,0,t)) || !xmixed(x,4)) { // input as tensor or k array
   if(!(p=t.defined())) t=r.defined() ? kput(x,0) : kput(x);
   if(r.defined())
     return (b ? torch::prod_out(r,t.flatten(),0) : torch::sum_out(r,t,d)), (K)0;
   else 
    return r=b ? torch::prod(t) : torch::sum(t), (p ? kten(r) : kget(r));
  } else if(xtype(x,1,s) && n==2) {             // (input;type)
   if(!(p=xten(x,0,t))) t=kput(x,0);
   if(r.defined())
    return (b ? torch::prod_out(r,t.flatten(),0,k,s) : torch::sum_out(r,t,d,k,s)), (K)0;
   else
    return r=b ? torch::prod(t,s) : torch::sum(t,s), (p ? kten(r) : kget(r));
  } else if(xsize(x,1,d) &&
    (n==2 ||                                       // (input;dim)
    (xtype(x,2,s) &&  n==3)||                      // (input;dim;type)
    (xbool(x,2,k) && (n==3 ||                      // (input;dim;keepdim)
                     (n==4 && xtype(x,3,s)))))) {  // (input;dim;keepdim;type)
   if(b && d.size()!=1) AT_ERROR(e," requires a single dimension, ", d.size(), " supplied");
   if(!(p=xten(x,0,t))) t=kput(x,0);
   if(r.defined())
    return (b ? torch::prod_out(r,t,d[0],k,s) : torch::sum_out(r,t,d,k,s)), (K)0;
   else
    return r=b ? torch::prod(t,d[0],k,s) : torch::sum(t,d,k,s), (p ? kten(r) : kget(r));
  } else {
   if(b) {
    AT_ERROR("prod expects input, (input;type), (input;dim;type), (input;dim;keepdim) or (input;dim;keepdim;type) w'optional output tensor");
   } else {
    AT_ERROR("sum expects input, (input;type), (input;dim(s);type), (input;dim(s);keepdim) or (input;dim(s);keepdim;type) w'optional output tensor");
   }
   return KERR(e);
  }
 KCATCH(e);
}

KAPI Prod(K x) {return prodsum(x,true, "prod");}
KAPI Sum(K x)  {return prodsum(x,false,"sum");}

ZK cprodsum(K x,B b,cS e) { // b:true -> prod, false -> sum
 KTRY
  B p; Tensor r,t; c10::optional<ScalarType> s=c10::nullopt;
  J d,n=xten(x,x->n-1,r) ? x->n-1 : xlen(x); //optional output tensor at end, decrement arg count
  if(x->t==KJ && x->n==2) {
   t=kput(x)[0]; d=kJ(x)[1];
   return kget(b ? torch::cumprod(t,d,s) : torch::cumsum(t,d,s));
  } else if(xlong(x,1,d) && (n==2 || (n==3 && xtype(x,2,s)))) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
   if(r.defined())
    return (b ? torch::cumprod_out(r,t,d,s) : torch::cumsum_out(r,t,d,s)), (K)0;
   else
    return r=b ? torch::cumprod(t,d,s) : torch::cumsum(t,d,s), (p ? kten(r) : kget(r));
  } else {
   AT_ERROR(e," expects (input;dim) or (input;dim;type) w'optional output tensor as additional final argument");
   return KERR(e);
  }
 KCATCH(e);
}

KAPI Cumprod(K x) {return cprodsum(x,true, "cumprod");}
KAPI Cumsum(K x)  {return cprodsum(x,false,"cumsum");}

KAPI Logsumexp(K x) {
 KTRY
  B p,k=false; Tensor r,t; J d,n=xten(x,x->n-1,r) ? x->n-1 : xlen(x);
  if(xlong(x,1,d) && (n==2 || (n==3 && xbool(x,2,k)))) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
   if(r.defined())
    return torch::logsumexp_out(r,t,d,k), (K)0;
   else
    return r=torch::logsumexp(t,d,k), (p ? kten(r) : kget(r));
  } else {
   return KERR("logsumxp expects args of (input;dim) or (input;dim;keepdim) w'optional output tensor as additional final argument");
  }
 KCATCH("logsumexp");
}

// --------------------------------------------------------------------------------------------
// clamp - clamp with min & max, null for min or max does one-sided clamp
// lerp - linear interpolation
// mvlgamma - multivariate log gamma
// pow - power function with scalar or tensor exponent
// --------------------------------------------------------------------------------------------
KAPI Clamp(K x) {
 KTRY
  B p; c10::optional<Scalar> lo,hi; Tensor t,r;
  if(x->t) {
    AT_ERROR("clamp not implemented for ",kname(x->t));
  } else if( !(xnumn(x,1,lo) && xnumn(x,2,hi)) ) {
   AT_ERROR("expected 2nd & 3rd argument of scalar low & high limits (or nulls)");
  } else if(!(x->n==3 || (x->n==4 && xten(x,3,r)))) {
    AT_ERROR("Unexpected clamp arg(s), not one of (tensor/array;lo;hi) or (tensor;lo;hi;output tensor)");
  }
  if(!(p=xten(x,0,t))) t=kput(x,0);
  if(r.defined())
   return torch::clamp_out(r,t,lo,hi), (K)0;
  else
   return r=torch::clamp(t,lo,hi), p ? kten(r) : kget(r);
 KCATCH("clamp");
}

KAPI Lerp(K x) {
 KTRY
  B p; J n=xlen(x); Scalar w; Tensor a,b,r;
  if(n==3 && x->t) {
   return a=kput(x), kget(torch::lerp(a[0],a[1],a[2].item()));
  } else if(xnumt(x,2,w) && (n==3 || (n==4 && xten(x,3,r)))) {
    p=xtenarg(x,a,b);
    if(r.defined())
     return torch::lerp_out(r,a,b,w), (K)0;
    else
     return r=torch::lerp(a,b,w), p ? kten(r) : kget(r);
  } else {
   return KERR("lerp expects array/tensor inputs a,b and wt w: (a;b;w) or (a;b;w;output tensor)");
  }
 KCATCH("Linear interpolation");
}

KAPI Mvlgamma(K x) {
 KTRY
  B p; J d; Tensor t,r;
  if(xlong(x,1,d) && x->n==2) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
   return r=mvlgamma(t,d), p ? kten(r) : kget(r);
  } else {
   return KERR("mvlgamma expects input array/tensor and integer dimemsion, e.g. (input;dim)");
  }
 KCATCH("Multivariate log-gamma");
}

KAPI Pow(K x) {
 KTRY
  Scalar s; Tensor a,b,r;
  B p; J m,n=xten(x,2,r) ? x->n-1 : xlen(x);
  if(n != 2) {             m=-1;
  } else if(xnum(x,0,s)) { m=0; if(!(p=xten(x,1,b))) b=kput(x,1);
  } else if(xnum(x,1,s)) { m=1; if(!(p=xten(x,0,a))) a=kput(x,0);
  } else if(x->t) {        m=2; p=false; a=kput(x);  b=a[1]; a=a[0];
  } else {                 m=2; p=xtenarg(x,a,b);
  }
  if(m<0) {
   return KERR("pow expects input arrays/tensors a,b or scalar s: (a;b), (a;s), (s;b) w'optional output tensor as 3rd arg");
  } else if(r.defined()) {
   if     (m==0) torch::pow_out(r,s,b);
   else if(m==1) torch::pow_out(r,a,s);
   else if(m==2) torch::pow_out(r,a,b);
   return (K)0;
  } else {
   if     (m==0) r=torch::pow(s,b);
   else if(m==1) r=torch::pow(a,s);
   else if(m==2) r=torch::pow(a,b);
   return p ? kten(r) : kget(r);
  }
 KCATCH("Power function");
}

// ---------------------------------------------------------------------
// dist - p-norm of  a - b, with optional exponent p (default p=2)
// fnorm - frobenius norm (tensor, dim(s), keepdim, output tensor)
// nnorm - nuclear norm (tensor, keepdim, output tensor)
// pnorm - p-norm (tensor, p, dim(s), keepdim, output tensor)
// ---------------------------------------------------------------------
KAPI Dist(K x) {
 KTRY
  Scalar n=2; Tensor a,b;
  if(x->t<0) {
   AT_ERROR("dist not implemented for ",kname(x->t));
  } else if(!(x->n==2 || (x->n==3 && (x->t || xnum(x,2,n))))) {
   AT_ERROR("dist expects args of tensor/array a,b and optional exponent n, (a;b) or (a;b;n)");
  }
  if(x->t)
   return a=kput(x), kget(torch::dist(a[0],a[1],(x->n==2) ? n : a[2].item()));
  else
   return xtenarg(x,a,b) ? kten(torch::dist(a,b,n)) : kget(torch::dist(a,b,n));
 KCATCH("dist");
}

KAPI Fnorm(K x) {
 B b=false; JRef d={}; Tensor r,t;
 KTRY
  if(xten(x,t)) {
   return kten(torch::frobenius_norm(t));
  } else if(xten(x,0,t) && x->n>1) {
   if(xten(x,x->n-1,r)) {
    if(x->n==2 || (xsize(x,1,d) && (x->n==3 || (x->n==4 && xbool(x,2,b)))))
     return torch::frobenius_norm_out(r,t,d,b), (K)0;
   } else if(xsize(x,1,d) && (x->n==2 || (x->n==3 && xbool(x,2,b)))) {
    return kten(torch::frobenius_norm(t,d,b));
   }
  } else if(xsize(x,1,d) && (x->n==2 || (x->n==3 && xbool(x,2,b)))) {
   return kget(torch::frobenius_norm(kput(x,0),d,b));
  } else {
   return kget(torch::frobenius_norm(kput(x)));
  }
  return KERR("Unexpected arg(s) for frobenius norm");
 KCATCH("Frobenius norm");
}

KAPI Nnorm(K x) {
 KTRY
  B b=false; Tensor r,t;
  if(xten(x,t)) {
   return kten(torch::nuclear_norm(t));
  } else if((xten(x,1,r) && x->n==2) || (xbool(x,1,b) && xten(x,2,r) && x->n==3)) {
    if(!xten(x,0,t)) t=kput(x,0);
    return torch::nuclear_norm_out(r,t,b), (K)0;
  } else if(xbool(x,1,b) && x->n==2) {
    return xten(x,0,t) ? kten(torch::nuclear_norm(t,b)) : kget(torch::nuclear_norm(kput(x,0),b));
  } else {
   return kget(torch::nuclear_norm(kput(x),b));
  }
 KCATCH("Nuclear norm");
}

KAPI Pnorm(K x) {
 B b=false; JRef d={}; Scalar p=2; Tensor r,t;
 KTRY
  J n=xten(x,x->n-1,r) ? x->n-1 : xlen(x);
  if(xten(x,t)) {
   return kten(torch::norm(t));
  } else if((n==2 && xnum(x,1,p))
         || (n==2 && xsize(x,1,d) && d.size()==1)
         || (n==3 && xsize(x,1,d) && d.size()==1  && xbool(x,2,b))
         || (n==3 && xnum(x,1,p)  && xsize(x,2,d) && d.size()==1)
         || (n==4 && xnum(x,1,p)  && xsize(x,2,d) && d.size()==1 && xbool(x,3,b))) {
    if(xten(x,0,t)) { 
     if(r.defined())
      return (d.size() ? torch::norm_out(r,t,p,d[0],b) : torch::norm_out(r,t.flatten(),p,0,b)), (K)0;
     else
      return kten(d.size() ? torch::norm(t,p,d[0],b) : torch::norm(t,p)); 
    } else {
     t=kput(x,0);
     return kget(d.size() ? torch::norm(t,p,d[0],b) : torch::norm(t,p));
    }
  } else {
   return kget(torch::norm(kput(x)));
  }
 KCATCH("P-norm");
}

// ----------------------------------------------------------------------------
// std deviation & variance: same args from k, call with flag v=true if var()
// ----------------------------------------------------------------------------
ZK variance(K x,B v) {
 KTRY
  B b,k=false,u=true; J d; Tensor r,t;
  if(xten(x,t)) {
   return kten(v ? torch::var(t) : torch::std(t));
  } else if(xbool(x,1,u)) {
   if(x->n==2) {
    if(!(b=xten(x,0,t))) t=kput(x,0);
    r=v ? torch::var(t,u) : torch::std(t,u);
    return b ? kten(r) : kget(r);
   } else if(x->n==3 && xten(x,2,r)) {
    return (v ? torch::var_out(r,t.flatten(),0,u,k) : torch::std_out(r,t.flatten(),0,u,k)), (K)0;
   } else {
    return KERR("Expected args: (input;unbiased flag) or (tensor;unbiased flag;output)");
   }
  } else if(xlong(x,1,d)) {
   J n=xten(x,x->n-1,r) ? x->n-1 : x->n;
   if(n==2 || (n==3 && xbool(x,2,k)) || (n==4 && xbool(x,2,k) && xbool(x,3,u))) {
    if(!(b=xten(x,0,t))) t=kput(x,0);
    if(r.defined())
     return (v ? torch::var_out(r,t,d,k,u) : torch::std_out(r,t,d,k,u)), (K)0;
    r=v ? torch::var(t,d,k,u) : torch::std(t,d,k,u);
    return b ? kten(r) : kget(r);
   } else {
    return KERR("Expected args: (input;dim) or some subset of (input;dim;keepdim;unbiased;output)");
   }
  } else {
   t=kput(x);
   return kget(v ? torch::var(t) : torch::std(t));
  }
 KCATCH("Variance");
}

KAPI Std(K x) {return variance(x,false);}
KAPI Var(K x) {return variance(x,true);}

// ----------------------------------------------------------------------------------------------
// mean - return mean, optionally by dimension(s), will first convert to data type, if supplied 
// median - return median as tensor if no dimension/output args, else return/set values & indices
// mode - mode handled by same function used for median args, but always returns values & indices
// ----------------------------------------------------------------------------------------------
KAPI Mean(K x) {
 KTRY
  B p,k=false; JRef d; Tensor r,t; c10::optional<ScalarType> s=c10::nullopt;
  J n=xten(x,x->n-1,r) ? x->n-1 : xlen(x); //optional output tensor at end, decrement arg count
  if(xten(x,t) || (r.defined() && n==1 && xten(x,0,t)) || !xmixed(x,4)) { // input as tensor or k array
   if(!(p=t.defined())) t=r.defined() ? kput(x,0) : kput(x);
   if(r.defined())
     return torch::mean_out(r,t.flatten(),0), (K)0;
   else
    return r=torch::mean(t), (p ? kten(r) : kget(r));
  } else if(xtype(x,1,s) && n==2) {             // (input;type)
   if(!(p=xten(x,0,t))) t=kput(x,0);
   if(r.defined())
    return torch::mean_out(r,t.flatten(),0,k,s), (K)0;
   else
    return r=torch::mean(t,s), (p ? kten(r) : kget(r));
  } else if(xsize(x,1,d) &&
    (n==2 ||                                       // (input;dim)
    (xtype(x,2,s) &&  n==3)||                      // (imput;dim;type)
    (xbool(x,2,k) && (n==3 ||                      // (input;dim;keepdim)
                     (n==4 && xtype(x,3,s)))))) {  // (input;dim;keepdim;type)
   if(!(p=xten(x,0,t))) t=kput(x,0);
   if(r.defined())
    return torch::mean_out(r,t,d,k,s), (K)0;
   else
    return r=torch::mean(t,d,k,s), (p ? kten(r) : kget(r));
  } else {
   AT_ERROR("mean expects input, (input;type), (input;dim(s);type), (input;dim(s);keepdim) or (input;dim(s);keepdim;type) w'optional output tensor");
   return KERR("mean");
  }
 KCATCH("mean");
}

ZK kmed(K x,B m,cS e) {  //m-true if median, else mode
 KTRY
  B p,k=false; Tensor t,v,i; J d=-1,n=xtenpair(x,x->n-1,v,i) ? x->n-1 : xlen(x);
  if(xten(x,t) || (v.defined() && n==1 && xten(x,0,t)) || !xmixed(x,4)) { // input as tensor or k array
   if(!(p=t.defined())) t=v.defined() ? kput(x,0) : kput(x);
   if(v.defined())
     return (m ? torch::median_out(v,i,t,d,k) : torch::mode_out(v,i,t,d,k)), (K)0;
   else if(m)
     return v=torch::median(t), p ? kten(v) : kget(v);
   else
     return std::tie(v,i)=torch::mode(t), ktenpair(p,v,i);
  } else if(xlong(x,1,d) && (n==2 || (n==3 && xbool(x,2,k)))) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
   if(v.defined())
    return (m ? torch::median_out(v,i,t,d,k) : torch::mode_out(v,i,t,d,k)), (K)0;
   else
    return std::tie(v,i) = m ? torch::median(t,d,k) : torch::mode(t,d,k), ktenpair(p,v,i);
  } else {
   AT_ERROR(e," expects input tensor/array, (input;dim) or (input;dim;keepdim) w'optional output pair of tensors for values,indices");
   return KERR(e);
  }
 KCATCH(e);
}

KAPI Median(K x) {return kmed(x, true,  "median");}
KAPI   Mode(K x) {return kmed(x, false, "mode");}

// ----------------------------------------------------------------------------------------
// comparison fns with arg of (input1;input2;optional output tensor), input2 may be scalar
// ----------------------------------------------------------------------------------------
ZK compare2(K x,Ftt f,Fts fn,Gtt g,Gts gn,cS e) {
 KTRY
  B p; Scalar n; Tensor a,b,r;
  if(2 == (xten(x,2,r) ? x->n-1 : xlen(x))) {
   if(xnum(x,1,n)) {
    if(!(p=xten(x,0,a))) a=kput(x,0);
   } else {
    p=xtenarg(x,a,b);
   }
   if(r.defined())
    return (b.defined() ? g(r,a,b) : gn(r,a,n)), (K)0;
   else
    return r=b.defined() ? f(a,b) : fn(a,n), (p ? kten(r) : kbool(kget(r)));
  } else {
   AT_ERROR(e,": expects args of(input1;input2;optional output tensor), input1 is array or tensor, input2 may also be a number");
   return KERR(e);
  }
 KCATCH(e);
}

KAPI Eq(K x)  {return compare2(x, torch::eq,  torch::eq,  torch::eq_out,  torch::eq_out, "eq()");}
KAPI Ge(K x)  {return compare2(x, torch::ge,  torch::ge,  torch::ge_out,  torch::ge_out, "ge()");}
KAPI GT(K x)  {return compare2(x, torch::gt,  torch::gt,  torch::gt_out,  torch::gt_out, "gt()");}
KAPI Le(K x)  {return compare2(x, torch::le,  torch::le,  torch::le_out,  torch::le_out, "le()");}
KAPI Lt(K x)  {return compare2(x, torch::lt,  torch::lt,  torch::lt_out,  torch::lt_out, "lt()");}
KAPI Ne(K x)  {return compare2(x, torch::ne,  torch::ne,  torch::ne_out,  torch::ne_out, "ne()");}

// -------------------------------------------------------------------------------------------
// comparison functions that return single boolean if arrays are equal or approx. equal
// -------------------------------------------------------------------------------------------
KAPI Allclose(K x) {
 B na=false; F rt=1e-05,at=1e-08; Tensor a,b;
 KTRY
  if(x->t)
   AT_ERROR("allclose not implemented for single ",kname(x->t));
  J n=xbool(x,x->n-1,na) ? x->n-1 : x->n;
  if(n==2 || (xdouble(x,2,rt) && (n==3 || (n==4 && xdouble(x,3,at))))) {
   xtenarg(x,a,b);
   return kb(torch::allclose(a,b,rt,at,na));
  } else {
   AT_ERROR("allclose expects (a;b), (a;b;nan equal), (a;b;rel tol), (a;b;rel tol;abs tol), or (a;b;rel;abs;nan equal)");
   return KERR("allclose");
  }
 KCATCH("allclose");
}

KAPI Equal(K x) {
 KTRY
  Tensor a,b;
  if(xlen(x)==2) {
   if(x->t) a=kput(x), b=a[1], a=a[0];
   else     xtenarg(x,a,b);
   return kb(torch::equal(a,b));
  } else {
   return KERR("equal expects two input arrays/tensors to compare");
  }
 KCATCH("equal");
}

// ----------------------------------------------------------------------------------------------
// comparison functions that check for special values (nan, +/- inf) on floating point tensors
// ----------------------------------------------------------------------------------------------
ZK special(K x, I m, cS e) {
 KTRY
  Tensor r,t; B b=xten(x,t); if(!b) t=kput(x);
  switch(m) {
   case 1:  r = t.is_floating_point() ? ((t==t) == (t.abs()!=wf)) : torch::ones_like(t,torch::dtype(torch::kByte)); break;
   case 2:  r = t.is_floating_point() ? t.abs()==wf               : torch::ones_like(t,torch::dtype(torch::kByte)); break;
   default: r = t != t; break;
  }
  return b ? kten(r) : kbool(kget(r));
 KCATCH(e);
}

KAPI Isfinite(K x) {return special(x, 1, "Check for absence of NaN or infinity");}
KAPI Isinf(K x)    {return special(x, 2, "Check for +/- infinity");}
KAPI Isnan(K x)    {return special(x, 3, "Check for NaN");}

// ------------------------------------------------------------------------------------------------
// minmaxerr - error message for min/max, argmin/max, min/max_values if unrecognized args
// minmax1 - given a single input array/tensor, return min/max value or flattened index of min/max
// minmax2 - given two input arrays/tensors, return/set an array or tensor w'pointwise min/max
// minmaxdim - given single input array/tensor and dimension, return/set values/indices of min/max
// minmaxout - given mode and arg(s), check for output tensor or pair, return remaining arg count
// minmax - main function to evaluate argument pattern and call relevant min/max routine
// ------------------------------------------------------------------------------------------------
ZV minmaxerr(I m,cS e) {
 if(m==0 || m==1) {
  AT_ERROR(e," expects input array/tensor a, or a pair a,b, or input w'dim d and optional keepdim flag k, output val v,ind i:\n"
             "a, (a;b), (a;b;v), (a;d), (a;d;k), (a;d;(v;i)) or (a;d;k;(v;i))");
 } else if(m>1 && m<6) {
  AT_ERROR(e," expects input array/tensor a, dimension d, optional keep dim flag k: a, (a;d), (a;d;k)");
 } else {
  AT_ERROR("Invalid mode for min/max: ",m);
 }
}

ZK minmax1(K x,I m,cS e) {
 B p; Tensor r,t;
 if(!(p=xten(x,t))) t=kput(x);
 switch(m) {
  case 0: r=torch::min(t); break;
  case 1: r=torch::max(t); break;
  case 2: r=torch::argmin(t); break;
  case 3: r=torch::argmax(t); break;
  default: minmaxerr(m,e); break;
 }
 return p ? kten(r) : kget(r);
}

ZK minmax2(K x,I m,Tensor &r,cS e) {
 Tensor a,b; B o=r.defined(),p=xtenarg(x,a,b);
 switch(m) {
  case 0: if(o) torch::min_out(r,a,b); else r=torch::min(a,b); break;
  case 1: if(o) torch::max_out(r,a,b); else r=torch::max(a,b); break;
  default: minmaxerr(m,e); break;
 }
 return o ? (K)0 : (p ? kten(r) : kget(r));
}

ZK minmaxdim(K x,I m,J d,B k,Tensor &v,Tensor &i,cS e) {
 B p,o=v.defined(); Tensor t;
 if(!(p=xten(x,0,t))) t=kput(x,0);
 switch(m) {
  case 0: if(o) torch::min_out(v,i,t,d,k);  else std::tie(v,i)=torch::min(t,d,k); break;
  case 1: if(o) torch::max_out(v,i,t,d,k);  else std::tie(v,i)=torch::max(t,d,k); break;
  case 2: v=torch::argmin(t,d,k); break;
  case 3: v=torch::argmax(t,d,k); break;
  case 4: v=torch::min_values(t,d,k); break;
  case 5: v=torch::max_values(t,d,k); break;
  default: minmaxerr(m,e); break;
 }
 if(o)        return (K)0;
 else if(m<2) return ktenpair(p,v,i);
 else         return p ? kten(v) : kget(v);
}

ZJ minmaxout(K x,I m,B b,Tensor& v,Tensor& i) {
 if(!x->t && x->n>2 && (m==0 || m==1) && ((b && xtenpair(x,x->n-1,v,i)) || (!b && xten(x,x->n-1,v))))
  return x->n-1;
 else
  return xlen(x);
}

ZK minmax(K x,I m,cS e) {
 KTRY
  B k=false; J d; Tensor v,i; B b=xlong(x,1,d); J n=minmaxout(x,m,b,v,i);
  if(b && (n==2 || (n==3 && xbool(x,2,k))))
   return minmaxdim(x,m,d,k,v,i,e);  // (input;dim) or (input;dim;keepdim)
  else if(n==2)                      // (input a;input b), 
   return minmax2(x,m,v,e);          // enlist for single array, e.g. Max enlist(a;b)
  else if(!xmixed(x,4))              // treat single tensor or homogeneous array as one arg
   return minmax1(x,m,e);
  else
   return minmaxerr(m,e), KERR(e);
 KCATCH(e);
}

KAPI        Min(K x) {return minmax(x, 0, "min");}
KAPI        Max(K x) {return minmax(x, 1, "max");}
KAPI     Argmin(K x) {return minmax(x, 2, "argmin");}
KAPI     Argmax(K x) {return minmax(x, 3, "argmax");}
KAPI Min_values(K x) {return minmax(x, 4, "min_values");}
KAPI Max_values(K x) {return minmax(x, 5, "max_values");}

// ----------------------------------------------------------------------------------------------
// sort - sort, optionally by a dimension,optional descending, with output a pair: values,indices
// argsort - sort call, but only return indices (as tensor if tensor supplied, else as array)
// topk - largest/smallest k values by optional dimension & sort flag, return values & indices
// kthvalue - return the kth smallest by optional dimension, return values,indices
// ----------------------------------------------------------------------------------------------
ZK ksort(K x,B a,cS e) {  //x:arg(s), a:flag for argsort() call (only return indices),e:errmsg
 KTRY
  B b=false,p; J d=-1,n=1; Tensor t,v,i;
  if(x->t<0)
   AT_ERROR("sort not implemented for ",kname(x->t));
  n=xtenpair(x,x->n-1,v,i) ? x->n-1 : x->n;  // check for pair of output tensors at end
  if(v.defined()) {
   if(a) {         AT_ERROR("No output pair for argsort() call");
   } else if(!n) { AT_ERROR("Output tensor pair cannot be 1st argument");
   }
  }
  // input tensor, (input;desc), (input;dim) or (input;dim;desc)
  if(xten(x,t) || (xbool(x,1,b) &&  n==2) || (xlong(x,1,d) && (n==2 || (n==3 && xbool(x,2,b))))) {
   if(!(p=t.defined()))
    if(!(p=xten(x,0,t))) t=kput(x,0);
  } else {
   p=false, t=v.defined() ? kput(x,0) : kput(x);
  }
  if(v.defined()) {
   return torch::sort_out(v,i,t,d,b), (K)0;
  } else {
   std::tie(v,i)=torch::sort(t,d,b);
   return a ? (p ? kten(i) : kget(i)) : ktenpair(p,v,i);
  }
 KCATCH(e);
}

KAPI Sort(K x)    {return ksort(x, false, "sort");}
KAPI Argsort(K x) {return ksort(x, true,  "argsort");}

KAPI Topk(K x) {
 KTRY
  B l=true,s=true,p; J k,d=-1,n; Tensor t,v,i;
  if(x->t)
   AT_ERROR("topk not implemented for single ",kname(x->t));
  n=xtenpair(x,x->n-1,v,i) ? x->n-1 : x->n;
  if(!n && v.defined())
   AT_ERROR("Output tensor pair cannot be 1st argument");
  if(xlong(x,1,k) && (n==2 ||                  // (input;k)
    (xlong(x,2,d) && (n==3 ||                  // (input;k;dim)
    (xbool(x,3,l) && (n==4 ||                  // (input;k;dim;desc)
    (xbool(x,4,s) &&  n==5))))))) {            // (input;k;dim;desc;sort)
   if(!(p=xten(x,0,t)))
    t=kput(x,0);
   if(v.defined()) {
    return torch::topk_out(v,i,t,k,d,l,s), (K)0;
   } else {
    std::tie(v,i)=torch::topk(t,k,d,l,s);
    return ktenpair(p,v,i);
   }
  } else {
   AT_ERROR("topk expects: (input;k), (input;k;dim), (input;k;dim;largest) or (input;k;dim;largest;sorted) w'optional (output value;index) tensors");
   return KERR("topk");
  }
 KCATCH("topk");
}

KAPI Kthvalue(K x) {
 KTRY
  B b=false,p; J k,d=-1,n; Tensor t,v,i;
  if(x->t)
   AT_ERROR("kth value not implemented for ",kname(x->t));
  n=xtenpair(x,x->n-1,v,i) ? x->n-1 : x->n;
  if(!n && v.defined())
   AT_ERROR("Output tensor pair cannot be 1st argument");
  if(xlong(x,1,k) && (n==2 ||                 // (input;k)
    (xlong(x,2,d) && (n==3 ||                 // (input;k;dim)
    (xbool(x,3,b) &&  n==4))))) {             // (input;k;dim;keepdim)
   if(!(p=xten(x,0,t)))
    t=kput(x,0);
   if(v.defined()) {
    return torch::kthvalue_out(v,i,t,k,d,b), (K)0;
   } else {
    std::tie(v,i)=torch::kthvalue(t,k,d,b);
    return ktenpair(p,v,i);
   }
  } else {
   AT_ERROR("kth value expects: (input;k), (input;k;dim) or (input;k;dim;keepdim) with optional (output value;index) tensors");
   return KERR("kth value");
  }
 KCATCH("kth value");
}

// -------------------------------------------------------------------------
//  windowing functions: bartlett, blackman, hann & hamming window options
// -------------------------------------------------------------------------
ZK kwindow(K x,I m,cS e) { // m: 0-bartlett, 1-blackman, 2-hann, 3-hamming
 KTRY
  B p; J w; F a,b; Tensor t; TensorOptions o;
  J n=xopt(x,x->n-1,o) ? x->n-1 : xlen(x);
  if(xlong(x,w) ||
    (n==1 && xlong(x,0,w))||
    (n==2 && xlong(x,0,w) && xbool(x,1,p)) ||
    (n==3 && xlong(x,0,w) && xbool(x,1,p) && xdouble(x,2,a) && m==3) ||
    (n==4 && xlong(x,0,w) && xbool(x,1,p) && xdouble(x,2,a) && xdouble(x,3,b) && m==3)) {
   switch(m) {
    case 0: t=(n==1) ? torch::bartlett_window(w,o) : torch::bartlett_window(w,p,o); break;
    case 1: t=(n==1) ? torch::blackman_window(w,o) : torch::blackman_window(w,p,o); break;
    case 2: t=(n==1) ? torch::hann_window(w,o)     : torch::hann_window(w,p,o);     break;
    case 3: if(n==1) t=torch::hamming_window(w,o);
       else if(n==2) t=torch::hamming_window(w,p,o);
       else if(n==3) t=torch::hamming_window(w,p,a,o);
       else if(n==4) t=torch::hamming_window(w,p,a,b,o);
       break;
    default: AT_ERROR("Unrecognized windowing mode, expecting 0-3, received: ",m); break;
   }
   return kten(t);
  } else {
   if(m<3) {
    AT_ERROR(e," expects arg(s) of window, (window;tensor options), (window;periodic;tensor options)");
   } else {
    AT_ERROR(e," expects arg(s) of window, (window;periodic), (window;periodic;alpha) or (window;periodic;alpha;beta), along w'optional tensor options");
   }
   return KERR(e);
  }
 KCATCH(e);
}

KAPI Bartlett_window(K x) {return kwindow(x, 0, "bartlett_window");}
KAPI Blackman_window(K x) {return kwindow(x, 1, "blackman_window");}
KAPI     Hann_window(K x) {return kwindow(x, 2, "hann_window");}
KAPI  Hamming_window(K x) {return kwindow(x, 3, "hamming_window");}

// ---------------------------------------------------------------------------------
// fft - complex-to-complex discrete Fourier transform
// ifft - complex-to-complex inverse discrete Fourier transform
// rfft - real-to-complex discrete Fourier transform
// irfft - complex-to-real inverse discrete Fourier transform
// stft - short-time Fourier transform
// ---------------------------------------------------------------------------------
ZK kfft(K x,I m,cS e) {
 KTRY
  B p,b1=false,b2=true; J d,n=xlen(x); JRef s; Tensor r,t;  // b1-normalized, b2-onesided
  if(xlong(x,1,d) &&
    (n==2 ||
    (n==3 && xbool(x,2,b1)) ||
    (n==4 && xbool(x,2,b1) && xbool(x,3,b2) && m>1) ||
    (n==5 && xbool(x,2,b1) && xbool(x,3,b2) && xsize(x,4,s) && m>2))) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
   switch(m) {
    case 0: r=torch::fft(t,d,b1); break;
    case 1: r=torch::ifft(t,d,b1); break;
    case 2: r=torch::rfft(t,d,b1,b2); break;
    case 3: r=torch::irfft(t,d,b1,b2,s); break;
    default: AT_ERROR("Unrecognized fft mode, expecting 0-3, received: ",m); break;
   }
   return p ? kten(r) : kget(r);
  } else {
   switch(m) {
    case 0:
    case 1: AT_ERROR(e," expects args of (input;dim) or (input;dim;normalized)"); break;
    case 2: AT_ERROR(e," expects args of (input;dim), (input;dim;normalized) or (input;dim;normalized;onesided)"); break;
    case 3: AT_ERROR(e," expects args of (input;dim), (input;dim;normalized), (input;dim;normalized;onesided) or (input;dim;normalized;onesided; sizes)"); break;
    default: AT_ERROR("Unrecognized fft mode, expecting 0-3, received: ",m); break;
   }
   return KERR(e);
  }
 KCATCH(e);
}

KAPI   Fft(K x) {return kfft(x, 0, "fft");}
KAPI  Ifft(K x) {return kfft(x, 1, "ifft");}
KAPI  Rfft(K x) {return kfft(x, 2, "rfft");}
KAPI Irfft(K x) {return kfft(x, 3, "irfft");}
/*
Tensor stft(const Tensor & self,J n_fft,J hop_length,J win_length,const Tensor& window={},B normalized=false,B onesided=true);
torch.stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=True)
      stft(input, n_fft, hop_length, win_length, window, normalized, onesided)
*/

/* OTHER & BLAS routine chained_matmul
broadcast_tensors(*tensors) -> List of Tensors[SOURCE]
einsum(equation, *operands) -> Tensor[SOURCE]
meshgrid(*tensors, **kwargs)[SOURCE]
chain_matmul(*matrices)[SOURCE]
*/

// -------------------------------------------------------------------------------------
// bincount - frequency of each value in an array of non-negative integers
// flatten - flatten a contiguous range of dimensions in a tensor
// flip - reverse the order of a tensor along given dim(s)
// trace - return sum of elements of diagonal of 2-d matrix
// -------------------------------------------------------------------------------------
KAPI Bincount(K x) {
 KTRY
  B p=false; J m=0; Tensor t,w,r;
  if(x->t<0) {
   AT_ERROR("bincount not implemented for single ",kname(x->t));
  } else if(x->t) {
   t=kput(x);
  } else if(xten(x,t)) {
   p=true;
  } else if(x->n==2 && xlong(x,1,m)) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
  } else if(x->n==2 || (x->n==3 && xlong(x,2,m))) {
   p=xtenarg(x,t,w);
  } else {
   AT_ERROR("bincount expects input, (input;min bins), (input;weight) or (input;weight;min bins)");
  }
  r=torch::bincount(t,w,m);
  return p ? kten(r) : kget(r);
 KCATCH("bincount");
}

KAPI Flatten(K x) {
 KTRY
  J i=0,j=-1; Tensor t;
  if(xten(x,t) || (xten(x,0,t) && xlong(x,1,i) && (x->n==2 || (x->n==3 && xlong(x,2,j))))) {
   return kten(torch::flatten(t,i,j));
  } else {
   t=(xlong(x,1,i) && (x->n==2 || (x->n==3 && xlong(x,2,j)))) ? kput(x,0) : kput(x);
   return kget(torch::flatten(t,i,j));
  }
 KCATCH("flatten");
}

KAPI Flip(K x) {
 KTRY
  Tensor t; JRef s;
  if(xten(x,0,t) && xsize(x,1,s))
   return kten(torch::flip(t,s));
  else if(xsize(x,1,s))
   return kget(torch::flip(kput(x,0),s));
  else
   return KERR("Unrecognized arg(s) for pytorch flip: expected (tensor;dims) or (array;dims)");
 KCATCH("flip");
}

KAPI Trace(K x) {
 KTRY
  Tensor t;
  return xten(x,t) ? kten(torch::trace(t)) : kget(torch::trace(kput(x)));
 KCATCH("trace");
}

// -------------------------------------------------------------------------------------
// fns dealing with diagonals, upper & lower triangles
// -------------------------------------------------------------------------------------
ZK diagfns(K x, Fti f, Gti g, cS e) {
 KTRY
  B p; J d=0; Tensor r,t;
  if((x->n==2 && (xlong(x,1,d) || xten(x,1,r))) ||  // (input;diag) or (input;output tensor)
     (x->n==3 &&  xlong(x,1,d) && xten(x,2,r))) {   // (input;diag;output tensor)
   if(!(p=xten(x,0,t))) t=kput(x,0);
  } else if(xten(x,t) || !xmixed(x,3)) {            // input tensor or vector/matrix
   if(!(p=t.defined())) t=kput(x);
  } else {
   AT_ERROR(e," expects vector/matrix/tensor a, optional diagonal d, optional output tensor r: a, (a;d), (a;r) or (a;d;r)");
  }
  if(r.defined())
   return g(r,t,d), (K)0;
  else
   return r=f(t,d), p ? kten(r) : kget(r);
 KCATCH(e);
}

Z Tensor& diagflat_out(Tensor& r, const Tensor& t, int64_t d) {
 AT_ERROR("Error: diagflat() not implemented with output tensor");
}

KAPI Diag(K x)     {return diagfns(x, torch::diag,     torch::diag_out,  "Diagonal diag()");}
KAPI Diagflat(K x) {return diagfns(x, torch::diagflat, diagflat_out,     "Diagonal fill diagflat()");}
KAPI Tril(K x)     {return diagfns(x, torch::tril,     torch::tril_out,  "Lower triangle tril()");}
KAPI Triu(K x)     {return diagfns(x, torch::triu,     torch::triu_out,  "Upper triangle triu()");}

KAPI Diagonal(K x) {  //extract diagonal elements, optional offset & dimensions i,j
 KTRY
  B p; J o=0,i=0,j=1; Tensor r,t;
  if(x->t) {
   AT_ERROR("diagonal not implemented for ",kname(x->t));
  } else if(xlong(x,1,o) && (x->n==2 || (xlong(x,2,i) && (x->n==3 || (x->n==4 && xlong(x,3,j)))))) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
  } else {
   if(!(p=xten(x,t))) t=kput(x);
  }
  r=torch::diagonal(t,o,i,j);
  return p ? kten(r) : kget(r);
 KCATCH("diagonal");
}

// -------------------------------------------------------------------------------------------
// histc - histogram of inputs with optional number of bins, min & max bin, output tensor
// cross - cross product of two tensors/arrays, with optional dimension & output tensor
// renorm - renormalize along given dimension using given p-norm exponent and max norm
// roll - specify single or multiple shifts & dimensions to rotate tensor
// tensordot - returns a contraction of a and b over multiple dimensions (new for version 1.0)
// unique - return unique elements in input along with optional indicies
// -------------------------------------------------------------------------------------------
KAPI Histc(K x) {
 KTRY
  J b=100,p; Scalar lo=0,hi=0; Tensor r,t;
  if(x->t)
   return kget(torch::histc(kput(x)));
  else if(xten(x,t))
   return kten(torch::histc(t));
  J n=(x->n>1 && xten(x,x->n-1,r)) ? x->n-1 : x->n;  //arg count excluding output tensor if supplied
  if( !(n==1 || (xlong(x,1,b) && (n==2 || (xnum(x,2,lo) && (n==3 || (xnum(x,3,hi) && n==4)))))) )
   AT_ERROR("histc arg(s): (input tensor/array; optional bin count; optional lo; optional hi; optional output tensor)");
  if( !(p=xten(x,0,t)) )
   t=kput(x,0);
  if(r.defined())
   return torch::histc_out(r,t,b,lo,hi), (K)0;
  else
   return r=torch::histc(t,b,lo,hi), p ? kten(r) : kget(r);
 KCATCH("histc");
}

KAPI Cross(K x) {
 KTRY
  J d=-1; Tensor r,a,b;
  if( !(!x->t && (x->n==2 || (xlong(x,2,d) && (x->n==3 || (x->n==4 && xten(x,3,r)))))) )
   AT_ERROR("Unexpected arg(s) for cross, expected (tensor/array; tensor/array; optional dim; optional output tensor)");
  B p=xtenarg(x,a,b);
  if(r.defined()) {
   return torch::cross_out(r,a,b,d), (K)0;
  } else {
   r=torch::cross(a,b,d);
   return p ? kten(r) : kget(r);
  }
 KCATCH("cross");
}

KAPI Renorm(K x) {
 KTRY
  B k; J d; Scalar p,m; Tensor r,t;
  if( !(xnum(x,1,p) && xlong(x,2,d) && xnum(x,3,m) && (x->n==4 || (x->n==5 && xten(x,4,r)))) )
   AT_ERROR("Unexpected arg(s) for renorm, expected (tensor/array; power; dim; maxnorm; optional output tensor)");
  if((k=!xten(x,0,t))) t=kput(x,0);
  if(r.defined()) {
   return torch::renorm_out(r,t,p,d,m), (K)0;
  } else {
   r=torch::renorm(t,p,d,m);
   return k ? kget(r) : kten(r);
  }
 KCATCH("renorm");
}

KAPI Roll(K x) {
 KTRY
  B p; JRef s,d; Tensor r,t;
  if(xsize(x,1,s) && (x->n==2 || (xsize(x,2,d) && x->n==3))) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
   r=x->n==2 ? torch::roll(t,s) : torch::roll(t,s,d);
   return p ? kten(r) : kget(r);
  } else {
   return KERR("roll expects (tensor/array input; shift(s); optional dimension(s))");
  }
 KCATCH("roll");
}

KAPI Tensordot(K x) {
 KTRY
  J d=2; JRef i,j; Tensor a,b,r;
  if(x->t<0) {
   AT_ERROR("tensor dot is not implemented for ",kname(x->t));
  } else if(x->t && (x->n==2 || x->n==3)) {
   a=kput(x);
   if(x->n==3 && a[2].item().toDouble()!=0)
    AT_ERROR("tensordot: non-zero dimension specified for scalars");
   return kget(torch::tensordot(a[0],a[1],i,j));
  } else if(x->n==2 || (x->n==3 && xlong(x,2,d)) || (x->n==4 && xsize(x,2,i) && xsize(x,3,j))) {
   B p=xtenarg(x,a,b);
   if(x->n<4) {
    Ksize s1,s2;
    for(I k=0;k<d;++k) s1.push_back(k-d), s2.push_back(k);
    i=s1; j=s2;
   }
   r=torch::tensordot(a,b,i,j);
   return p ? kten(r) : kget(r);
  } else {
   AT_ERROR("tensor args: (array/tensor; array/tensor) with optional dim or (dimension list;dimension list)");
   return KERR("tensordot");
  }
 KCATCH("tensordot");
}

ZK uniqres(B p,B bi,B bc,Tensor &u,Tensor &i, Tensor &c) {
 if(bi && bc)return kten3(p,u,i,c);
 else if(bi) return ktenpair(p,u,i);
 else if(bc) return ktenpair(p,u,c);
 else        return p ? kten(u) : kget(u);
}

KAPI Unique(K x) {
 KTRY
  B p,bs=true,bi=false,bc=false; J d=nj,n=xlen(x); Tensor t,u,i,c;
  if(xten(x,t)) {
   p=true;
  } else if(!xmixed(x,5)) {
   p=false, t=kput(x);
  } else if(
   (n==2 && (xbool(x,1,bs) ||  xlong(x,1,d)))||
   (n==3 &&  xbool(x,1,bs) && (xbool(x,2,bi) || xlong(x,2,d))) ||
   (n==4 &&  xbool(x,1,bs) &&  xbool(x,2,bi) && (xbool(x,3,bc) || xlong(x,3,d))) ||
   (n==5 &&  xbool(x,1,bs) &&  xbool(x,2,bi) &&  xbool(x,3,bc) && xlong(x,4,d))) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
  } else {
   AT_ERROR("unique expects input array/tensor, followed by optional flag(s) and optional dimension as last arg:\n"
            "(input;sort flag;indices flag;counts flag;dimension)");
  }
  std::tie(u,i,c)=(d==nj) ? torch::_unique2(t,bs,bi,bc) : torch::unique_dim(t,d,bs,bi,bc);
  return uniqres(p,bi,bc,u,i,c);
 KCATCH("unique");
}

KAPI Uniquec(K x) {
 KTRY
  B p,bi=false,bc=false; J n=xlen(x); int64_t d=nj; Tensor t,u,i,c;
  if(xten(x,t)) {
   p=true;
  } else if(!xmixed(x,4)) {
   p=false, t=kput(x);
  } else if(
   (n==2 && (xbool(x,1,bi) || xint64(x,1,d))) ||
   (n==3 &&  xbool(x,1,bi) && (xbool(x,2,bc) || xint64(x,2,d))) ||
   (n==4 &&  xbool(x,1,bi) &&  xbool(x,2,bc) && xint64(x,3,d))) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
  } else {
   AT_ERROR("unique consecutive expects input array/tensor, followed by optional flag(s) and optional dimension as last arg:\n"
            "(input;indices flag;counts flag;dimension)");
  }
  std::tie(u,i,c)=torch::unique_consecutive(t,bi,bc,(d==nj) ? torch::nullopt : torch::make_optional(d));
  return uniqres(p,bi,bc,u,i,c);
 KCATCH("unique consecutive");
}

// --------------------------------------------------------------------------
// addmm - beta * mat + alpha * mat1 * mat2
// addbmm - beta * mat + alpha * sum of batch1 * batch2
// baddbmm - beta * batchmat + alpha * sum of batch1 * batch2
// addmv - beta * vector + alpha * mat1 * vec1
// addr - beta * mat + alpha * outter product of vec1,vec2
// --------------------------------------------------------------------------
ZK kaddmm(K x,Fmm f,Gmm g,cS e) {
 KTRY
  J p=3,n=xlen(x); Scalar a=1,b=1; Tensor r,t,t1,t2;
  if(n>3 && xten(x,n-1,r)) n-=1;
  if(n==3 || (xnum(x,3,b) && (n==4 || (n==5 && xnum(x,4,a))))) {
   if(!xten(x,0,t))   t=kput(x,0), p-=1;
   if(!xten(x,1,t1)) t1=kput(x,1), p-=1;
   if(!xten(x,2,t2)) t2=kput(x,2), p-=1;
   if(r.defined())
    return g(r,t,t1,t2,b,a), (K)0;
   else
    return r=f(t,t1,t2,b,a), (p ? kten(r) : kget(r));
  }
  AT_ERROR(e," expects 3 tensor or array inputs, followed by optional beta,alpha & output tensor(if both beta & alpha, supply beta first)");
  return KERR(e);
 KCATCH(e);
}

KAPI   Addmm(K x) {return kaddmm(x, torch::addmm,   torch::addmm_out,   "addmm");}
KAPI  Addbmm(K x) {return kaddmm(x, torch::addbmm,  torch::addbmm_out,  "addbmm");}
KAPI Baddbmm(K x) {return kaddmm(x, torch::baddbmm, torch::baddbmm_out, "baddbmm");}
KAPI   Addmv(K x) {return kaddmm(x, torch::addmv,   torch::addmv_out,   "addmv");}
KAPI    Addr(K x) {return kaddmm(x, torch::addr,    torch::addr_out,    "addr");}

// ------------------------------------------------------------------------------------
// lu,lu_info - LU factorization, returns tuple w'LU factors, pivots & optional info
//              originally 4 separate routines, btrifact/out/info/info_out, 
//              as of Mar/Apr'19 collapsed & renamed to single routine: _lu_with_info
//              preserved two calls from k to simplify boolean and output pair/triplet
// lu_solve - batch LU solve of the linear system Ax = bAx=b
// lu_unpack - unpack data & pivots from batched LU factorization of tensor
// ------------------------------------------------------------------------------------
KAPI lufact(K x,B y,cS e) {  // x:args, y:true if info required, e:error label
 KTRY
  B p,b=true; Tensor t,u,v,i,U,V,I; // p-true if ptr(s) in/out, else arrays, b:pivot flag
  J n=((!y && xtenpair(x,x->n-1,u,v)) || (y && xten3(x,x->n-1,u,v,i))) ? x->n-1 : xlen(x);
  if(xten(x,t) || (u.defined() && n==1 && xten(x,0,t)) || !xmixed(x,4)) { // input as tensor or k array
   if(!(p=t.defined())) t=u.defined() ? kput(x,0) : kput(x);
  } else if(xbool(x,1,b) && x->n==2) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
  } else {
   AT_ERROR(e, " expects input tensor/array or (input;pivot flag) w'optional set of ",(y ? 3 : 2)," output tensors at end");
  }
  std::tie(U,V,I)=torch::_lu_with_info(t,b,!y); // set check error flag true if no info required, else unneccessary
  if(u.defined()) {
    tensorcopy(u,U); tensorcopy(v,V); if(y) tensorcopy(i,I);  // (u,v,..) <- (U,V,..)
    return (K)0;
  } else {
   return y ? kten3(p,U,V,I) : ktenpair(p,U,V);
  }
 KCATCH(e);
}

KAPI Lu(K x)      {return lufact(x, false, "LU factorization");}
KAPI Lu_info(K x) {return lufact(x, true,  "LU factorization w'info");}


KAPI Lu_solve(K x) {
 KTRY
  J p=3; Tensor r,t,d,v;  //r:result,t:input,d:lu data,v:lu pivots
  if(!x->t && (x->n==3 || (x->n==4 && xten(x,3,r)))) {
   if(!xten(x,0,t)) t=kput(x,0), p-=1;
   if(!xten(x,1,d)) d=kput(x,1), p-=1;
   if(!xten(x,2,v)) v=kput(x,2), p-=1;
   if(r.defined())
    return torch::lu_solve_out(r,t,d,v), (K)0;
   else
    return t=torch::lu_solve(t,d,v), (p ? kten(r) : kget(r));
  } else {
   return KERR("lu_solve expects three input tensors/arrays and optional output tensor");
  }
 KCATCH("lu_solve");
}

KAPI Lu_unpack(K x) {
 KTRY
  B b1=true,b2=true; Tensor a,b,l,u,v;
  if(!(x->n==2 || (xbool(x,2,b1) && (x->n==3 || (xbool(x,3,b2) && x->n==4))))) {
   AT_ERROR("lu_unpack expects 2 input arrays/tensors and 2 optional boolean flags");
   return KERR("lu_unpack");
  }
  B p=xtenarg(x,a,b); J n=a.size(-1);
  K r=ktn(0,3);
  if(b1) {
   auto t0=torch::tensor({0.0}).type_as(a);
   auto t1=torch::ones({n,n},TensorOptions().device(a.device()).dtype(torch::kByte)).triu_().expand_as(a);
   auto  u=torch::where(t1,a,t0);
   auto  l=torch::where(t1,t0,a);
   l.diagonal(0,-2,-1).fill_(1);
   kK(r)[1]=p ? kten(l) : kget(l);
   kK(r)[2]=p ? kten(u) : kget(u);
  } else {
   kK(r)[1]=ktn(0,0); kK(r)[2]=ktn(0,0);
  }
  if(b2) {
   auto v=torch::eye(n,TensorOptions().device(a.device()).dtype(a.dtype())).expand_as(a).clone();
   J i,j,m=v.size(-1),n=1,d=v.dim()-2;
   b=(b-1).contiguous().flatten().to(torch::kCPU,torch::kInt); auto bp=(I*)b.data_ptr();
   // vi:tensor list for indices e.g. v.index(vi) equivalent to v[i][j] for 4-dim v, ignored for 2-dim
   auto vi=torch::split(torch::zeros(d,TensorOptions().device(v.device()).dtype(torch::kLong)),1);
   auto vj=torch::empty(v.size(-1),TensorOptions().device(torch::kCPU).dtype(torch::kLong));
   auto vp=(J*)vj.data_ptr();
   for(i=0;i<d;++i) n*=v.size(i);
   for(i=0;i<n;++i) {
    for(j=0;j<m;++j) vp[j]=j;
    for(j=0;j<m;++j) {auto k=*bp++; auto v=vp[j]; vp[j]=vp[k]; vp[k]=v;}
    if(d) v.index_put_(vi,v.index(vi).index_select(-1,vj.to(v.device())));
    else v=index_select(v,-1,vj.to(v.device()));
    for(j=d-1;j>-1;--j) {if(vi[j].item().toLong()==v.size(j)-1) {vi[j].zero_();} else {vi[j]+=1; break;}}
   }
   kK(r)[0]=p ? kten(v) : kget(v);
  } else {
   kK(r)[0]=ktn(0,0);
  }
  return r;
 KCATCH("lu_unpack");
}

// ------------------------------------------------------------------------------------------
// matrix_power - raise matrix or batch of matrices to given integer power (may be negative)
// matrix_rank - return rank of 2-d tensor, specify optional tolerance and symmetric flag
// ------------------------------------------------------------------------------------------
KAPI Matrix_power(K x) {
 KTRY
  B p; J n; Tensor r,t;
  if(xlong(x,1,n)) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
   return r=torch::matrix_power(t,n), (p ? kten(r) : kget(r));
  } else {
   return KERR("matrix_power expects arguments of (array/tensor;long n)");
  }
 KCATCH("matrix_power");
}

KAPI Matrix_rank(K x) {
 KTRY
  B s=false,p; F f; Tensor r,t;
  if(xten(x,t)) {
   return kten(torch::matrix_rank(t));
  } else if(xbool(x,1,s) && x->n==2) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
   return r=torch::matrix_rank(t,s), (p ? kten(r) : kget(r));
  } else if(xdouble(x,1,f) && (x->n==2 || (x->n==3 && xbool(x,2,s)))) {
   if(!(p=xten(x,0,t))) t=kput(x,0);
   return r=torch::matrix_rank(t,f,s), (p ? kten(r) : kget(r));
  } else {
   return kget(torch::matrix_rank(kput(x)));
  }
 KCATCH("matrix_rank");
}

// ----------------------------------------------------------------------------------------
// det,logdet,slogdet - determinant, log determinant & log determinant w'sign
// ----------------------------------------------------------------------------------------
ZK kdet(K x,I m,cS e) { //x:arg, m:mode 0-det,1-logdet,2-slogdet, e:errmsg
 KTRY
  B p; Tensor a,d,s;
  if(!(p=xten(x,a))) a=kput(x);
  if(m==2) {
   std::tie(s,d)=torch::slogdet(a);
   return ktenpair(p,s,d);
  } else {
   d=m ? torch::logdet(a) : torch::det(a);
   return p ? kten(d) : kget(d);
  }
 KCATCH(e);
}

KAPI Det(K x)     {return kdet(x, 0, "determinant");}
KAPI Logdet(K x)  {return kdet(x, 1, "determinant");}
KAPI Slogdet(K x) {return kdet(x, 2, "sign & log determinant");}

// --------------------------------------------------------------------------------------------------
// blas2 - BLAS fns with 2 input tensors/arrays & optional output tensor, return tensor or set output
// --------------------------------------------------------------------------------------------------
ZK blas2(K x, Ftt f, Gtt g, cS e) {
 KTRY
  B p; Tensor a,b,r;
  if(!x->t && (x->n==2 || (x->n==3 && xten(x,2,r)))) {
   p=xtenarg(x,a,b);
   if(r.defined())
    return g(r,a,b), (K)0;
   else
    return p ? kten(f(a,b)) : kget(f(a,b));
  } else {
  if(x->t) {
   AT_ERROR(e," not implemented for single ",kname(x->t));
  } else {
   AT_ERROR("Unrecognized arg(s) for: ",e,", expecting (a;b) or (a;b;output tensor) where a & b are arrays or tensors");
  }
  }
 KCATCH(e);
}

/*
using Ftt     = Tensor  (*)(const Tensor&, const Tensor&);
using Gtt     = Tensor& (*)(Tensor&, const Tensor&, const Tensor&);
*/

Z Tensor  mtm(              const Tensor& a,const Tensor&b) {return torch::mm(      a.t(),b);}
Z Tensor& mtm_out(Tensor& r,const Tensor& a,const Tensor&b) {return torch::mm_out(r,a.t(),b);}
Z Tensor  mmt(              const Tensor& a,const Tensor&b) {return torch::mm(      a,b.t());}
Z Tensor& mmt_out(Tensor& r,const Tensor& a,const Tensor&b) {return torch::mm_out(r,a,b.t());}

KAPI Bmm(K x)    {return blas2(x, torch::bmm,     torch::bmm_out,    "batch matrix-matrix product");}
KAPI Dot(K x)    {return blas2(x, torch::dot,     torch::dot_out,    "dot product");}
KAPI Ger(K x)    {return blas2(x, torch::ger,     torch::ger_out,    "outer product");}
KAPI Matmul(K x) {return blas2(x, torch::matmul,  torch::matmul_out, "generalized matrix product");}
KAPI Mm(K x)     {return blas2(x, torch::mm,      torch::mm_out,     "matrix multiplication");}
KAPI Mmt(K x)    {return blas2(x, mmt,            mmt_out,           "matrix multiplication(A*B')");}
KAPI Mtm(K x)    {return blas2(x, mtm,            mtm_out,           "matrix multiplication(A'*B)");}
KAPI Mv(K x)     {return blas2(x, torch::mv,      torch::mv_out,     "matrix-vector product");}
KAPI Orgqr(K x)  {return blas2(x, torch::orgqr,   torch::orgqr_out,  "orthorganal matrix of QR factorization");}

// --------------------------------------------------------------------------------------
// pinverse - pseudo-inverse (the Moore-Penrose inverse) of a 2D tensor
// qr - qr decomposition, returns orthoganal and upper triangular matrix
// geqrf - qr decomposition using lower level BLAS routine, returns "reflector" matrices
// ormqr - multiply mat by orthogonal Q matrix of the QR factorization formed by geqrf
// svd - singular value decomposition of a real matrix
// --------------------------------------------------------------------------------------
KAPI Pinverse(K x) {
 KTRY
  B p; F f=1e-15; Tensor t,r;
  if (xten(x,t))
   return kten(torch::pinverse(t));
  else if (xdouble(x,1,f))
   return r=torch::pinverse((p=xten(x,0,t)) ? t : kput(x,0), f), (p ? kten(r) : kget(r));
  else
   return kget(torch::pinverse(kput(x)));
 KCATCH("Psuedo-inverse");
}

KAPI Qr(K x) {
 KTRY
  B b=true; Tensor t,q,r;  //flag true for reduced, false for complete QR decomposition
  if(x->t) {
   AT_ERROR("QR factorization not supported for ",kname(x->t));
  } else if(xtenpair(x,x->n-1,q,r)) {
    if (x->n==2 || (x->n==3 && xbool(x,1,b))) {
     torch::qr_out(q,r,xten(x,0,t) ? t : kput(x,0),b);
     return (K)0;
    } else {
     AT_ERROR("AR factorization: output pair detected, but args not of form: (matrix/tensor;output pair) or (matrix/tensor;flag;output pair)");
    }
  } else {
   B p;
   if(xbool(x,1,b) && x->n==2) {
    if(!(p=xten(x,0,t))) t=kput(x,0);
   } else {
    if(!(p=xten(x,t))) t=kput(x);
   }
   std::tie(q,r)=torch::qr(t,b);
   return ktenpair(p,q,r);
  }
 KCATCH("qr");
}

KAPI Geqrf(K x) {
 KTRY
  Tensor t,q,r;
  if(xten(x,t))
   return std::tie(q,r)=torch::geqrf(t), ktenpair(true,q,r);
  else if(xtenpair(x,1,q,r))
   return torch::geqrf_out(q,r,xten(x,0,t) ? t : kput(x,0)), (K)0;
  else
   return std::tie(q,r)=torch::geqrf(kput(x)), ktenpair(false,q,r);
 KCATCH("geqrf");
}

KAPI Ormqr(K x) {
 KTRY
  B l=true,t=false; Tensor a,b,c,r; J p=3,n=xten(x,x->n-1,r) ? x->n-1 : xlen(x);
  if(n==3 || (xbool(x,3,l) && (n==4 || (n==5 && xbool(x,4,t))))) {
   if(!xten(x,0,a)) a=kput(x,0), p-=1;
   if(!xten(x,1,b)) b=kput(x,1), p-=1;
   if(!xten(x,2,c)) c=kput(x,2), p-=1;
   if(r.defined())
    return torch::ormqr_out(r,a,b,c,l,t), (K)0;
   else
    return r=torch::ormqr(a,b,c,l,t), (p ? kten(r) : kget(r));
  } else {
   return KERR("ormqr expects 3 input arrays/tensors, one or two optional boolean flags and an optional output tensor");
  }
 KCATCH("ormqr");
}

KAPI Svd(K x) {
 KTRY
  B p=true,b1=true,b2=true; Tensor t,u,s,v; J n=xten3(x,x->n-1,u,s,v) ? x->n-1 : xlen(x);
  if(xten(x,t) || (n==1 && u.defined()) || (xbool(x,1,b1) && (n==2 || (n==3 && xbool(x,2,b2))))) {
   if(!t.defined() && !(p=xten(x,0,t))) t=kput(x,0);
  } else if(!xmixed(x,2)) {
   p=false,t=kput(x);
  } else {
   AT_ERROR("svd expects matrix/tensor input or (input;optional flag1;optional flag2;optional output triplet of tensors");
  }
  if(u.defined())
   return torch::svd_out(u,s,v,t,b1,b2), (K)0;
  else
   return std::tie(u,s,v)=torch::svd(t,b1,b2), kten3(p,u,s,v);
 KCATCH("svd");
}

// --------------------------------------------------------------------------------------
// gels - solution to least squares/norm problems for Ax=B, returns x & QR factorization
// solve - solution to least squares for Ax=B, returns x & LU factorization (was: gesv)
// triangular_solve - solves Ax=b w'triangular matrix A and multiple right-hand sides b
// --------------------------------------------------------------------------------------
ZK gls(K x,Ftuple2 f,Gtuple2 g,cS e) {
 KTRY
  B p; Tensor a,b,c,d;
  if(x->n==2)
   return p=xtenarg(x,b,a), std::tie(c,d)=f(b,a), ktenpair(p,c,d);
  else if(x->n==3 && xtenpair(x,2,c,d))
   return p=xtenarg(x,b,a), g(c,d,b,a), (K)0;
  else
   AT_ERROR(e," expects args of (tensor/matrix;tensor/matrix;optional output pair of tensors)");
 KCATCH(e);
}

KAPI Gels(K x) {return gls(x, torch::gels, torch::gels_out, "gels");}
KAPI Solve(K x) {return gls(x, torch::solve, torch::solve_out, "solve(gesv)");}

KAPI Triangular_solve(K x) {
 KTRY
  B p,u=true,t=false,g=false; Tensor a,b,c,d;  //u-upper triangle, t-transpose, a=unitriangular
  J n=xtenpair(x,x->n-1,c,d) ? x->n-1 : xlen(x);
  if(n==2 || (xbool(x,2,u) && (n==3 || (xbool(x,3,t) && (n==4 || (xbool(x,4,g) && n==5)))))) {
   p=xtenarg(x,b,a);
   if(c.defined())
    return torch::triangular_solve_out(c,d,b,a,u,t,g), (K)0;
   else
    return std::tie(c,d)=torch::triangular_solve(b,a,u,t,g), ktenpair(p,c,d);
  } else {
   return KERR("triangular_solve expects two input matrices/tensors, up to three optional boolean flags, optional output tensor pair");
  }
 KCATCH("trangular solve");
}

// ---------------------------------------------------------------------------------------------
// cholesky - cholesky decomposition
// cholesky_inverse - inverse of symmetric positive-definite matrix using cholesky factorization
// cholesky_solve - solves equations w'positive semidefinite matrix and cholesky factors
// ---------------------------------------------------------------------------------------------
ZK chol(K x,Ftb f,Gtb g,B b,cS e) {
 KTRY
  Tensor r,t;
  if(xten(x,t)) {
   return kten(f(t,b));
  } else if((xten(x,1,r) && x->n==2) || (xbool(x,1,b) && xten(x,2,r) && x->n==3)) {
    if(!xten(x,0,t)) t=kput(x,0);
    return g(r,t,b), (K)0;
  } else if(x->n==2 && xbool(x,1,b)) {
    return xten(x,0,t) ? kten(f(t,b)) : kget(f(kput(x,0),b));
  } else {
   return kget(f(kput(x),b));
  }
 KCATCH(e);
}

KAPI Cholesky(K x)         {return chol(x, torch::cholesky,         torch::cholesky_out,         false, "Cholesky decomposition");}
KAPI Cholesky_inverse(K x) {return chol(x, torch::cholesky_inverse, torch::cholesky_inverse_out, true,  "Invert positive semi-definite matrix given Cholesky factors");}

KAPI Cholesky_solve(K x) {
 KTRY
  B p,u=false; J n=xlen(x); Tensor a,b,r;
  if(n==2 || (n==3 && (xbool(x,2,u) || xten(x,2,r))) || (n==4 && xbool(x,2,u) && xten(x,3,r))) {
   p=xtenpair(x,a,b);
   if(r.defined())
    return torch::cholesky_solve_out(r,a,b,u), (K)0;
   else
    return r=torch::cholesky_solve(a,b,u), (p ? kten(r) : kget(r));
  } else {
   AT_ERROR("Unexpected args for cholesky_solve, expected (matrix;matrix;optional upper flag;optional output tensor)");
   return KERR("Cholesky solve");
  }
 KCATCH("Cholesky solve");
}
// -------------------------------------------------------------------------
// BLAS routines for eigenvalues of matrix & symmetric matrix
// -------------------------------------------------------------------------
KAPI Eig(K x) {
 KTRY
 B b=false,p; Tensor a,e,v;
 if(xten(x,a)) {
  p=true;
 } else if((x->n==2 && (xbool(x,1,b) || xtenpair(x,1,e,v))) ||
           (x->n==3 &&  xbool(x,1,b) && xtenpair(x,2,e,v))) {
   if(!(p=xten(x,0,a))) 
    a=kput(x,0);
 } else {
  p=false;
  a=kput(x);
 }
 if(e.defined())
  return torch::eig_out(e,v,a,b), (K)0;
 else
  return std::tie(e,v)=torch::eig(a,b), ktenpair(p,e,v);
 KCATCH("Eigenvalues");
}

KAPI Symeig(K x) {
 KTRY
 B b=false,u=true,p; Tensor a,e,v;
 if(xten(x,a)) {
  p=true;
 } else if((x->n==2 && (xbool(x,1,b) ||                  xtenpair(x,1,e,v))) ||
           (x->n==3 &&  xbool(x,1,b) && (xbool(x,2,u) || xtenpair(x,2,e,v))) ||
           (x->n==4 &&  xbool(x,1,b) &&  xbool(x,2,u) && xtenpair(x,3,e,v))) {
   if(!(p=xten(x,0,a))) 
    a=kput(x,0);
 } else {
  p=false;
  a=kput(x);
 }
 if(e.defined())
  return torch::symeig_out(e,v,a,b), (K)0;
 else
  return std::tie(e,v)=torch::symeig(a,b), ktenpair(p,e,v);
 KCATCH("Eigenvalues of a symmetric matrix");
}

// -------------------------------------------------------------------------------------
// map api function to name in q session, upper case for 1st letter if reserved in k
// -------------------------------------------------------------------------------------
V mathfn(K x) {
 fn(x, "Abs",                KFN(Abs),                1);
 fn(x, "Acos",               KFN(Acos),               1);
 fn(x, "add",                KFN(Add),                1);
 fn(x, "addbmm",             KFN(Addbmm),             1);
 fn(x, "addmm",              KFN(Addmm),              1);
 fn(x, "addmv",              KFN(Addmv),              1);
 fn(x, "addr",               KFN(Addr),               1);
 fn(x, "addcdiv",            KFN(Addcdiv),            1);
 fn(x, "addcmul",            KFN(Addcmul),            1);
 fn(x, "allclose",           KFN(Allclose),           1);
 fn(x, "argmax",             KFN(Argmax),             1);
 fn(x, "argmin",             KFN(Argmin),             1);
 fn(x, "argsort",            KFN(Argsort),            1);
 fn(x, "Asin",               KFN(Asin),               1);
 fn(x, "Atan",               KFN(Atan),               1);
 fn(x, "atan2",              KFN(Atan2),              1);
 fn(x, "baddbmm",            KFN(Baddbmm),            1);
 fn(x, "bartlett_window",    KFN(Bartlett_window),    1);
 fn(x, "blackman_window",    KFN(Blackman_window),    1);
 fn(x, "bincount",           KFN(Bincount),           1);
 fn(x, "bmm",                KFN(Bmm),                1);
 fn(x, "ceil",               KFN(Ceil),               1);
 fn(x, "cholesky",           KFN(Cholesky),           1);
 fn(x, "cholesky_inverse",   KFN(Cholesky_inverse),   1);
 fn(x, "cholesky_solve",     KFN(Cholesky_solve),     1);
 fn(x, "clamp",              KFN(Clamp),              1);
 fn(x, "Cos",                KFN(Cos),                1);
 fn(x, "cosh",               KFN(Cosh),               1);
 fn(x, "Cross",              KFN(Cross),              1);
 fn(x, "cumprod",            KFN(Cumprod),            1);
 fn(x, "cumsum",             KFN(Cumsum),             1);
 fn(x, "det",                KFN(Det),                1);
 fn(x, "diag",               KFN(Diag),               1);
 fn(x, "diagflat",           KFN(Diagflat),           1);
 fn(x, "diagonal",           KFN(Diagonal),           1);
 fn(x, "digamma",            KFN(Digamma),            1);
 fn(x, "dist",               KFN(Dist),               1);
 fn(x, "Div",                KFN(Div),                1);
 fn(x, "dot",                KFN(Dot),                1);
 fn(x, "eig",                KFN(Eig),                1);
 fn(x, "eq",                 KFN(Eq),                 1);
 fn(x, "equal",              KFN(Equal),              1);
 fn(x, "erf",                KFN(Erf),                1);
 fn(x, "erfc",               KFN(Erfc),               1);
 fn(x, "erfinv",             KFN(Erfinv),             1);
 fn(x, "Exp",                KFN(Exp),                1);
 fn(x, "expm1",              KFN(Expm1),              1);
 fn(x, "fft",                KFN(Fft),                1);
 fn(x, "flatten",            KFN(Flatten),            1);
 fn(x, "Flip",               KFN(Flip),               1);
 fn(x, "Floor",              KFN(Floor),              1);
 fn(x, "fmod",               KFN(Fmod),               1);
 fn(x, "fnorm",              KFN(Fnorm),              1);
 fn(x, "frac",               KFN(Frac),               1);
 fn(x, "ge",                 KFN(Ge),                 1);
 fn(x, "gels",               KFN(Gels),               1);
 fn(x, "geqrf",              KFN(Geqrf),              1);
 fn(x, "ger",                KFN(Ger),                1);
 fn(x, "gt",                 KFN(GT),                 1);
 fn(x, "hann_window",        KFN(Hann_window),        1);
 fn(x, "hamming_window",     KFN(Hamming_window),     1);
 fn(x, "histc",              KFN(Histc),              1);
 fn(x, "inverse",            KFN(Inverse),            1);
 fn(x, "ifft",               KFN(Ifft),               1);
 fn(x, "irfft",              KFN(Irfft),              1);
 fn(x, "isfinite",           KFN(Isfinite),           1);
 fn(x, "isinf",              KFN(Isinf),              1);
 fn(x, "isnan",              KFN(Isnan),              1);
 fn(x, "kthvalue",           KFN(Kthvalue),           1);
 fn(x, "le",                 KFN(Le),                 1);
 fn(x, "lerp",               KFN(Lerp),               1);
 fn(x, "Log",                KFN(Log),                1);
 fn(x, "log10",              KFN(Log10),              1);
 fn(x, "log1p",              KFN(Log1p),              1);
 fn(x, "log2",               KFN(Log2),               1);
 fn(x, "logdet",             KFN(Logdet),             1);
 fn(x, "logsumexp",          KFN(Logsumexp),          1);
 fn(x, "lt",                 KFN(Lt),                 1);
 fn(x, "lu",                 KFN(Lu),                 1);
 fn(x, "lu_info",            KFN(Lu_info),            1);
 fn(x, "lu_solve",           KFN(Lu_solve),           1);
 fn(x, "lu_unpack",          KFN(Lu_unpack),          1);
 fn(x, "matmul",             KFN(Matmul),             1);
 fn(x, "matrix_power",       KFN(Matrix_power),       1);
 fn(x, "matrix_rank",        KFN(Matrix_rank),        1);
 fn(x, "Max",                KFN(Max),                1);
 fn(x, "mean",               KFN(Mean),               1);
 fn(x, "median",             KFN(Median),             1);
 fn(x, "Min",                KFN(Min),                1);
 fn(x, "max_values",         KFN(Max_values),         1);
 fn(x, "min_values",         KFN(Min_values),         1);
 fn(x, "mm",                 KFN(Mm),                 1);
 fn(x, "mmt",                KFN(Mmt),                1);
 fn(x, "mode",               KFN(Mode),               1);
 fn(x, "mtm",                KFN(Mtm),                1);
 fn(x, "mul",                KFN(Mul),                1);
 fn(x, "mv",                 KFN(Mv),                 1);
 fn(x, "mvlgamma",           KFN(Mvlgamma),           1);
 fn(x, "ne",                 KFN(Ne),                 1);
 fn(x, "Neg",                KFN(Neg),                1);
 fn(x, "nnorm",              KFN(Nnorm),              1);
 fn(x, "orgqr",              KFN(Orgqr),              1);
 fn(x, "ormqr",              KFN(Ormqr),              1);
 fn(x, "pinverse",           KFN(Pinverse),           1);
 fn(x, "pnorm",              KFN(Pnorm),              1);
 fn(x, "pow",                KFN(Pow),                1);
 fn(x, "prod",               KFN(Prod),               1);
 fn(x, "qr",                 KFN(Qr),                 1);
 fn(x, "Reciprocal",         KFN(Reciprocal),         1);
 fn(x, "remainder",          KFN(Remainder),          1);
 fn(x, "roll",               KFN(Roll),               1);
 fn(x, "renorm",             KFN(Renorm),             1);
 fn(x, "rfft",               KFN(Rfft),               1);
 fn(x, "round",              KFN(Round),              1);
 fn(x, "rsqrt",              KFN(Rsqrt),              1);
 fn(x, "sigmoid",            KFN(Ksigmoid),           1);
 fn(x, "sign",               KFN(Sign),               1);
 fn(x, "Sin",                KFN(Sin),                1);
 fn(x, "sinh",               KFN(Sinh),               1);
 fn(x, "slogdet",            KFN(Slogdet),            1);
 fn(x, "solve",              KFN(Solve),              1);
 fn(x, "sort",               KFN(Sort),               1);
 fn(x, "Sqrt",               KFN(Sqrt),               1);
 fn(x, "std",                KFN(Std),                1);
 fn(x, "Sum",                KFN(Sum),                1);
 fn(x, "svd",                KFN(Svd),                1);
 fn(x, "symeig",             KFN(Symeig),             1);
 fn(x, "Tan",                KFN(Tan),                1);
 fn(x, "tanh",               KFN(Ktanh),              1);
 fn(x, "tensordot",          KFN(Tensordot),          1);
 fn(x, "topk",               KFN(Topk),               1);
 fn(x, "trace",              KFN(Trace),              1);
 fn(x, "tril",               KFN(Tril),               1);
 fn(x, "triu",               KFN(Triu),               1);
 fn(x, "triangular_solve",   KFN(Triangular_solve),   1);
 fn(x, "trunc",              KFN(Trunc),              1);
 fn(x, "unique",             KFN(Unique),             1);
 fn(x, "uniquec",            KFN(Uniquec),            1);
 fn(x, "Var",                KFN(Var),                1);
}
