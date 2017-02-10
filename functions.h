/* 
 * functions.h
 */
#pragma once
#include <math.h>

#include "mat.h"

/* y = x (if x > 0) or 0 otherwise */
template<int M, int N>
mat<M,N> relu2(mat<M,N> x, mat<M,N> c) {
  mat<M,N> y(x.m);
  for (long i = 0; i < x.m; i++) {
    for (long j = 0; j < N; j++) {
      y(i,j) = (c(i,j) > 0 ? x(i,j) : 0);
    }
  }
  return y;
}

template<int M, int N>
mat<M,N> logsoftmax(mat<M,N> x) {
  mat<M,N> lsm(x.m);
  for (long i = 0; i < x.m; i++) {
    long m = 0;
    for (long j = 0; j < N; j++) {
      m = (x(i,m) < x(i,j) ? j : m);
    }
    real s = 0.0;
    for (long j = 0; j < N; j++) {
      lsm(i,j) = x(i,j) - x(i,m);
      s += exp(lsm(i,j));
    }
    for (long j = 0; j < N; j++) {
      lsm(i,j) -= log(s);
    }
  }
  return lsm;
}

template<int M, int N>
mat<M,1> softmax_cross_entropy(mat<M,N> x, mat<M,1> c) {
  mat<M,N> lsm = logsoftmax(x);
  mat<M,1> smxe(lsm.m);
  for (long i = 0; i < lsm.m; i++) {
    smxe(i,0) = -lsm(i,c(i,0));
  }
  return smxe;
}

template<int M, int N>
mat<M,N> softmax(mat<M,N> x) {
  mat<M,N> y = logsoftmax(x);
  for (long i = 0; i < y.m; i++) {
    for (long j = 0; j < N; j++) {
      y(i,j) = exp(y(i,j));
    }
  }
  return y;
}

template<int M, int N>
mat<M,N> softmax_minus_one(mat<M,N> x, mat<M,1> c) {
  mat<M,N> y = softmax(x);
  for (long i = 0; i < y.m; i++) {
    y(i,c(i,0)) -= 1.0;
  }
  return y;
}

template<int M, int N>
mat<M,1> argmax(mat<M,N> a) {
  mat<M,1> am(a.m);
  for (long i = 0; i < a.m; i++) {
    long m = 0;
    for (long j = 0; j < N; j++) {
      if (a(i,m) < a(i,j)) m = j;
    }
    am(i,0) = m;
  }
  return am;
}

