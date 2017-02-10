/* 
 * score_counter.h
 */

#pragma once
#include <assert.h>
#include <stdlib.h>

#include "mat.h"

struct score {
  real e;			// error value
  bool c;			// classification (0 : NG, 1 : OK)
};

struct score_counter {
  score * R;
  long capacity;
  long p;
  long n;
  long sum_c;
  double sum_e;
  score_counter(long capacity_) {
    capacity = capacity_;
    p = 0;
    n = 0;
    R = (score *)alloc64(sizeof(score) * capacity);
    sum_c = 0;
    sum_e = 0.0;
  }
  void fini() {
    free(R);
  }
  void add(bool c, real e) {
    if (p == n) {
      n++;
    } else {
      assert(n == capacity);
      assert(R[p].c == 0 || R[p].c == 1);
      sum_c -= R[p].c;
      sum_e -= R[p].e;
    }
    R[p].c = c;
    R[p].e = e;
    sum_c += c;
    sum_e += e;
    p++;
    if (p == capacity) p = 0;
  }
  template<int M>
  void update_score(mat<M,1> y, mat<M,1> c, mat<M,1> e) {
    assert(y.m == c.m);
    assert(y.m == e.m);
    assert(y.m <= M);
    for (long i = 0; i < y.m; i++) {
      bool classify = (y(i,0) == c(i,0));
      add(classify, e(i, 0));
    }
  }
};

