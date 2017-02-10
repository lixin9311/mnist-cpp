// 0.15812783
#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tbb/task_group.h"
// #include <thread>
#include "immintrin.h"

#include "mem.h"

#define CACHE_SIZE 64000
#ifndef V1
#define V1 64
#endif
#ifndef V2
#define V2 64
#endif
#define V3 128
#define NUM_OF_CORES 16
/* this file defines matrix classes.

   it currently implements the most straightforward
   matrix multiply. you are supposed to work mainly
   on this file (not to say you shouldn't change other
   files).
*/

typedef float real;
typedef float float8 __attribute__((vector_size(32)));

/* row-major matrix
   (elements in a row are consecutive in memory) */
template <int M, int N>
struct mat;

/* column-major matrix
   (elements in a column are consecutive in memory) */
template <int M, int N>
struct cmat;

template <int M, int N>
struct mat {
  long m; /* the actual number of rows */
  real* a;
  mat(real* a_, long m_) {
    assert(M % 8 == 0 || M == 1);
    assert(N % 8 == 0 || N == 1);
    assert(m_ % 8 == 0 || m_ == 1);
    m = m_;
    a = a_;
  }
  mat(long m_) {
    assert(M % 8 == 0 || M == 1);
    assert(N % 8 == 0 || N == 1);
    assert(m_ % 8 == 0 || m_ == 1);
    m = m_;
    a = (real*)mm.alloc(sizeof(real) * m * N);
  }
  /* transpose (you get a column-major matrix) */
  cmat<N, M> T() {
    assert(m <= M);
    cmat<N, M> b(a, m);
    return b;
  }
  /* a(i,j) */
  real& operator()(long i, long j) {
    assert(i < m);
    assert(j < N);
    return a[i * N + j];
  }

  float8& r(long i) {
    assert(i <= M * N - 8);
    return *((float8*)&a[i]);
  }

  float8& v(long i, long j) {
    assert(i < m);
    assert(j <= N - 8);
    return *((float8*)&a[i * N + j]);
  }
  void zero() { memset(a, 0, sizeof(real) * m * N); }
};

template <int M, int N>
struct cmat {
  long n;
  real* a;
  cmat(real* a_, long n_) {
    n = n_;
    a = a_;
  }
  cmat(long n_) {
    n = n_;
    a = (real*)mm.alloc(sizeof(real) * M * n);
  }
  /* transpose (you get a row-major matrix) */
  mat<N, M> T() {
    assert(n <= N);
    mat<N, M> b(a, n);
    return b;
  }
  mat<M, N> ToMat() {
    mat<M, N> b(M);
    for (long i = 0; i < M; i++) {
      for (long j = 0; j < N; j++) {
        b(i, j) = (*this)(i, j);
      }
    }
    return b;
  }
  /* a(i,j) */
  real& operator()(long i, long j) {
    assert(i < M);
    assert(j < N);
    return a[i + j * M];
  }
  void zero() { memset(a, 0, sizeof(real) * M * n); }
};

/* matrix + matrix
   mat<3,4> a; mat<3,4> b;  // for (long i = 0; i < a.m; i++) {
  //   for (long j = 0; j < N; j++) {
  //     for (long k = 0; k < K; k++) {
  //       c(i, j) += a(i, k) * b(k, j);
  //     }
  //   }
  // }
  // return c;
   mat<3,4> c = a + b;
 */
// 0.19595770
// 9402
// 0.19636775
// 0.19600458
template <int M, int N>
mat<M, N> operator+(mat<M, N> a, mat<M, N> b) {
  mat<M, N> c(a.m);
  assert(a.m == b.m);
  tbb::task_group tg;
  long split = a.m * N / 8 / NUM_OF_CORES;
  // long split = a.m * N / 8 / 2;
  // tbb::task_group tg;
  // tg.run([&] {
  //   for (long i = 0; i < split; i += 8) {
  //     c.r(i) = a.r(i) + b.r(i);
  //   }
  // });
  // for (long i = split; i < a.m * N; i += 8) {
  //   c.r(i) = a.r(i) + b.r(i);
  // }
  // tg.wait();
  // for (long j = 0; j < NUM_OF_CORES - 1; j++) {
  //   tg.run([&] {
  //     for (long i = split * j; i < split * (j + 1); i += 8) {
  //       c.r(i) = a.r(i) + b.r(i);
  //     }
  //   });
  // }
  split = 0;
  for (long i = split * (NUM_OF_CORES - 1); i < a.m * N; i += 8) {
    c.r(i) = a.r(i) + b.r(i);
  }
  tg.wait();
  return c;
}

/* matrix - matrix
   mat<3,4> a; mat<3,4> b;
   mat<3,4> c = a - b;
 */
template <int M, int N>
mat<M, N> operator-(mat<M, N> a, mat<M, N> b) {
  mat<M, N> c(a.m);
  assert(a.m == b.m);
  // long split = a.m * N / 8 / NUM_OF_CORES;
  // tbb::task_group tg;
  // for (long j = 0; j < NUM_OF_CORES - 1; j++) {
  //   tg.run([&] {
  //     for (long i = split * j; i < split * (j + 1); i += 8) {
  //       c.r(i) = a.r(i) - b.r(i);
  //     }
  //   });
  // }
  long split = 0;
  for (long i = split * (NUM_OF_CORES - 1); i < a.m * N; i += 8) {
    c.r(i) = a.r(i) - b.r(i);
  }
  // tg.wait();
  return c;
}

/* matrix -= matrix
   mat<3,4> a; mat<3,4> b;
   a -= b;
 */
// 0.20679681
template <int M, int N>
static inline void minusMatMat(mat<M, N> a, mat<M, N> b, long i0, long i1) {
  for (long i = i0; i < i1; i += 8) {
    a.r(i) -= b.r(i);
  }
  return;
}

template <int M, int N>
mat<M, N> operator-=(mat<M, N> a, mat<M, N> b) {
  assert(a.m == b.m);
  long split = a.m * N / 8 / NUM_OF_CORES;
  tbb::task_group tg;
  tg.run([&] { minusMatMat(a, b, split * 0, split * 1); });
  tg.run([&] { minusMatMat(a, b, split * 1, split * 2); });
  tg.run([&] { minusMatMat(a, b, split * 2, split * 3); });
  tg.run([&] { minusMatMat(a, b, split * 3, split * 4); });
  tg.run([&] { minusMatMat(a, b, split * 4, split * 5); });
  tg.run([&] { minusMatMat(a, b, split * 5, split * 6); });
  tg.run([&] { minusMatMat(a, b, split * 6, split * 7); });

  tg.run([&] { minusMatMat(a, b, split * 7, split * 8); });
  tg.run([&] { minusMatMat(a, b, split * 8, split * 9); });
  tg.run([&] { minusMatMat(a, b, split * 9, split * 10); });
  tg.run([&] { minusMatMat(a, b, split * 10, split * 11); });
  tg.run([&] { minusMatMat(a, b, split * 11, split * 12); });
  tg.run([&] { minusMatMat(a, b, split * 12, split * 13); });
  tg.run([&] { minusMatMat(a, b, split * 13, split * 14); });
  tg.run([&] { minusMatMat(a, b, split * 14, split * 15); });

  for (long i = split * (NUM_OF_CORES - 1); i < a.m * N; i += 8) {
    a.r(i) -= b.r(i);
  }
  tg.wait();
  return a;
}

/* scalar * matrix
   mat<3,4> a;
   mat<3,4> b = 5.6 * a;
 */
template <int M, int N>
mat<M, N> operator*(real k, mat<M, N> a) {
  mat<M, N> b(a.m);
  for (long i = 0; i < a.m * N; i += 8) {
    b.r(i) = k * a.r(i);
  }
  return b;

  long split = a.m * N / 8 / NUM_OF_CORES;
  tbb::task_group tg;
  for (long j = 0; j < NUM_OF_CORES - 1; j++) {
    tg.run([&] {
      for (long i = split * j; i < split * (j + 1); i += 8) {
        b.r(i) = k * a.r(i);
      }
    });
  }
  for (long i = split * (NUM_OF_CORES - 1); i < a.m * N; i += 8) {
    b.r(i) = k * a.r(i);
  }
  tg.wait();
}

/* matrix * matrix (both are row-major)
   mat<3,4> a; mat<4,5> b;
   mat<3,5> c = a * b;
 */

#define DATA_SIZE i_size* k_size + k_size* j_size + i_size* j_size
template <int M, int N, int K>
static inline void mulMatMat(mat<M, K> a, mat<K, N> b, mat<M, N> c, long i0,
                             long i1, long j0, long j1, long k0, long k1) {
  long i_size = i1 - i0;
  long j_size = j1 - j0;
  long k_size = k1 - k0;

  if (i_size > V1) {
    long i_split = i0 + i_size / 8 / 2 * 8;
    tbb::task_group tg;
    tg.run([&] { mulMatMat(a, b, c, i0, i_split, j0, j1, k0, k1); });
    mulMatMat(a, b, c, i_split, i1, j0, j1, k0, k1);
    tg.wait();
  } else if (j_size > V3) {
    long j_split = j0 + j_size / 64 / 2 * 64;
    tbb::task_group tg;
    tg.run([&] { mulMatMat(a, b, c, i0, i1, j0, j_split, k0, k1); });
    mulMatMat(a, b, c, i0, i1, j_split, j1, k0, k1);
    tg.wait();
  } else if (k_size > V2) {
    long k_split = k0 + k_size / 8 / 2 * 8;
    mulMatMat(a, b, c, i0, i1, j0, j1, k0, k_split);
    mulMatMat(a, b, c, i0, i1, j0, j1, k_split, k1);
  } else {
    long j_reduced = j1 - (j1 - j0) % 64;
    __m256 buff[8][8];
    for (long i = i0; i < i1; i += 8) {
      for (long j = j0; j < j_reduced; j += 64) {
        for (int di = 0; di < 8; di++) {
          for (int dj = 0; dj < 8; dj++) {
            buff[di][dj] = c.v(i + di, j + dj * 8);
          }
        }

        for (long k = k0; k < k1; k++) {
          for (int di = 0; di < 8; di++) {
            for (int dj = 0; dj < 8; dj++) {
              buff[di][dj] += a(i + di, k) * b.v(k, j + dj * 8);
            }
          }
        }

        for (int di = 0; di < 8; di++) {
          for (int dj = 0; dj < 8; dj++) {
            c.v(i + di, j + dj * 8) = buff[di][dj];
          }
        }
      }

      for (long j = j_reduced; j < j1; j += 8) {
        for (int di = 0; di < 8; di++) {
          buff[di][0] = c.v(i + di, j);
        }

        for (long k = k0; k < k1; k++) {
          for (int di = 0; di < 8; di++) {
            buff[di][0] += a(i + di, k) * b.v(k, j);
          }
        }

        for (int di = 0; di < 8; di++) {
          c.v(i + di, j) = buff[di][0];
        }
      }
    }
  }
}

template <int M, int N, int K>
mat<M, N> operator*(mat<M, K> a, mat<K, N> b) {
  mat<M, N> c(a.m);
  assert(K == b.m);
  assert(M % 8 == 0);
  assert(N % 8 == 0);
  assert(K % 8 == 0);
  assert(a.m % 8 == 0);
  c.zero();
  asm volatile("# begin loop");
  mulMatMat(a, b, c, 0, a.m, 0, N, 0, K);
  asm volatile("# begin end");
  return c;
}

template <int M, int N, int K>
static inline void mulMatCMat(mat<M, K> a, mat<N, K> bt, mat<M, N> c, long i0,
                              long i1, long j0, long j1, long k0, long k1) {
  long i_size = i1 - i0;
  long j_size = j1 - j0;
  long k_size = k1 - k0;

  if (i_size > V1) {
    long i_split = i0 + i_size / 8 / 2 * 8;
    tbb::task_group tg;
    tg.run([&] { mulMatCMat(a, bt, c, i0, i_split, j0, j1, k0, k1); });
    mulMatCMat(a, bt, c, i_split, i1, j0, j1, k0, k1);
    tg.wait();
  } else if (j_size > V1) {
    long j_split = j0 + j_size / 8 / 2 * 8;
    tbb::task_group tg;
    tg.run([&] { mulMatCMat(a, bt, c, i0, i1, j0, j_split, k0, k1); });
    mulMatCMat(a, bt, c, i0, i1, j_split, j1, k0, k1);
    tg.wait();
  } else if (k_size > V3) {
    long k_split = k0 + k_size / 64 / 2 * 64;
    mulMatCMat(a, bt, c, i0, i1, j0, j1, k0, k_split);
    mulMatCMat(a, bt, c, i0, i1, j0, j1, k_split, k1);
    // }
  } else {
    __m256 buff[8][8];
    long k_reduced = k1 - (k1 - k0) % 64;
    for (long i = i0; i < i1; i++) {
      for (long j = j0; j < j1; j++) {
        for (int dk = 0; dk < 8; dk++) {
          buff[0][dk] = _mm256_setzero_ps();
        }

        for (long k = k0; k < k_reduced; k += 64) {
          for (int dk = 0; dk < 8; dk++) {
            buff[0][dk] += a.v(i, k + dk * 8) * bt.v(j, k + dk * 8);
          }
        }

        buff[0][0] += buff[0][1];
        buff[0][2] += buff[0][3];
        buff[0][4] += buff[0][5];
        buff[0][6] += buff[0][7];
        buff[0][0] += buff[0][2];
        buff[0][6] += buff[0][4];
        buff[0][0] += buff[0][6];

        for (long k = k_reduced; k < k1; k += 8) {
          buff[0][0] += a.v(i, k) * bt.v(j, k);
        }
        for (int di = 0; di < 8; di++) {
          c(i, j) += buff[0][0][di];
        }
      }
    }
  }
}
/* row-major matrix * column-major matrix
   (return row-major matrix)
   mat<3,4> a; cmat<4,5> b;
   mat<3,5> c = a * b;
 */
// (B * A)' = A' * B'
// 0.15936706
template <int M, int N, int K>
mat<M, N> operator*(mat<M, K> a, cmat<K, N> b) {
  return a * b.ToMat();
  mat<M, N> c(a.m);
  assert(M % 8 == 0);
  assert(N % 8 == 0);
  assert(K % 8 == 0);
  assert(a.m % 8 == 0);
  c.zero();
  // for (long i = 0; i < a.m; i++) {
  //   for (long j = 0; j < N; j++) {
  //     for (long k = 0; k < K; k++) {
  //       c(i, j) += a(i, k) * b(k, j);
  //     }
  //   }
  // }
  // return c;
  mat<N, K> bt = b.T();
  mulMatCMat(a, bt, c, 0, a.m, 0, N, 0, K);
  return c;
}

/* column-major matrix * row-major matrix
   (return row-major matrix)
   mat<3,4> a; cmat<4,5> b;
   mat<3,5> c = a * b;
 */
template <int M, int N, int K>
static inline void mulCMatMat(cmat<M, K> a, mat<K, N> b, mat<M, N> c, long i0,
                              long i1, long j0, long j1, long k0, long k1) {
  long i_size = i1 - i0;
  long j_size = j1 - j0;
  long k_size = k1 - k0;

  if (i_size > V1) {
    long i_split = i0 + i_size / 8 / 2 * 8;
    tbb::task_group tg;
    tg.run([&] { mulCMatMat(a, b, c, i0, i_split, j0, j1, k0, k1); });
    mulCMatMat(a, b, c, i_split, i1, j0, j1, k0, k1);
    tg.wait();
  } else if (j_size > V3) {
    long j_split = j0 + j_size / 64 / 2 * 64;
    tbb::task_group tg;
    tg.run([&] { mulCMatMat(a, b, c, i0, i1, j0, j_split, k0, k1); });
    mulCMatMat(a, b, c, i0, i1, j_split, j1, k0, k1);
    tg.wait();
  } else if (k_size > V2) {
    long k_split = k0 + k_size / 8 / 2 * 8;
    mulCMatMat(a, b, c, i0, i1, j0, j1, k0, k_split);
    mulCMatMat(a, b, c, i0, i1, j0, j1, k_split, k1);
  } else {
    long j_reduced = j1 - (j1 - j0) % 64;
    __m256 buff[8][8];
    for (long i = i0; i < i1; i += 8) {
      for (long j = j0; j < j_reduced; j += 64) {
        for (int di = 0; di < 8; di++) {
          for (int dj = 0; dj < 8; dj++) {
            buff[di][dj] = c.v(i + di, j + dj * 8);
          }
        }

        for (long k = k0; k < k1; k++) {
          for (int di = 0; di < 8; di++) {
            for (int dj = 0; dj < 8; dj++) {
              buff[di][dj] += a(i + di, k) * b.v(k, j + dj * 8);
            }
          }
        }

        for (int di = 0; di < 8; di++) {
          for (int dj = 0; dj < 8; dj++) {
            c.v(i + di, j + dj * 8) = buff[di][dj];
          }
        }
      }

      for (long j = j_reduced; j < j1; j += 8) {
        for (int di = 0; di < 8; di++) {
          buff[di][0] = c.v(i + di, j);
        }

        for (long k = k0; k < k1; k++) {
          for (int di = 0; di < 8; di++) {
            buff[di][0] += a(i + di, k) * b.v(k, j);
          }
        }

        for (int di = 0; di < 8; di++) {
          c.v(i + di, j) = buff[di][0];
        }
      }
    }
  }
}

template <int M, int N, int K>
mat<M, N> operator*(cmat<M, K> a, mat<K, N> b) {
  mat<M, N> c(M);
  assert(M % 8 == 0);
  assert(N % 8 == 0);
  assert(K % 8 == 0);
  assert(a.n == b.m);
  c.zero();

  mulCMatMat(a, b, c, 0, M, 0, N, 0, a.n);
  return c;
}
