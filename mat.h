/*
 * mat.h
 */
// 0.15644995
#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "immintrin.h"

#include "mem.h"

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
template <int M, int N>
mat<M, N> operator+(mat<M, N> a, mat<M, N> b) {
  mat<M, N> c(a.m);
  assert(a.m == b.m);
  for (long i = 0; i < a.m * N; i += 8) {
    c.r(i) = a.r(i) + b.r(i);
  }
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
  for (long i = 0; i < a.m * N; i += 8) {
    c.r(i) = a.r(i) - b.r(i);
  }
  return c;
}

/* matrix -= matrix
   mat<3,4> a; mat<3,4> b;
   a -= b;
 */
template <int M, int N>
mat<M, N> operator-=(mat<M, N> a, mat<M, N> b) {
  assert(a.m == b.m);
  for (long i = 0; i < a.m * N; i += 8) {
    a.r(i) -= b.r(i);
  }
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
}

/* matrix * matrix (both are row-major)
   mat<3,4> a; mat<4,5> b;
   mat<3,5> c = a * b;
 */
template <int M, int N, int K>
mat<M, N> operator*(mat<M, K> a, mat<K, N> b) {
  mat<M, N> c(a.m);
  assert(K == b.m);
  assert(M % 8 == 0);
  assert(N % 8 == 0);
  assert(K % 8 == 0);
  assert(a.m % 8 == 0);
  c.zero();
  __m256 buff[8][8];
  long j_reduced = N - N % 64;
  asm volatile("# begin loop");
  for (long i = 0; i < a.m; i += 8) {
    for (long j = 0; j < j_reduced; j += 64) {
      for (int di = 0; di < 8; di++) {
        for (int dj = 0; dj < 8; dj++) {
          buff[di][dj] = c.v(i + di, j + dj * 8);
        }
      }

      for (long k = 0; k < K; k++) {
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

    for (long j = j_reduced; j < N; j += 8) {
      for (int di = 0; di < 8; di++) {
        buff[di][0] = c.v(i + di, j);
      }

      for (long k = 0; k < K; k++) {
        for (int di = 0; di < 8; di++) {
          buff[di][0] += a(i + di, k) * b.v(k, j);
        }
      }

      for (int di = 0; di < 8; di++) {
        c.v(i + di, j) = buff[di][0];
      }
    }
  }
  asm volatile("# begin end");
  return c;
}

/* row-major matrix * column-major matrix
   (return row-major matrix)
   mat<3,4> a; cmat<4,5> b;
   mat<3,5> c = a * b;
 */
// (B * A)' = A' * B'
template <int M, int N, int K>
mat<M, N> operator*(mat<M, K> a, cmat<K, N> b) {
  mat<M, N> c(a.m);
  assert(M % 8 == 0);
  assert(N % 8 == 0);
  assert(K % 8 == 0);
  assert(a.m % 8 == 0);
  c.zero();
  mat<N, K> bt = b.T();
  __m256 buff[8][8];
  long k_reduced = K - K % 64;
  // for (long i = 0; i < a.m; i++) {
  //   for (long j = 0; j < N; j++) {
  //     for (long k = 0; k < K; k++) {
  //       c(i, j) += a(i, k) * b(k, j);
  //     }
  //   }
  // }
  // return c;
  for (long i = 0; i < a.m; i++) {
    for (long j = 0; j < N; j++) {
      for (int dk = 0; dk < 8; dk++) {
        buff[0][dk] = _mm256_setzero_ps();
      }

      for (long k = 0; k < k_reduced; k += 64) {
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

      for (long k = k_reduced; k < K; k += 8) {
        buff[0][0] += a.v(i, k) * bt.v(j, k);
      }
      for (int di = 0; di < 8; di++) {
        c(i, j) += buff[0][0][di];
      }
    }
  }
  return c;
}

/* column-major matrix * row-major matrix
   (return row-major matrix)
   mat<3,4> a; cmat<4,5> b;
   mat<3,5> c = a * b;
 */
template <int M, int N, int K>
mat<M, N> operator*(cmat<M, K> a, mat<K, N> b) {
  mat<M, N> c(M);
  assert(M % 8 == 0);
  assert(N % 8 == 0);
  assert(K % 8 == 0);
  assert(a.n == b.m);
  c.zero();
  __m256 buff[8][8];
  long j_reduced = N - N % 64;
  // for (long i = 0; i < M; i++) {
  //   for (long j = 0; j < N; j++) {
  //     for (long k = 0; k < a.n; k++) {
  //       c(i, j) += at(k, i) * b(k, j);
  //     }
  //   }
  // }
  // return c;
  for (long i = 0; i < M; i += 8) {
    for (long j = 0; j < j_reduced; j += 64) {
      for (int di = 0; di < 8; di++) {
        for (int dj = 0; dj < 8; dj++) {
          buff[di][dj] = c.v(i + di, j + dj * 8);
        }
      }

      for (long k = 0; k < a.n; k++) {
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

    for (long j = j_reduced; j < N; j += 8) {
      for (int di = 0; di < 8; di++) {
        buff[di][0] = c.v(i + di, j);
      }

      for (long k = 0; k < a.n; k++) {
        for (int di = 0; di < 8; di++) {
          buff[di][0] += a(i + di, k) * b.v(k, j);
        }
      }

      for (int di = 0; di < 8; di++) {
        c.v(i + di, j) = buff[di][0];
      }
    }
  }
  return c;
}
