/* 
 * data.h
 */
#pragma once
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include "mat.h"

/* map filename on memory and return a matrix in the file */
template<int M,int N>
static mat<M,N> map_npy_file(const char * filename) {
  /* I don't know its exact format, but the header seems 80 bytes */
  const off_t header_sz = 80;
  const off_t expected_sz = sizeof(real) * M * N + header_sz;
  (void)expected_sz;
  int fd = open(filename, O_RDWR);
  if (fd == -1) {
    int e = errno;
    fprintf(stderr, "open: %s (%s)\n", strerror(e), filename);
    exit(1);
  }
  off_t sz = lseek(fd, 0, SEEK_END);
  if (sz == -1) {
    perror("lseek"); exit(1);
  }
  if (lseek(fd, 0, SEEK_SET) == -1) {
    perror("lseek"); exit(1);
  }
  assert(sz == expected_sz);
  char * p = (char *)mmap(0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  assert(strncmp(p + 1, "NUMPY", 5) == 0);
  assert(p[header_sz - 1] == '\n');
  mat<M,N> a((real *)(p + header_sz), M);
  return a;
}

/* generate a random number from a normal distribution whose 
   mean is mu and variance is sigma^2. used to initialize
   the weight matrices. see
   https://en.wikipedia.org/wiki/Normal_distribution
   for how the following method works */
static real gen_normal(unsigned short rg[3], real mu, real sigma) {
  real u = erand48(rg);
  real v = erand48(rg);
  real x = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
  return mu + x * sigma;
}

/* initialize a matrix a with normal distribution */
template<int M, int N>
static void init_normal(unsigned short rg[3], mat<M,N> a) {
  real q = sqrt(1.0/N);
  for (long i = 0; i < a.m; i++) {
    for (long j = 0; j < N; j++) {
      a(i,j) = gen_normal(rg, 0.0, q);
    }
  }
}

/* choose designated rows from a */
template<int B, int N, int M>
  mat<B,N> get_rows(mat<M,N> a, long row_idxs[B], long n_rows) {
  mat<B,N> b(n_rows);
  for (long i = 0; i < n_rows; i++) {
    long idx = row_idxs[i];
    for (long j = 0; j < N; j++) {
      b(i,j) = a(idx,j);
    }
  }
  return b;
}

/* choose n_idxs numbers from 0..M-1 */
template<int M, int B>
void choose_random_samples(unsigned short rg[3], long idxs[B], long n_idxs) {
  assert(n_idxs <= B);
  for (long i = 0; i < n_idxs; i++) {
    idxs[i] = nrand48(rg) % M;
  }
}

/* generate n_idxs sequential indexes from begin, except they
   wrap around at M */
template<int M, int B>
void get_seq_samples(long begin, long idxs[B], long n_idxs) {
  assert(n_idxs <= B);
  for (long i = 0; i < n_idxs; i++) {
    idxs[i] = (begin + i) % M;
  }
}

void dump_as_ppm(real * a, int m, int n, int magnify, const char * filename) {
  FILE * wp = fopen(filename, "wb");
  if (!wp) {
    int e = errno;
    fprintf(stderr, "fopen: failed to create %s (%s). make sure the parent directory exists\n",
	    filename, strerror(e));
    exit(1);
  }
  int b = magnify;
  fprintf(wp, "P3 %d %d 255\n", m * b, n * b);
  for (long i = 0; i < m; i++) {
    for (long _ = 0; _ < b; _++) {
      for (long j = 0; j < n; j++) {
	for (long __ = 0; __ < b; __++) {
	  int v = a[i * n + j] * 255.0 + 0.1;
	  if (v >= 0) {
	    fprintf(wp, "%d %d %d\n", v, v, v);
	  } else {
	    fprintf(wp, "%d %d %d\n", -v, 0, 0);
	  }
	}
      }
    }
  }
  fclose(wp);
}
