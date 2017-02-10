/* 
 * mem.h --- memory management
 */
#pragma once
#include <stdlib.h>

/* allocate sz bytes at an address aligned to 64 */
static inline void * alloc64(size_t sz) {
  void * a = 0;
  if (posix_memalign(&a, 64, sz) != 0) {
    perror("posix_memalign");
    exit(1);
  }
  return a;
}

/* memory manager used by matrix library.
   you probably do not have to look inside it.
   the basic usage:
   
   mem_manager mm;
   mm.init(L);
   // call mm.alloc(sz) as you want, up to L times;
   mm.alloc(sz);
   mm.alloc(sz');
      ...
   // free all memory allocated, except for addrs[0]
   // addrs[1], ..., addrs[n-1]
   mm.free_except(addrs, n);
       
   (1) void * a = mm.alloc(sz):
       allocate sz bytes with 64 bytes alignment
   (2) mm.free_except(addrs, n) 
       free all memory allocated so far, except
       addrs[0], addrs[1], ..., addrs[n-1]

    matrix library (mat.h) uses this memory allocator
    to allocate memory for intermediate matrices
 */

struct mem_manager {
  void ** used;			/* an array of addresses in use */
  long n_used;			/* number of addresses in use */
  long region_capacity;		/* capacity of used array */

  /* initialize the memory manager, so that it can 
     have up to CAP addresses in use */
  void init(long cap) {
    n_used = 0;
    used = (void **)alloc64(sizeof(void *) * cap);
    region_capacity = cap;
  }
  /* free all resources */
  void fini() {
    n_used = 0;
    if (used) free(used);
    region_capacity = 0;
  }
  /* allocate sz bytes */
  void * alloc(size_t sz) {
    void * a = alloc64(sz);
    long n = __sync_fetch_and_add(&n_used, 1);
    if (n >= region_capacity) {
      fprintf(stderr,
	      "sorry, this region can allocate only up to %ld objects\n", region_capacity);
      fprintf(stderr,
	      "change the parameter to the mem_manager object to increase the limit\n");
      exit(1);
    }
    used[n] = a;
    return a;
  }
  /* free all addresses in use, except for a[0], a[1], ..., a[n-1] */
  void free_except(void ** a, long n) {
    long p = 0;
    for (long i = 0; i < n_used; i++) {
      void * u = used[i];
      for (long j = 0; j < n; j++) {
	if (a[j] == u) {
	  used[p++] = u;
	  break;
	}
      }
    }
    for (long j = p; j < n_used; j++) {
      free(used[j]);
    }
    n_used = p;
  }
  /* free all addresses except for a0, a1, ..., a4 */
  void free_except5(void * a0, void * a1, void * a2, void * a3, void * a4) {
    void * a[5] = { a0, a1, a2, a3, a4 };
    free_except(a, sizeof(a) / sizeof(a[0]));
  }
  /* free all addresses */
  void free_all() {
    free_except(0, 0);
  }
};

extern mem_manager mm;
