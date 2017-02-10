/* 
 * util.h
 */

#pragma once
#include <x86intrin.h>

typedef long long tsc_t;

tsc_t get_tsc() {
  return _rdtsc();
}

tsc_t diff_tsc(tsc_t c1, tsc_t c0) {
  /* want to know what the heck is this *(2.7/2.3)?
     => see the exercise page */
  return (c1 - c0) * (2.7/2.3);
}
