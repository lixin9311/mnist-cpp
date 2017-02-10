
# number of units in the hidden layer (change it as you like)
n_units:=128
# maximum mini batch size (you can change the actual size at runtime)
max_batch_sz:=1000

opts := 
# opts += -O0 -g
opts += -O3
opts += -DNDEBUG
opts += -march=native
opts += -Wall
opts += -Wextra
opts += -fopt-info-vec-optimized
opts += -Dn_units=$(n_units)
opts += -Dmax_batch_sz=$(max_batch_sz)

debug := 
debug += -O0 -g
# debug += -O3
# debug += -DNDEBUG
debug += -march=native
debug += -Wall
debug += -Wextra
debug += -fopt-info-vec-optimized
debug += -Dn_units=$(n_units)
debug += -Dmax_batch_sz=$(max_batch_sz)

CXXFLAGS := $(opts)
DEBUGFLAGS := $(debug)
mnist_$(n_units)_$(max_batch_sz) : mnist.cc data.h functions.h  mat.h  mem.h  score.h  util.h
	g++ -o $@ $(CXXFLAGS) $<

asm : mnist.cc data.h functions.h mat.h  mem.h  score.h  util.h
	g++ -S -o $@ $(CXXFLAGS) $<

debug : mnist.cc data.h functions.h mat.h  mem.h  score.h  util.h
	g++ -o $@ $(DEBUGFLAGS) $<

clean :
	rm -i -f mnist_*_* debug asm
