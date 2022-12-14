CXX=nvcc
LD=${CXX}
CXXFLAGS+=-Xcompiler="-Wall -Wextra -Werror" -std c++11 -O3
LDFLAGS+=-lm $(CXXFLAGS) -lcublas -lopenblas

OBJS=cg_main.o cg.o matrix.o matrix_coo.o mmio.o conjugate_gradient.o

all: cgsolver

cgsolver: $(OBJS) 
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

conjugate_gradient.o: conjugate_gradient.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f cgsolver *.o *~