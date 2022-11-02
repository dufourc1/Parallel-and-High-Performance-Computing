CXX=mpiicpc # mpicxx
CC=mpiicpc #mpicxx
LD=${CXX}
CXXFLAGS+= -Wall -Wextra -Werror -pedantic -std=c++11

# Uncomment the line below is you use intel compiler: mpiicpc 
CXXFLAGS+=-no-multibyte-chars
# module load intel intel-mpi

# if using gcc compiler: mpicxx 
# comment above
# module load  gcc mvapich2

LDFLAGS+=-lm $(CXXFLAGS)

OBJS=poisson.o simulation.o double_buffer.o grid.o dumpers.o

all: poisson

poisson: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f hello poisson *.o *~
