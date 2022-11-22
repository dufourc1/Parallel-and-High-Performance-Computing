CXX=nvcc
LD=${CXX}
CXXFLAGS+=-Xcompiler="-Wall -Wextra -Werror" -std c++11 -O3
LDFLAGS+=-lm $(CXXFLAGS)

OBJS=poisson.o simulation.o double_buffer.o dumpers.o

all: poisson_per_row poisson_per_entry

poisson_per_row: $(OBJS) simulation_gpu_row.o
	$(LD) -o $@ $(OBJS) simulation_gpu_row.o $(LDFLAGS)

poisson_per_entry: $(OBJS) simulation_gpu_entry.o
	$(LD) -o $@ $(OBJS) simulation_gpu_entry.o $(LDFLAGS)

simulation_gpu_row.o: simulation_gpu.cu
	$(CXX) $(CXXFLAGS) -DPER_ROW -c $< -o $@

simulation_gpu_entry.o: simulation_gpu.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f hello poisson *.o *~
	rm poisson_per_entry poisson_per_row