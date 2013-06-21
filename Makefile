all: cuRBM

#cuRBM: main.o kernels.o cublasRBM.o cudaMem.o
#	g++ -g main.o kernels.o cublasRBM.o cudaMem.o -o cuRBM -L/home/yzhu7/.local/cuda-5.0/lib64 -lcuda -lcudart -lcublas

cuRBM: main.o kernels.o cublasRBM.o cudaMem.o dlink.o
	g++ -g main.o kernels.o cublasRBM.o cudaMem.o dlink.o -o cuRBM -L/home/yzhu7/.local/cuda-5.0/lib64 -lcuda -lcudart -lcudadevrt -lcublas

main.o: cuRBM.cpp
	g++ -g -c cuRBM.cpp -o main.o

kernels.o: cuRBM.cu
	nvcc -g -G -dc -gencode arch=compute_20,code=sm_20 cuRBM.cu -o kernels.o

cublasRBM.o: cublasRBM.cu
	nvcc -g -G -dc -gencode arch=compute_20,code=sm_20 cublasRBM.cu -o cublasRBM.o

cudaMem.o: cudaMem.cu
	nvcc -g -G -dc -gencode arch=compute_20,code=sm_20 cudaMem.cu -o cudaMem.o

dlink.o:
	nvcc -g -G -dlink -gencode arch=compute_20,code=sm_20 kernels.o cublasRBM.o cudaMem.o -o dlink.o

clean:
	rm -rf *.o cuRBM
