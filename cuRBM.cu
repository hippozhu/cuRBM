#include "cuRBM.h"

__constant__ unsigned nVis;
__constant__ unsigned nHid;
__constant__ unsigned batch;
__constant__ unsigned miniBatch;
__constant__ int *data;
__constant__ weight_t *weight;
__constant__ size_t pitch_data;
__constant__ size_t pitch_weight;

void cudaErrorCheck(cudaError_t error){
  if(error != cudaSuccess){
	cout << "CUDA error: " << cudaGetErrorString(error) << endl;
	exit(-1);
   }
}

void deviceInit(unsigned nvisible, unsigned nhidden){
  cudaErrorCheck(cudaMemcpyToSymbol(miniBatch, &h_miniBatch, sizeof(unsigned), 0, cudaMemcpyHostToDevice));

  cudaErrorCheck(cudaMemcpyToSymbol(nVis, &nvisible, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpyToSymbol(nHid, &nhidden, sizeof(unsigned), 0, cudaMemcpyHostToDevice));

  cudaErrorCheck(cudaMallocPitch((void **)&d_data, &d_pitch, len * sizeof(int), h_batch));
  cudaErrorCheck(cudaMemcpyToSymbol(data, &d_data, sizeof(int *), 0, cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpyToSymbol(pitch_data, &d_pitch, sizeof(size_t), 0, cudaMemcpyHostToDevice));

  cudaErrorCheck(cudaMalloc((void **)&d_data, len * sizeof(int)));
}

void batchTransfer(int *data, unsigned nCase){
  // Copy data to device coalesced
  cudaErrorCheck(cudaMemcpy2D(d_data, d_pitch, data, h_pitch, width, nCase, cudaMemcpyHostToDevice));
}
