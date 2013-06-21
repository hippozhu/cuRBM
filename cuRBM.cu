#include "cuRBM.h"
#define DEVMAIN

__constant__ int *data, *data_hid;
__constant__ float *data_vis_float, *data_hid_float;
__constant__ unsigned nVis, nHid, nCase, miniBatch, lenVis, lenHid;
__constant__ float *weight, *a, *b;
__constant__ size_t pitch_data, pitch_data_hid, pitch_weight;

__device__ float getData(float* base, int row, int col, size_t pitch){
  return *((float *)((char*)base + row * pitch) + col);
}

__device__ int getData(int* base, int row, int col, size_t pitch){
  return *((int *)((char*)base + row * pitch) + col);
}

__device__ void setData(float* base, int row, int col, size_t pitch, float v){
  *((float *)((char*)base + row * pitch) + col) = v;
}


__global__ void kernel1(){
  __shared__ int ds[32][8];
  __shared__ float ws[256];
  __shared__ float sum[256];
  __shared__ float result[32];

  int tid = threadIdx.x, stride;
  int nActive = miniBatch - blockIdx.x * 32 > 32?32:miniBatch - blockIdx.x * 32;

  int nIter = 0;
  // data prefetching
  /*
  int d;
  float w;
  if(nIter * 256 + tid < nVis)
    w = getData(weight, nIter * 256 + tid, blockIdx.y, pitch_weight);
  if(nIter * 8 + tid % 8 < lenVis && blockIdx.x * 32 + tid/8 < miniBatch)
    d = getData(data, blockIdx.x * 32 + tid/8, nIter * 8 + tid % 8, pitch_data);
  */
  // initialize result
  if(tid < 32)
    result[tid] = .0;


  for(; nIter < (nVis - 1)/256 + 1; ++ nIter){
    // copy data from register to shared memory
    if(nIter * 256 + tid < nVis)
      //ws[tid] = w;
      ws[tid] = getData(weight, nIter * 256 + tid, blockIdx.y, pitch_weight);
    else
      ws[tid] = .0;
    if(nIter * 8 + tid % 8 < lenVis && blockIdx.x * 32 + tid/8 < miniBatch)
      //ds[tid/8][tid%8] = d;
      ds[tid/8][tid%8] = getData(data, blockIdx.x * 32 + tid/8, nIter * 8 + tid % 8, pitch_data);
    else
      ds[tid/8][tid%8] = .0;

    __syncthreads();
    // prefetch next element
    /*
    if((nIter + 1) * 256 + tid < nVis)
      w = getData(weight, (nIter + 1) * 256 + tid, blockIdx.y, pitch_weight);
    if((nIter + 1) * 8 + tid % 8 < lenVis && blockIdx.x * 32 + tid/8 < miniBatch)
      d = getData(data, blockIdx.x * 32 + tid / 8, (nIter + 1) * 8 + tid % 8, pitch_data);
    */
    for(int i = 0; i < nActive; ++i){
      sum[tid] = .0;
      if(nIter * 256 + tid < nVis && (ds[i][tid/32] & (1<<(31-tid%32))))
        sum[tid] = ws[tid];
      stride = 128;
      while(stride > 0){
	__syncthreads();
        if(tid<stride)
	  sum[tid] += sum[tid + stride];
        stride /= 2;
      }
      __syncthreads();
      if(tid==0)
        result[i] += sum[0];
    }
  }

  __syncthreads();
  if(tid < nActive)
    setData(data_hid_float, 32 * blockIdx.x + tid, blockIdx.y, nHid * sizeof(float) , result[tid]);
}

void deviceInit();
void batchTransfer(unsigned start, unsigned batch_size);

void runRBM(){
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, NULL));
	
  deviceInit();
  for(unsigned i = 0; i < ninst; i += h_miniBatch){
    unsigned currentBatch = h_miniBatch > (ninst - i)? (ninst - i): h_miniBatch;
    batchTransfer(i, currentBatch);
    dim3 grid((ninst - 1)/32 + 1, nhidden);
    kernel1<<<grid, 256>>>();
    cudaThreadSynchronize();
    cudaError_t ret = cudaGetLastError();
    HANDLE_ERROR(ret);
    float *h_data_hid_float = (float *)malloc(ninst * nhidden * sizeof(float));
    HANDLE_ERROR(cudaMemcpy(h_data_hid_float, d_data_hid_float, h_miniBatch * nhidden * sizeof(float), cudaMemcpyDeviceToHost));
    cout << "result:"  << h_data_hid_float[0] << " " << h_data_hid_float[1] << " " << h_data_hid_float[nhidden];
    //printArray(h_data_hid_float, h_miniBatch, nhidden);
    free(h_data_hid_float);
    /*
    unsigned *d = (unsigned *)malloc(len * ninst * sizeof(unsigned));
    HANDLE_ERROR(cudaMemcpy2D(d, h_pitch_data, d_data, d_pitch_data, 
                                width, currentBatch, cudaMemcpyDeviceToHost));
    cout << *(h_data + i * len) << endl;
    cout << d[0] << endl;
    */
  }

        HANDLE_ERROR(cudaEventRecord(stop, NULL));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float msecTotal = 0.0f;
        HANDLE_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("\tcuRMB: %.2f msec\n", msecTotal);
}

