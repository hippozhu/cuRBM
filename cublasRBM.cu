#include "cuRBM.h"
//#include "cuRBM.cuh"

extern float *d_weight, *d_a, *d_b;

void cublasRunRBM(){
  // data
  float *m_data = (float *)malloc(sizeof(float)*ninst*nvisible);
  arrayToMatrix(m_data);
  
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, NULL));
	
  float *d_data_a, *d_data_c;
  // allocate mini batch on device
  HANDLE_ERROR(cudaMalloc((void **)&d_data_a, h_miniBatch * nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&d_data_c, h_miniBatch * nhidden * sizeof(float)));
  
  // weights 
  HANDLE_ERROR(cudaMalloc((void **)&d_weight, nhidden * nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_weight, h_weight, nhidden * nvisible * sizeof(float), cudaMemcpyHostToDevice));
  
  /*
  // bias to global memory
  HANDLE_ERROR(cudaMalloc((void **)&d_a, nvisible * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_a, h_a, nvisible * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(a, &d_a, sizeof(float *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&d_b, nhidden * sizeof(float)));
  HANDLE_ERROR(cudaMemcpy(d_b, h_b, nhidden * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(b, &d_b, sizeof(float *), 0, cudaMemcpyHostToDevice));
  */
  
  cublasHandle_t handle;
  cublasStatus_t ret;
  ret = cublasCreate(&handle);
  CUBLAS_HANDLE_ERROR(ret);
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  
  float *h_data_c = (float *)malloc(sizeof(float)*h_miniBatch*nhidden);

  for(unsigned i = 0; i < ninst; i += h_miniBatch){
    unsigned currentBatch = h_miniBatch > (ninst - i)? (ninst - i): h_miniBatch;
    HANDLE_ERROR(cudaMemcpy(d_data_a, m_data + i * nvisible, currentBatch * nvisible * sizeof(float), cudaMemcpyHostToDevice));
    ret = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
                      currentBatch, nhidden, nvisible, &alpha,
                      d_data_a, nvisible, d_weight, nhidden, &beta, d_data_c, h_miniBatch);
    CUBLAS_HANDLE_ERROR(ret);
    HANDLE_ERROR(cudaMemcpy(h_data_c, d_data_c, sizeof(float)*nhidden*h_miniBatch, cudaMemcpyDeviceToHost));
    //printArray(h_data_c, nhidden, h_miniBatch);
    cout << "result:" << h_data_c[0] << " " << h_data_c[h_miniBatch] << " " << h_data_c[1];
  }
  cublasDestroy(handle);

        HANDLE_ERROR(cudaEventRecord(stop, NULL));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float msecTotal = 0.0f;
        HANDLE_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("\tcublas: %.2f msec\n", msecTotal);

  HANDLE_ERROR(cudaFree(d_data_a));
  HANDLE_ERROR(cudaFree(d_data_c));
  HANDLE_ERROR(cudaFree(d_a));
  HANDLE_ERROR(cudaFree(d_b));
  free(h_data_c);
  free(m_data);
}
