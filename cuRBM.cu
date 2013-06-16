#include "cuRBM.h"

__constant__ unsigned nVis;
__constant__ unsigned nHid;
__constant__ unsigned batch;
__constant__ unsigned miniBatch;
__constant__ int *data;
__constant__ weight_t *weight;
__constant__ weight_t *a;
__constant__ weight_t *b;
__constant__ size_t pitch_data;
__constant__ size_t pitch_weight;

int *d_data;
weight_t *d_weight, *d_a, *d_b;
size_t d_pitch_weight, d_pitch_data;
/*
void HANDLE_ERROR(cudaError_t error){
  if(error != cudaSuccess){
	cout << "CUDA error: " << cudaGetErrorString(error) << endl;
	exit(-1);
   }
}
*/
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}


void batchTransfer(unsigned start, unsigned nCase){
  // Copy data to device coalesced
  int *data = h_data + len * start;
  HANDLE_ERROR(cudaMemcpy2D(d_data, d_pitch_data, data, h_pitch_data, width, nCase, cudaMemcpyHostToDevice));
}

void runRBM(){
  for(unsigned i = 0; i < nInst; i += h_miniBatch){
    unsigned currentBatch = h_miniBatch > (nInst - i)? (nInst - i): h_miniBatch;
    batchTransfer(i, currentBatch);
    /*
    unsigned *d = (unsigned *)malloc(len * nInst * sizeof(unsigned));
    HANDLE_ERROR(cudaMemcpy2D(d, h_pitch_data, d_data, d_pitch_data, 
                                width, currentBatch, cudaMemcpyDeviceToHost));
    cout << *(h_data + i * len) << endl;
    cout << d[0] << endl;
    */
  }
}

void deviceInit(){
  // basic parameters to constant memory
  HANDLE_ERROR(cudaMemcpyToSymbol(miniBatch, &h_miniBatch, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(nVis, &nvisible, sizeof(unsigned), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(nHid, &nhidden, sizeof(unsigned), 0, cudaMemcpyHostToDevice));

  // allocate global memory for data of mini batch 
  HANDLE_ERROR(cudaMallocPitch((void **)&d_data, &d_pitch_data, len * sizeof(int), h_miniBatch));
  HANDLE_ERROR(cudaMemcpyToSymbol(data, &d_data, sizeof(int *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(pitch_data, &d_pitch_data, sizeof(size_t), 0, cudaMemcpyHostToDevice));

  // weights to global memory
  HANDLE_ERROR(cudaMallocPitch((void **)&d_weight, &d_pitch_weight, nhidden * sizeof(weight_t), nvisible));
  HANDLE_ERROR(cudaMemcpyToSymbol(weight, &d_weight, sizeof(weight_t *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(pitch_weight, &d_pitch_weight, sizeof(size_t), 0, cudaMemcpyHostToDevice));
  
  // bias to global memory
  HANDLE_ERROR(cudaMalloc((void **)&d_a, nvisible * sizeof(weight_t)));
  HANDLE_ERROR(cudaMemcpyToSymbol(a, &d_a, sizeof(weight_t *), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&d_b, nhidden * sizeof(weight_t)));
  HANDLE_ERROR(cudaMemcpyToSymbol(b, &d_b, sizeof(weight_t *), 0, cudaMemcpyHostToDevice));
}
