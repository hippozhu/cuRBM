#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//#include <helper_string.h>  // helper for shared functions common to CUDA SDK samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

//typedef float weight_t;

extern size_t h_pitch_data, h_pitch_data_hid, width, width_hid;
extern unsigned len, len_hid, nvisible, nhidden, ninst, h_miniBatch;
extern int *h_data, *h_data_hid;
extern float *h_weight, *h_a, *h_b;

extern int *d_data, d_data_hid;
extern float *d_data_hid_float;
extern float *d_weight, *d_a, *d_b;
extern size_t d_pitch_weight, d_pitch_data,  d_pitch_data_hid;

extern void runRBM();
extern void cublasRunRBM();
extern void arrayToMatrix(float *);
extern void printArray(float *array, unsigned height, unsigned width);

#ifndef DEVMAIN
extern __constant__ int *data, *data_hid;
extern __constant__ float *data_vis_float, *data_hid_float;
extern __constant__ unsigned nVis, nHid, nCase, miniBatch, lenVis, lenHid;
extern __constant__ float *weight, *a, *b;
extern __constant__ size_t pitch_data, pitch_data_hid, pitch_weight;
#endif

#define CUBLAS_HANDLE_ERROR( err ) (CublasHandleError( err, __FILE__, __LINE__ ))
static void CublasHandleError( cublasStatus_t err,
                         const char *file,
                         int line ) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf("cublasCreate returned error code %d, line(%d)\n", err, __LINE__);
        exit( EXIT_FAILURE );
    }
}

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
