#include <cstring>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "cuRBM.h"

using namespace boost;

size_t h_pitch, width, h_batch, h_miniBatch, d_pitch;
unsigned len, nvisible, nhidden, nInst;
int *h_data, *d_data;
weight_t *h_weight, *h_a, *h_b;

void initData(){
  // Initialize data with random values on host
  len = (nvisible - 1)/sizeof(int) + 1;
  h_pitch = len * sizeof(int);
  width = len * sizeof(int);

  h_data = (int *)malloc(len * nInst *sizeof(int));
  unsigned i = 0;
  while(i < len * nInst)
    h_data[i++] = rand();
}

void initWeight(){
  // Initialize weights by random numbers of a normal distribution (0, 0.01)
  mt19937 rng;
  normal_distribution<weight_t> nd(0.0, .01);
  variate_generator<mt19937&, normal_distribution<weight_t> > var_nor(rng, nd);

  h_weight = (weight_t *)malloc(nvisible * nhidden * sizeof(weight_t));
  unsigned i = 0;
  while(i < nvisible * nhidden)
    h_weight[i++] = var_nor();
}

void initVisBias(){
  h_a = (weight_t *)malloc(nvisible * sizeof(weight_t));
  unsigned *on_count = (unsigned *)malloc(nvisible * sizeof(unsigned));
  memset(on_count, 0, nvisible * sizeof(unsigned));
  for(int i = 0; i < nInst; ++ i){
    for(int j = 0; j < nvisible; ++j){
      if(h_data[j/sizeof(unsigned)] & (1<<(sizeof(unsigned)- 1 -j%sizeof(unsigned))))
        ++ on_count[j];
    }
  }
  for(int i = 0; i < nvisible; ++ i)
    h_a[i] = 1.0 * on_count[i] / nInst;
}

void initHidBias(){
  h_b = (weight_t *)malloc(nhidden * sizeof(weight_t));
  for(int i = 0; i < nhidden; ++ i)
    h_b[i] = -4;
}
int main(int argc, char **argv){
  cout << "hello" << endl;
  h_batch = 1000;
  h_miniBatch = 100;
  nvisible = 1000;
  nhidden = 1000;
  nInst = 10000;
  initData();
  initWeight();
  //batchTransfer(h_data, 100);
}
