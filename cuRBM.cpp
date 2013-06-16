#include <cstring>
#include <cmath>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <Eigen/Dense>
#include "cuRBM.h"

using namespace boost;
using namespace Eigen;

size_t h_pitch_data, width, h_miniBatch;
unsigned len, nvisible, nhidden, nInst;
int *h_data;
weight_t *h_weight, *h_a, *h_b;
int nbits = sizeof(int) * 8;

void initData(){
  // Initialize data with random values on host
  len = (nvisible - 1)/nbits + 1;
  h_pitch_data = len * sizeof(int);
  width = len * sizeof(int);

  h_data = (int *)malloc(len * nInst * sizeof(int));
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
  // Initialize bias for visible units
  h_a = (weight_t *)malloc(nvisible * sizeof(weight_t));
  unsigned *on_count = (unsigned *)malloc(nvisible * sizeof(unsigned));
  memset(on_count, 0, nvisible * sizeof(unsigned));
  for(int i = 0; i < nInst; ++ i){
    for(int j = 0; j < nvisible; ++j){
      if(h_data[j/nbits] & (1<<(nbits-1-j%nbits)))
        ++ on_count[j];
    }
  }
  for(int i = 0; i < nvisible; ++ i){
    double p = 1.0 * on_count[i] / nInst;
    h_a[i] = log(p) - log(1-p);
  }
  free(on_count);
}

void initHidBias(){
  // Initialize bias for hidden units
  h_b = (weight_t *)malloc(nhidden * sizeof(weight_t));
  for(int i = 0; i < nhidden; ++ i)
    h_b[i] = -4;
}

void arrayToMatrix(MatrixXf &m_data){
  for(unsigned i = 0; i < nInst; ++i)
    for(unsigned j = 0; j < nvisible; ++j){
	  int compressed = *(h_data + i*len + j/nbits);
	  unsigned mask = 1 << (nbits - 1 - j%nbits);
	  if(compressed & mask)
	    m_data(i, j) = 1;
	  else
	    m_data(i, j) = 0;
	}
}

void printArray(float *array, unsigned height, unsigned width){
  cout << endl;
  for(unsigned i = 0; i < height; ++ i){
    for(unsigned j = 0; j < width; ++ j)
      cout << *(array + i * width + j) << " ";
    cout << endl;
  }
}

void rbm(){
  MatrixXf m_data(nInst, nvisible);
  arrayToMatrix(m_data);
  Map<MatrixXf> m_weight(h_weight, nvisible, nhidden);
  Map<VectorXf> m_a(h_a, nvisible);
  Map<VectorXf> m_b(h_b, nhidden);
  cout << "data * weight" << endl;
  cout << m_data << endl;
  cout << m_weight << endl;
  cout << endl << "result:" << endl;
  MatrixXf result = m_data*m_weight;
  result.rowwise() += m_b.transpose();
  cout << result << endl;
  /*
  int array[8];
  for(int i = 0; i < 8; ++i) array[i] = i;
  Map<Matrix<int,2,4,RowMajor> > m_aa(array);
  MatrixXi m = MatrixXi::Random(3,3);
  cout << m << endl;
  */
  printArray(h_weight, nhidden, nvisible);
}

int main(int argc, char **argv){
  h_miniBatch = 2;
  nvisible = 6;
  nhidden = 4;
  nInst = 3;

  initData();
  initWeight();
  initVisBias();
  initHidBias();
  
  deviceInit();
  runRBM();
  rbm();	
}

