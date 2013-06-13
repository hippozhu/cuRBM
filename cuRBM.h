#include <iostream>
#include <stdlib.h>

using namespace std;

typedef float weight_t;

extern size_t h_pitch, width, h_batch, h_miniBatch, d_pitch;
extern unsigned len;
extern int *h_data, *d_data;
extern weight_t *h_weight, *h_a, *h_b;
