#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

typedef float weight_t;

extern size_t h_pitch_data, width, h_miniBatch;
extern unsigned len, nvisible, nhidden, nInst;
extern int *h_data;
extern weight_t *h_weight, *h_a, *h_b;

extern void deviceInit();
extern void runRBM();
