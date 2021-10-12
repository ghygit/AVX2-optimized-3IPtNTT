#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "params.h"
#include "brv.h"
#include "getw.h"

void ntt(short *p);

void invntt(short* p);

void poly_mul(short* a, short* b, short* ab);