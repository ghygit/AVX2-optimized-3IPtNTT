#include <stdio.h>
#include <math.h>
#include "params.h"
#include "brv.h"

int m = 0;

int reverse_bit(int value)
{
    int sum = 0, n = Pt3_N, mm = 0;
    while (n > 1)
    {
        n >>= 1;
        mm++;
    }
    m = mm;
    for (; value != 0, mm != 0; value = value >> 1, mm--)
        if (value & 1)sum += 1 << (mm - 1);
    return sum;
}

void bitrv()
{
    int i;
    for (i = 0; i < Pt3_N; i++)
        brv[i] = reverse_bit(i);
}