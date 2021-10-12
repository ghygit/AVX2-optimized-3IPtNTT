#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "params.h"
#include "ntt.h"
#include "Pt_3.h"

double test_ImPt3NTT_time(short* gamma)
{
	clock_t start, finish;
	double total_time, cyc_NTT;
	short a[Original_N] = { 0 }, ab[Original_N] = { 0 }, b[Original_N] = { 0 };
	short r0[Pt3_N], r1[Pt3_N], r2[Pt3_N], r3[Pt3_N], r4[Pt3_N], r5[Pt3_N], r6[Pt3_N], r7[Pt3_N],
          s0[Pt3_N], s1[Pt3_N], s2[Pt3_N], s3[Pt3_N], s4[Pt3_N], s5[Pt3_N], s6[Pt3_N], s7[Pt3_N];
	int i, t, count;
	srand(clock());
	for (i = 0; i < Original_N; i++)
	{
		a[i] = ((int)rand()) % Original_Q;
	}
	for (i = 0; i < Original_N; i++)
	{
		b[i] = ((int)rand()) % Original_Q;
	}
	start = clock();
	for (count = 0; count < TI_TEST; count++)
	{
		pt_f(a, r0, r1, r2, r3, r4, r5, r6, r7);
		pt_f(b, s0, s1, s2, s3, s4, s5, s6, s7);
		ntt(r0); ntt(r1); ntt(r2); ntt(r3); ntt(r4); ntt(r5); ntt(r6); ntt(r7);
		ntt(s0); ntt(s1); ntt(s2); ntt(s3); ntt(s4); ntt(s5); ntt(s6); ntt(s7);
		Imp3ntt_mul(r0,  r1,  r2,  r3,  r4,  r5,  r6,  r7, s0,  s1,  s2,  s3,  s4,  s5,  s6,  s7, gamma);
		for (i = 0; i < Pt3_N; i++)
		{
			ab[8 * i] = s0[i];
			ab[8 * i + 1] = s1[i];
			ab[8 * i + 2] = s2[i];
			ab[8 * i + 3] = s3[i];
			ab[8 * i + 4] = s4[i];
			ab[8 * i + 5] = s5[i];
			ab[8 * i + 6] = s6[i];
			ab[8 * i + 7] = s7[i];
		}
	}
	finish = clock();
	total_time = (double)(finish - start) / CLOCKS_PER_SEC;
	cyc_NTT = FREQ_CPU / TI_TEST * total_time;
	printf("ImPt1NTT_mul time * 65536    :%f s\n", total_time);
	return cyc_NTT;
}