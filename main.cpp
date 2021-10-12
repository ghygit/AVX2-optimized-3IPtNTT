#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "params.h"
#include "brv.h"
#include "getw.h"
#include "ntt.h"
#include "test_speed.h"
#include "Pt_3.h"

extern int countw;
extern int m;
int invn;
int invq;
int invn2;
int n2modq;
int brv[Pt3_N] = { 0 };
int wn[Pt3_N] = { 0 };
int wn_chosen[Pt3_N] = { 0 };
int invwn_chosen[Pt3_N] = { 0 };
int wn_chosen_brv[Pt3_N] = { 0 };
int invwn_chosen_brv[Pt3_N] = { 0 };
int zetas[Pt3_N] = { 0 };
int inv_zetas[Pt3_N] = { 0 };
int vv = ((1U << (16 + Bit_mov)) + Original_Q / 2) / Original_Q;

/**/
int main()
{
    int i, j, l, flag = 0, weight, wnn;
    int q = Original_Q, n;
    short a[Original_N], b[Original_N], ab[Original_N], ab_poly[Original_N];
    short r[Original_N], s[Original_N], x[Original_N], y[Original_N];
    short gamma[Pt3_N] = { 0 };
	clock_t start, finish;
	double total_time, cyc_NTT;
    /**/
    short r0[Pt3_N], r1[Pt3_N], r2[Pt3_N], r3[Pt3_N], r4[Pt3_N], r5[Pt3_N], r6[Pt3_N], r7[Pt3_N],
          s0[Pt3_N], s1[Pt3_N], s2[Pt3_N], s3[Pt3_N], s4[Pt3_N], s5[Pt3_N], s6[Pt3_N], s7[Pt3_N];
    /*****************************************************/
    //Sth. must be done before ImPt3Ntt
    bitrv();
    findw();
    wnn = wn[0];
    table_wn_chosen(wnn);
    table_invwn_chosen(wnn);
    table_wn_chosen_brv();
    table_invwn_chosen_brv();
    table_zetas();
    table_inv_zetas();
    n = Pt3_N;
    invn = mod_invese(n, q);
    invn2 = (invn << 16) % q;
    invq = mod_invese(q, 1 << 16);
    n2modq = ((1 << 16) % q) * (1 << 16) % q;
    gamma[1] = 1;
    ntt(gamma);
    for (i = 0; i < n; i++)
        gamma[i] = (gamma[i] << 16) % q;
    /*****************************************************/
    n = Original_N;
    for (l = 0; l < 10; l++)
    {
        flag = 0;
        for (j = 0; j < 1000; j++)
        {
            for (i = 0; i < n; i++)
            {
                srand(clock() * (i * j+1));
                a[i] = rand() % q;
                b[i] = rand() % q;
            }
            for (i = 0; i < n; i++)
            {
                x[i] = r[i] = a[i];
                y[i] = s[i] = b[i];
            }
            poly_mul(x, y, ab_poly);
    /**********3IPtNTT***********/
    pt_f(r, r0, r1, r2, r3, r4, r5, r6, r7);
    pt_f(s, s0, s1, s2, s3, s4, s5, s6, s7);
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
    /****************************/
            for (i = 0; i < n; i++)
                if (ab[i] - ab_poly[i] != 0)
                {
                    flag++;
                }
        }
    
    printf("\t");
    if (flag != 0)
        printf("AVX_Pt3NTT Error!\n");
    else
        printf("AVX_Pt3NTT Right!\t");
    }
    printf("\n******************************************************************************\n");

    double cyc_ImPt3NTT;

    weight = 0;
    while (wnn > 0)
    {
        if (wnn & 1) weight++;
        wnn >>= 1;
    }
    printf("wn=%4d,", wn[0]);
    printf("\tweight=%4d\n", weight);

    /**/
    FILE* fp;
#ifdef VS
#ifdef Original_N256
#ifdef Original_Q1409
    fopen_s(&fp, "ImPt3NTT256_1409.txt", "w+");
#endif
#endif
#ifdef Original_N512
#ifdef Original_Q1409
    fopen_s(&fp, "ImPt3NTT512_1409.txt", "w+");
#endif
#endif
#ifdef Original_N256
#ifdef Original_Q3329
    fopen_s(&fp, "ImPt3NTT256_3329.txt", "w+");
#endif
#endif
#ifdef Original_N512
#ifdef Original_Q3329
    fopen_s(&fp, "ImPt3NTT512_3329.txt", "w+");
#endif
#endif
#ifdef Original_N1024
#ifdef Original_Q3329
    fopen_s(&fp, "ImPt3NTT1024_3329.txt", "w+");
#endif
#endif
#ifdef Original_N256
#ifdef Original_Q7681
    fopen_s(&fp, "ImPt3NTT256_7681.txt", "w+");
#endif
#endif
#ifdef Original_N512
#ifdef Original_Q7681
    fopen_s(&fp, "ImPt3NTT512_7681.txt", "w+");
#endif
#endif
#ifdef Original_N1024
#ifdef Original_Q7681
    fopen_s(&fp, "ImPt3NTT1024_7681.txt", "w+");
#endif
#endif
#ifdef Original_N256
#ifdef Original_Q12289
    fopen_s(&fp, "ImPt3NTT256_12289.txt", "w+");
#endif
#endif
#ifdef Original_N512
#ifdef Original_Q12289
    fopen_s(&fp, "ImPt3NTT512_12289.txt", "w+");
#endif
#endif
#ifdef Original_N1024
#ifdef Original_Q12289
    fopen_s(&fp, "ImPt3NTT1024_12289.txt", "w+");
#endif
#endif
#endif
#ifdef VSCODE
#ifdef Original_N256
#ifdef Original_Q1409
    fp=fopen("ImPt3NTT256_1409.txt", "w+");
#endif
#endif
#ifdef Original_N512
#ifdef Original_Q1409
    fp=fopen("ImPt3NTT512_1409.txt", "w+");
#endif
#endif
#ifdef Original_N256
#ifdef Original_Q3329
    fp=fopen("ImPt3NTT256_3329.txt", "w+");
#endif
#endif
#ifdef Original_N512
#ifdef Original_Q3329
    fp=fopen("ImPt3NTT512_3329.txt", "w+");
#endif
#endif
#ifdef Original_N1024
#ifdef Original_Q3329
    fp=fopen("ImPt3NTT1024_3329.txt", "w+");
#endif
#endif
#ifdef Original_N256
#ifdef Original_Q7681
    fp=fopen("ImPt3NTT256_7681.txt", "w+");
#endif
#endif
#ifdef Original_N512
#ifdef Original_Q7681
    fp=fopen("ImPt3NTT512_7681.txt", "w+");
#endif
#endif
#ifdef Original_N1024
#ifdef Original_Q7681
    fp=fopen("ImPt3NTT1024_7681.txt", "w+");
#endif
#endif
#ifdef Original_N256
#ifdef Original_Q12289
    fp=fopen("ImPt3NTT256_12289.txt", "w+");
#endif
#endif
#ifdef Original_N512
#ifdef Original_Q12289
    fp=fopen("ImPt3NTT512_12289.txt", "w+");
#endif
#endif
#ifdef Original_N1024
#ifdef Original_Q12289
    fp=fopen("ImPt3NTT1024_12289.txt", "w+");
#endif
#endif
#endif
    if (fp == NULL)
    {
        printf("File cannot open! ");
        exit(0);
    }
    for (l = 0; l < 10; l++)
    {
        
        start = clock();
        for (j = 0; j < TI_TEST; j++)
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
        cyc_ImPt3NTT = cyc_NTT;
        fprintf(fp, "%f\n", cyc_ImPt3NTT);
    }
    fclose(fp);/**/
    return 0;
}
