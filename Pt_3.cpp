#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include "params.h"
#include "ntt.h"

extern int n2modq;
extern int invq;
extern int vv;

void pt_f(short *a, short*b, short*c, short* d, short* e, short* f, short* g, short* h, short* l)
{
     int i;
     int n = Pt3_N;
     for (i = 0; i < n; i++)
     {
        b[i] = a[8 * i];
        c[i] = a[8 * i + 1];
        d[i] = a[8 * i + 2];
        e[i] = a[8 * i + 3];
        f[i] = a[8 * i + 4];
        g[i] = a[8 * i + 5];
        h[i] = a[8 * i + 6];
        l[i] = a[8 * i + 7];
     }
}

void Imp3ntt_mul(short* r0, short* r1, short* r2, short* r3, short* r4, short* r5, short* r6, short* r7, 
                short* s0, short* s1, short* s2, short* s3, short* s4, short* s5, short* s6, short* s7, short* g)
{
    __m256i _s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7,
            _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7,
            _r01, _r02, _r03, _r04, _r05, _r06, _r07,
            _r12, _r13, _r14, _r15, _r16, _r17,
            _r23, _r24, _r25, _r26, _r27,
            _r34, _r35, _r36, _r37,
            _r45, _r46, _r47,
            _r56, _r57,
            _r67,
            _m0, _m1, _m2, _m3, _m4, _m5, _m6, _m7,
            _m01, _m02, _m03, _m04, _m05, _m06, _m07, _m67,
            _m12, _m13, _m14, _m15, _m16, _m17, _m56, _m57,
            _m23, _m24, _m25, _m26, _m27, _m45, _m46, _m47,
            _m34, _m35, _m36, _m37, _m77,
            modq, _n2modq, _tmp, modinvq, _ga, _VV, _h;
    int i, n2 = Pt3_N;
    //the operation to do piontwise multiplication
    _n2modq = _mm256_set1_epi16(n2modq);
    modq = _mm256_set1_epi16(Original_Q);
    modinvq = _mm256_set1_epi16(invq);
    _VV = _mm256_set1_epi16(vv);
    for (i = 0; i < n2; i += 16)
    {
        _r0 = _mm256_load_si256((__m256i*)(r0 + i));
        _r1 = _mm256_load_si256((__m256i*)(r1 + i));
        _r2 = _mm256_load_si256((__m256i*)(r2 + i));
        _r3 = _mm256_load_si256((__m256i*)(r3 + i));
        _r4 = _mm256_load_si256((__m256i*)(r4 + i));
        _r5 = _mm256_load_si256((__m256i*)(r5 + i));
        _r6 = _mm256_load_si256((__m256i*)(r6 + i));
        _r7 = _mm256_load_si256((__m256i*)(r7 + i));
        _s0 = _mm256_load_si256((__m256i*)(s0 + i));
        _s1 = _mm256_load_si256((__m256i*)(s1 + i));
        _s2 = _mm256_load_si256((__m256i*)(s2 + i));
        _s3 = _mm256_load_si256((__m256i*)(s3 + i));
        _s4 = _mm256_load_si256((__m256i*)(s4 + i));
        _s5 = _mm256_load_si256((__m256i*)(s5 + i));
        _s6 = _mm256_load_si256((__m256i*)(s6 + i));
        _s7 = _mm256_load_si256((__m256i*)(s7 + i));
        /**/
        //m0[i] = r0[i] * s0[i] % q;///////////////////////////////
        //Montgomery red
        _h = _mm256_mulhi_epi16(_r0, _n2modq);
        _r0 = _mm256_mullo_epi16(_r0, _n2modq);
        _tmp = _mm256_mullo_epi16(_r0, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _r0 = _mm256_sub_epi16(_h, _tmp);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r0, _s0);
        _m0 = _mm256_mullo_epi16(_r0, _s0);
        _tmp = _mm256_mullo_epi16(_m0, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m0 = _mm256_sub_epi16(_h, _tmp);

        //m1[i] = r1[i] * s1[i] % q;///////////////////////////////
        //Montgomery red
        _h = _mm256_mulhi_epi16(_r1, _n2modq);
        _r1 = _mm256_mullo_epi16(_r1, _n2modq);
        _tmp = _mm256_mullo_epi16(_r1, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _r1 = _mm256_sub_epi16(_h, _tmp);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r1, _s1);
        _m1 = _mm256_mullo_epi16(_r1, _s1);
        _tmp = _mm256_mullo_epi16(_m1, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m1 = _mm256_sub_epi16(_h, _tmp);

        //m2[i] = r2[i] * s2[i] % q;///////////////////////////////
        //Montgomery red
        _h = _mm256_mulhi_epi16(_r2, _n2modq);
        _r2 = _mm256_mullo_epi16(_r2, _n2modq);
        _tmp = _mm256_mullo_epi16(_r2, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _r2 = _mm256_sub_epi16(_h, _tmp);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r2, _s2);
        _m2 = _mm256_mullo_epi16(_r2, _s2);
        _tmp = _mm256_mullo_epi16(_m2, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m2 = _mm256_sub_epi16(_h, _tmp);

        //m3[i] = r3[i] * s3[i] % q;///////////////////////////////
        //Montgomery red
        _h = _mm256_mulhi_epi16(_r3, _n2modq);
        _r3 = _mm256_mullo_epi16(_r3, _n2modq);
        _tmp = _mm256_mullo_epi16(_r3, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _r3 = _mm256_sub_epi16(_h, _tmp);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r3, _s3);
        _m3 = _mm256_mullo_epi16(_r3, _s3);
        _tmp = _mm256_mullo_epi16(_m3, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m3 = _mm256_sub_epi16(_h, _tmp);

        //m4[i] = r4[i] * s4[i] % q;///////////////////////////////
        //Montgomery red
        _h = _mm256_mulhi_epi16(_r4, _n2modq);
        _r4 = _mm256_mullo_epi16(_r4, _n2modq);
        _tmp = _mm256_mullo_epi16(_r4, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _r4 = _mm256_sub_epi16(_h, _tmp);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r4, _s4);
        _m4 = _mm256_mullo_epi16(_r4, _s4);
        _tmp = _mm256_mullo_epi16(_m4, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m4 = _mm256_sub_epi16(_h, _tmp);

        //m5[i] = r5[i] * s5[i] % q;///////////////////////////////
        //Montgomery red
        _h = _mm256_mulhi_epi16(_r5, _n2modq);
        _r5 = _mm256_mullo_epi16(_r5, _n2modq);
        _tmp = _mm256_mullo_epi16(_r5, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _r5 = _mm256_sub_epi16(_h, _tmp);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r5, _s5);
        _m5 = _mm256_mullo_epi16(_r5, _s5);
        _tmp = _mm256_mullo_epi16(_m5, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m5 = _mm256_sub_epi16(_h, _tmp);

        //m6[i] = r6[i] * s6[i] % q;///////////////////////////////
        //Montgomery red
        _h = _mm256_mulhi_epi16(_r6, _n2modq);
        _r6 = _mm256_mullo_epi16(_r6, _n2modq);
        _tmp = _mm256_mullo_epi16(_r6, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_h, _tmp);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r6, _s6);
        _m6 = _mm256_mullo_epi16(_r6, _s6);
        _tmp = _mm256_mullo_epi16(_m6, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m6 = _mm256_sub_epi16(_h, _tmp);

        //m7[i] = r7[i] * s7[i] % q;///////////////////////////////
        //Montgomery red
        _h = _mm256_mulhi_epi16(_r7, _n2modq);
        _r7 = _mm256_mullo_epi16(_r7, _n2modq);
        _tmp = _mm256_mullo_epi16(_r7, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _r7 = _mm256_sub_epi16(_h, _tmp);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r7, _s7);
        _m7 = _mm256_mullo_epi16(_r7, _s7);
        _tmp = _mm256_mullo_epi16(_m7, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m7 = _mm256_sub_epi16(_h, _tmp);

        //m01[i] = (r0[i] + r1[i]) * (s0[i] + s1[i]) % q;/////////////////////////
        _r01 = _mm256_add_epi16(_r0, _r1);
        _m01 = _mm256_add_epi16(_s0, _s1);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r01, _m01);
        _m01 = _mm256_mullo_epi16(_r01, _m01);
        _tmp = _mm256_mullo_epi16(_m01, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m01 = _mm256_sub_epi16(_h, _tmp);

        //m02[i] = (r0[i] + r2[i]) * (s0[i] + s2[i]) % q;/////////////////////////
        _r02 = _mm256_add_epi16(_r0, _r2);
        _m02 = _mm256_add_epi16(_s0, _s2);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r02, _m02);
        _m02 = _mm256_mullo_epi16(_r02, _m02);
        _tmp = _mm256_mullo_epi16(_m02, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m02 = _mm256_sub_epi16(_h, _tmp);

        //m03[i] = (r0[i] + r3[i]) * (s0[i] + s3[i]) % q;/////////////////////////
        _r03 = _mm256_add_epi16(_r0, _r3);
        _m03 = _mm256_add_epi16(_s0, _s3);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r03, _m03);
        _m03 = _mm256_mullo_epi16(_r03, _m03);
        _tmp = _mm256_mullo_epi16(_m03, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m03 = _mm256_sub_epi16(_h, _tmp);

        //m04[i] = (r0[i] + r4[i]) * (s0[i] + s4[i]) % q;/////////////////////////
        _r04 = _mm256_add_epi16(_r0, _r4);
        _m04 = _mm256_add_epi16(_s0, _s4);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r04, _m04);
        _m04 = _mm256_mullo_epi16(_r04, _m04);
        _tmp = _mm256_mullo_epi16(_m04, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m04 = _mm256_sub_epi16(_h, _tmp);

        //m05[i] = (r0[i] + r5[i]) * (s0[i] + s5[i]) % q;/////////////////////////
        _r05 = _mm256_add_epi16(_r0, _r5);
        _m05 = _mm256_add_epi16(_s0, _s5);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r05, _m05);
        _m05 = _mm256_mullo_epi16(_r05, _m05);
        _tmp = _mm256_mullo_epi16(_m05, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m05 = _mm256_sub_epi16(_h, _tmp);

        //m06[i] = (r0[i] + r6[i]) * (s0[i] + s6[i]) % q;/////////////////////////
        _r06 = _mm256_add_epi16(_r0, _r6);
        _m06 = _mm256_add_epi16(_s0, _s6);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r06, _m06);
        _m06 = _mm256_mullo_epi16(_r06, _m06);
        _tmp = _mm256_mullo_epi16(_m06, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m06 = _mm256_sub_epi16(_h, _tmp);

        //m07[i] = (r0[i] + r7[i]) * (s0[i] + s7[i]) % q;/////////////////////////
        _r07 = _mm256_add_epi16(_r0, _r7);
        _m07 = _mm256_add_epi16(_s0, _s7);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r07, _m07);
        _m07 = _mm256_mullo_epi16(_r07, _m07);
        _tmp = _mm256_mullo_epi16(_m07, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m07 = _mm256_sub_epi16(_h, _tmp);

        //m12[i] = (r1[i] + r2[i]) * (s2[i] + s1[i]) % q;/////////////////////////
        _r12 = _mm256_add_epi16(_r1, _r2);
        _m12 = _mm256_add_epi16(_s1, _s2);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r12, _m12);
        _m12 = _mm256_mullo_epi16(_r12, _m12);
        _tmp = _mm256_mullo_epi16(_m12, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m12 = _mm256_sub_epi16(_h, _tmp);

        //m13[i] = (r1[i] + r3[i]) * (s1[i] + s3[i]) % q;/////////////////////////
        _r13 = _mm256_add_epi16(_r1, _r3);
        _m13 = _mm256_add_epi16(_s1, _s3);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r13, _m13);
        _m13 = _mm256_mullo_epi16(_r13, _m13);
        _tmp = _mm256_mullo_epi16(_m13, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m13 = _mm256_sub_epi16(_h, _tmp);

        //m14[i] = (r1[i] + r4[i]) * (s1[i] + s4[i]) % q;/////////////////////////
        _r14 = _mm256_add_epi16(_r1, _r4);
        _m14 = _mm256_add_epi16(_s1, _s4);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r14, _m14);
        _m14 = _mm256_mullo_epi16(_r14, _m14);
        _tmp = _mm256_mullo_epi16(_m14, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m14 = _mm256_sub_epi16(_h, _tmp);

        //m15[i] = (r1[i] + r5[i]) * (s1[i] + s5[i]) % q;/////////////////////////
        _r15 = _mm256_add_epi16(_r1, _r5);
        _m15 = _mm256_add_epi16(_s1, _s5);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r15, _m15);
        _m15 = _mm256_mullo_epi16(_r15, _m15);
        _tmp = _mm256_mullo_epi16(_m15, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m15 = _mm256_sub_epi16(_h, _tmp);

        //m16[i] = (r1[i] + r6[i]) * (s1[i] + s6[i]) % q;/////////////////////////
        _r16 = _mm256_add_epi16(_r1, _r6);
        _m16 = _mm256_add_epi16(_s1, _s6);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r16, _m16);
        _m16 = _mm256_mullo_epi16(_r16, _m16);
        _tmp = _mm256_mullo_epi16(_m16, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m16 = _mm256_sub_epi16(_h, _tmp);

        //m17[i] = (r1[i] + r7[i]) * (s1[i] + s7[i]) % q;/////////////////////////
        _r17 = _mm256_add_epi16(_r1, _r7);
        _m17 = _mm256_add_epi16(_s1, _s7);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r17, _m17);
        _m17 = _mm256_mullo_epi16(_r17, _m17);
        _tmp = _mm256_mullo_epi16(_m17, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_h, _tmp);

        //m23[i] = (r2[i] + r3[i]) * (s2[i] + s3[i]) % q;/////////////////////////
        _r23 = _mm256_add_epi16(_r2, _r3);
        _m23 = _mm256_add_epi16(_s2, _s3);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r23, _m23);
        _m23 = _mm256_mullo_epi16(_r23, _m23);
        _tmp = _mm256_mullo_epi16(_m23, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m23 = _mm256_sub_epi16(_h, _tmp);

        //m24[i] = (r2[i] + r4[i]) * (s2[i] + s4[i]) % q;/////////////////////////
        _r24 = _mm256_add_epi16(_r2, _r4);
        _m24 = _mm256_add_epi16(_s2, _s4);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r24, _m24);
        _m24 = _mm256_mullo_epi16(_r24, _m24);
        _tmp = _mm256_mullo_epi16(_m24, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m24 = _mm256_sub_epi16(_h, _tmp);

        //m25[i] = (r2[i] + r5[i]) * (s2[i] + s5[i]) % q;/////////////////////////
        _r25 = _mm256_add_epi16(_r2, _r5);
        _m25 = _mm256_add_epi16(_s2, _s5);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r25, _m25);
        _m25 = _mm256_mullo_epi16(_r25, _m25);
        _tmp = _mm256_mullo_epi16(_m25, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m25 = _mm256_sub_epi16(_h, _tmp);

        //m26[i] = (r2[i] + r6[i]) * (s2[i] + s6[i]) % q;/////////////////////////
        _r26 = _mm256_add_epi16(_r2, _r6);
        _m26 = _mm256_add_epi16(_s2, _s6);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r26, _m26);
        _m26 = _mm256_mullo_epi16(_r26, _m26);
        _tmp = _mm256_mullo_epi16(_m26, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m26 = _mm256_sub_epi16(_h, _tmp);

        //m27[i] = (r2[i] + r7[i]) * (s2[i] + s7[i]) % q;/////////////////////////
        _r27 = _mm256_add_epi16(_r2, _r7);
        _m27 = _mm256_add_epi16(_s2, _s7);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r27, _m27);
        _m27 = _mm256_mullo_epi16(_r27, _m27);
        _tmp = _mm256_mullo_epi16(_m27, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m27 = _mm256_sub_epi16(_h, _tmp);

        //m34[i] = (r3[i] + r4[i]) * (s3[i] + s4[i]) % q;/////////////////////////
        _r34 = _mm256_add_epi16(_r3, _r4);
        _m34 = _mm256_add_epi16(_s3, _s4);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r34, _m34);
        _m34 = _mm256_mullo_epi16(_r34, _m34);
        _tmp = _mm256_mullo_epi16(_m34, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m34 = _mm256_sub_epi16(_h, _tmp);

        //m35[i] = (r3[i] + r5[i]) * (s3[i] + s5[i]) % q;/////////////////////////
        _r35 = _mm256_add_epi16(_r3, _r5);
        _m35 = _mm256_add_epi16(_s3, _s5);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r35, _m35);
        _m35 = _mm256_mullo_epi16(_r35, _m35);
        _tmp = _mm256_mullo_epi16(_m35, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m35 = _mm256_sub_epi16(_h, _tmp);

        //m36[i] = (r3[i] + r6[i]) * (s3[i] + s6[i]) % q;/////////////////////////
        _r36 = _mm256_add_epi16(_r3, _r6);
        _m36 = _mm256_add_epi16(_s3, _s6);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r36, _m36);
        _m36 = _mm256_mullo_epi16(_r36, _m36);
        _tmp = _mm256_mullo_epi16(_m36, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m36 = _mm256_sub_epi16(_h, _tmp);

        //m37[i] = (r3[i] + r7[i]) * (s3[i] + s7[i]) % q;/////////////////////////
        _r37 = _mm256_add_epi16(_r3, _r7);
        _m37 = _mm256_add_epi16(_s3, _s7);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r37, _m37);
        _m37 = _mm256_mullo_epi16(_r37, _m37);
        _tmp = _mm256_mullo_epi16(_m37, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m37 = _mm256_sub_epi16(_h, _tmp);

        //m45[i] = (r4[i] + r5[i]) * (s4[i] + s5[i]) % q;/////////////////////////
        _r45 = _mm256_add_epi16(_r4, _r5);
        _m45 = _mm256_add_epi16(_s4, _s5);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r45, _m45);
        _m45 = _mm256_mullo_epi16(_r45, _m45);
        _tmp = _mm256_mullo_epi16(_m45, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m45 = _mm256_sub_epi16(_h, _tmp);

        //m46[i] = (r4[i] + r6[i]) * (s4[i] + s6[i]) % q;/////////////////////////
        _r46 = _mm256_add_epi16(_r4, _r6);
        _m46 = _mm256_add_epi16(_s4, _s6);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r46, _m46);
        _m46 = _mm256_mullo_epi16(_r46, _m46);
        _tmp = _mm256_mullo_epi16(_m46, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m46 = _mm256_sub_epi16(_h, _tmp);

        //m47[i] = (r4[i] + r7[i]) * (s4[i] + s7[i]) % q;/////////////////////////
        _r47 = _mm256_add_epi16(_r4, _r7);
        _m47 = _mm256_add_epi16(_s4, _s7);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r47, _m47);
        _m47 = _mm256_mullo_epi16(_r47, _m47);
        _tmp = _mm256_mullo_epi16(_m47, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m47 = _mm256_sub_epi16(_h, _tmp);

        //m56[i] = (r5[i] + r6[i]) * (s5[i] + s6[i]) % q;/////////////////////////
        _r56 = _mm256_add_epi16(_r5, _r6);
        _m56 = _mm256_add_epi16(_s5, _s6);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r56, _m56);
        _m56 = _mm256_mullo_epi16(_r56, _m56);
        _tmp = _mm256_mullo_epi16(_m56, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m56 = _mm256_sub_epi16(_h, _tmp);

        //m57[i] = (r5[i] + r7[i]) * (s5[i] + s7[i]) % q;/////////////////////////
        _r57 = _mm256_add_epi16(_r5, _r7);
        _m57 = _mm256_add_epi16(_s5, _s7);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r57, _m57);
        _m57 = _mm256_mullo_epi16(_r57, _m57);
        _tmp = _mm256_mullo_epi16(_m57, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m57 = _mm256_sub_epi16(_h, _tmp);

        //m67[i] = (r6[i] + r7[i]) * (s6[i] + s7[i]) % q;/////////////////////////
        _r67 = _mm256_add_epi16(_r6, _r7);
        _m67 = _mm256_add_epi16(_s6, _s7);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_r67, _m67);
        _m67 = _mm256_mullo_epi16(_r67, _m67);
        _tmp = _mm256_mullo_epi16(_m67, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m67 = _mm256_sub_epi16(_h, _tmp);
        /*****************************************************/
        _ga = _mm256_load_si256((__m256i*)(g + i));
        _s0 = _mm256_add_epi16(_m0, _m1);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _s0);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _s0 = _mm256_sub_epi16(_s0, _tmp);
#endif // Original_Q12289
        _s2 = _mm256_add_epi16(_m2, _m3);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _s2);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);

        _s2 = _mm256_sub_epi16(_s2, _tmp);
#endif // Original_Q12289
        _s1 = _mm256_add_epi16(_s0, _s2);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _s1);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _s1 = _mm256_sub_epi16(_s1, _tmp);
        
        _s3 = _mm256_add_epi16(_m4, _m5);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _s3);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _s3 = _mm256_sub_epi16(_s3, _tmp);
#endif // Original_Q12289

        _s5 = _mm256_add_epi16(_m6, _m7);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _s5);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _s5 = _mm256_sub_epi16(_s5, _tmp);
#endif // Original_Q12289

        _s4 = _mm256_add_epi16(_s3, _s5);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _s4);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _s4 = _mm256_sub_epi16(_s4, _tmp);
        
        //m17[i] = g[i] * (m17[i] - m1[i] - m7[i] + m26[i] - m2[i] - m6[i] + m35[i] - m3[i] - m5[i] + m4[i])/////////////////////////
        //       = g[i] * (m17[i] - m1[i] - m5[i] + m26[i] + m35[i] + m4[i] - (m2[i] + m3[i]) - (m6[i] + m7[i]));
        _m17 = _mm256_sub_epi16(_m17, _m1);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m17);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_m17, _tmp);
#endif // Original_Q12289
        _m17 = _mm256_sub_epi16(_m17, _m5);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m17);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_m17, _tmp);
#endif // Original_Q12289
        _m17 = _mm256_add_epi16(_m17, _m26);
#ifdef Original_Q7681
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m17);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_m17, _tmp);
#endif // Original_Q7681
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m17);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_m17, _tmp);
#endif // Original_Q12289
        _m17 = _mm256_add_epi16(_m17, _m35);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m17);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_m17, _tmp);
#endif // Original_Q12289
        _m17 = _mm256_sub_epi16(_m17, _s2);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m17);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_m17, _tmp);
#endif // Original_Q12289
        _m17 = _mm256_add_epi16(_m17, _m4);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m17);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_m17, _tmp);
#endif // Original_Q12289
#ifdef Original_Q7681
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m17);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_m17, _tmp);
#endif // Original_Q7681
        _m17 = _mm256_sub_epi16(_m17, _s5);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m17);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_m17, _tmp);
#endif // Original_Q12289

        //Montgomery red
        _h = _mm256_mulhi_epi16(_ga, _m17);
        _m17 = _mm256_mullo_epi16(_ga, _m17);
        _tmp = _mm256_mullo_epi16(_m17, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m17 = _mm256_sub_epi16(_h, _tmp);
        //m27[i] = g[i] * (m27[i] - m2[i] - m7[i] + m36[i] - m3[i] - m6[i] + m45[i] - m4[i] - m5[i])/////////////////////////
        //       = g[i] * (m27[i] + m36[i] + m45[i] - (m2[i] + m3[i]) - (m4[i] + m5[i]) - (m6[i] + m7[i]));
        _m27 = _mm256_add_epi16(_m27, _m36);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m27);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m27 = _mm256_sub_epi16(_m27, _tmp);
#endif // Original_Q12289
        _m27 = _mm256_sub_epi16(_m27, _s2);
#ifdef Original_Q7681
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m27);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m27 = _mm256_sub_epi16(_m27, _tmp);
#endif // Original_Q7681
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m27);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m27 = _mm256_sub_epi16(_m27, _tmp);
#endif // Original_Q12289
        _m27 = _mm256_add_epi16(_m27, _m45);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m27);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m27 = _mm256_sub_epi16(_m27, _tmp);
#endif // Original_Q12289
        _m27 = _mm256_sub_epi16(_m27, _s4);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_ga, _m27);
        _m27 = _mm256_mullo_epi16(_ga, _m27);
        _tmp = _mm256_mullo_epi16(_m27, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m27 = _mm256_sub_epi16(_h, _tmp);
        //m37[i] = g[i] * (m37[i] - m3[i] - m7[i] + m46[i] - m4[i] - m6[i] + m5[i]);/////////////////////////
        //       = g[i] * (m37[i] + m46[i] + m5[i] - m3[i] - m4[i] - (m6[i] + m7[i]));
        _m37 = _mm256_add_epi16(_m37, _m46);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m37);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m37 = _mm256_sub_epi16(_m37, _tmp);
#endif // Original_Q12289
        _m37 = _mm256_add_epi16(_m37, _m5);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m37);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m37 = _mm256_sub_epi16(_m37, _tmp);
#endif // Original_Q12289
        _m37 = _mm256_sub_epi16(_m37, _m3);
#ifdef Original_Q7681
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m37);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m37 = _mm256_sub_epi16(_m37, _tmp);
#endif // Original_Q7681
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m37);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m37 = _mm256_sub_epi16(_m37, _tmp);
#endif // Original_Q12289
        _m37 = _mm256_sub_epi16(_m37, _m4);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m37);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m37 = _mm256_sub_epi16(_m37, _tmp);
#endif // Original_Q12289
        _m37 = _mm256_sub_epi16(_m37, _s5);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_ga, _m37);
        _m37 = _mm256_mullo_epi16(_ga, _m37);
        _tmp = _mm256_mullo_epi16(_m37, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m37 = _mm256_sub_epi16(_h, _tmp);

        //m47[i] = g[i] * (m47[i] - m4[i] - m7[i] + m56[i] - m5[i] - m6[i])/////////////////////////
        //       = g[i] * (m47[i] + m56[i] - (m4[i] + m5[i]) - (m6[i] + m7[i]));
        _m47 = _mm256_add_epi16(_m47, _m56);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m47);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m47 = _mm256_sub_epi16(_m47, _tmp);
#endif // Original_Q12289
        _m47 = _mm256_sub_epi16(_m47, _s4);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_ga, _m47);
        _m47 = _mm256_mullo_epi16(_ga, _m47);
        _tmp = _mm256_mullo_epi16(_m47, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m47 = _mm256_sub_epi16(_h, _tmp);

        //m57[i] = g[i] * (m57[i] + m6[i] - m5[i] - m7[i]);/////////////////////////
        _m57 = _mm256_add_epi16(_m57, _m6);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m57);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m57 = _mm256_sub_epi16(_m57, _tmp);
#endif // Original_Q12289
        _m57 = _mm256_sub_epi16(_m57, _m5);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _m57);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _m57 = _mm256_sub_epi16(_m57, _tmp);
#endif // Original_Q12289
        _m57 = _mm256_sub_epi16(_m57, _m7);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_ga, _m57);
        _m57 = _mm256_mullo_epi16(_ga, _m57);
        _tmp = _mm256_mullo_epi16(_m57, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m57 = _mm256_sub_epi16(_h, _tmp);

        //m67[i] = g[i] * (m67[i] - m6[i] - m7[i])/////////////////////////
        //       = g[i] * (m67[i] - (m6[i] + m7[i]));
        _m67 = _mm256_sub_epi16(_m67, _s5);

        //Montgomery red
        _h = _mm256_mulhi_epi16(_ga, _m67);
        _m67 = _mm256_mullo_epi16(_ga, _m67);
        _tmp = _mm256_mullo_epi16(_m67, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m67 = _mm256_sub_epi16(_h, _tmp);

        //m77[i] = g[i] * m7[i];
        //Montgomery red
        _h = _mm256_mulhi_epi16(_ga, _m7);
        _m77 = _mm256_mullo_epi16(_ga, _m7);
        _tmp = _mm256_mullo_epi16(_m77, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _m77 = _mm256_sub_epi16(_h, _tmp);
        /*****************************************************/
        //ab0[i] = (m0[i] + m17[i]) % q;/////////////////////////
        _r0 = _mm256_add_epi16(_m0, _m17);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r0);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r0 = _mm256_sub_epi16(_r0, _tmp);

        //ab1[i] = (m01[i] - m0[i] - m1[i] + m27[i]) % q//////////////////
        //       = (m01[i] + m27[i] - (m0[i] + m1[i])) % q;
        _r1 = _mm256_add_epi16(_m01, _m27);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r1);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r1 = _mm256_sub_epi16(_r1, _tmp);
#endif // Original_Q12289
        _r1 = _mm256_sub_epi16(_r1, _s0);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r1);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r1 = _mm256_sub_epi16(_r1, _tmp);

        //ab2[i] = (m02[i] - m0[i] - m2[i] + m1[i] + m37[i]) % q;//////////////////
        _r2 = _mm256_sub_epi16(_m02, _m0);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r2);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r2 = _mm256_sub_epi16(_r2, _tmp);
#endif // Original_Q12289
        _r2 = _mm256_sub_epi16(_r2, _m2);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r2);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r2 = _mm256_sub_epi16(_r2, _tmp);
#endif // Original_Q12289
        _r2 = _mm256_add_epi16(_r2, _m1);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r2);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r2 = _mm256_sub_epi16(_r2, _tmp);
#endif // Original_Q12289
#ifdef Original_Q7681
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r2);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r2 = _mm256_sub_epi16(_r2, _tmp);
#endif // Original_Q7681
        _r2 = _mm256_add_epi16(_r2, _m37);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r2);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r2 = _mm256_sub_epi16(_r2, _tmp);

        //ab3[i] = (m03[i] - m0[i] - m3[i] + m12[i] - m1[i] - m2[i] + m47[i]) % q//////////////////
        //       = (m03[i] + m12[i] + m47[i] - (m0[i] + m1[i]) - (m2[i] + m3[i])) % q;
        _r3 = _mm256_add_epi16(_m03, _m12);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r3);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r3 = _mm256_sub_epi16(_r3, _tmp);
#endif // Original_Q12289
        _r3 = _mm256_add_epi16(_r3, _m47);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r3);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r3 = _mm256_sub_epi16(_r3, _tmp);
#endif // Original_Q12289
        _r3 = _mm256_sub_epi16(_r3, _s1);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r3);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r3 = _mm256_sub_epi16(_r3, _tmp);
        
        //ab4[i] = (m04[i] - m0[i] - m4[i] + m13[i] - m1[i] - m3[i] + m2[i] + m57[i]) % q
        //       = (m04[i] + m13[i] + m2[i] + m57[i] - (m0[i] + m1[i]) - m3[i] - m4[i]) % q
        _r4 = _mm256_add_epi16(_m04, _m13);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r4);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r4 = _mm256_sub_epi16(_r4, _tmp);
#endif // Original_Q12289
        _r4 = _mm256_add_epi16(_r4, _m2);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r4);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r4 = _mm256_sub_epi16(_r4, _tmp);
#endif // Original_Q12289
        _r4 = _mm256_add_epi16(_r4, _m57);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r4);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r4 = _mm256_sub_epi16(_r4, _tmp);
#endif // Original_Q12289
#ifdef Original_Q7681
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r4);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r4 = _mm256_sub_epi16(_r4, _tmp);
#endif // Original_Q7681
        _r4 = _mm256_sub_epi16(_r4, _s0);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r4);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r4 = _mm256_sub_epi16(_r4, _tmp);
#endif // Original_Q12289
        _r4 = _mm256_sub_epi16(_r4, _m3);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r4);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r4 = _mm256_sub_epi16(_r4, _tmp);
#endif // Original_Q12289
        _r4 = _mm256_sub_epi16(_r4, _m4);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r4);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r4 = _mm256_sub_epi16(_r4, _tmp);

        //ab5[i] = (m05[i] - m0[i] - m5[i] + m14[i] - m1[i] - m4[i] + m23[i] - m2[i] - m3[i] + m67[i]) % q
        //       = (m05[i] + m14[i] + m23[i] + m67[i] - (m1[i] + m0[i]) - (m2[i] + m3[i]) - (m4[i] + m5[i])) % q;
        _r5 = _mm256_add_epi16(_m05, _m14);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r5);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r5 = _mm256_sub_epi16(_r5, _tmp);
#endif // Original_Q12289
        _r5 = _mm256_add_epi16(_r5, _m23);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r5);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r5 = _mm256_sub_epi16(_r5, _tmp);
#endif // Original_Q12289
        _r5 = _mm256_add_epi16(_r5, _m67);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r5);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r5 = _mm256_sub_epi16(_r5, _tmp);
#endif // Original_Q12289
#ifdef Original_Q7681
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r5);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r5 = _mm256_sub_epi16(_r5, _tmp);
#endif // Original_Q7681
        _r5 = _mm256_sub_epi16(_r5, _s1);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r5);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r5 = _mm256_sub_epi16(_r5, _tmp);
#endif // Original_Q12289
        _r5 = _mm256_sub_epi16(_r5, _s3);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r5);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r5 = _mm256_sub_epi16(_r5, _tmp);

        //ab6[i] = (m06[i] - m0[i] - m6[i] + m15[i] - m1[i] - m5[i] + m24[i] - m2[i] - m4[i] + m3[i] + m77[i]) % q
        //       = (m06[i] + m15[i] + m24[i] + m3[i] + m77[i] - (m0[i] + m1[i]) - (m4[i] + m5[i]) - m2[i] - m6[i]) % q;
        _r6 = _mm256_add_epi16(_m06, _m15);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r6);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_r6, _tmp);
#endif // Original_Q12289
        _r6 = _mm256_add_epi16(_r6, _m24);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r6);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_r6, _tmp);
#endif // Original_Q12289
        _r6 = _mm256_add_epi16(_r6, _m3);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r6);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_r6, _tmp);
#endif // Original_Q12289
#ifdef Original_Q7681
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r6);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_r6, _tmp);
#endif // Original_Q7681
        _r6 = _mm256_add_epi16(_r6, _m77);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r6);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_r6, _tmp);
#endif // Original_Q12289
        _r6 = _mm256_sub_epi16(_r6, _s0);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r6);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_r6, _tmp);

        _r6 = _mm256_sub_epi16(_r6, _s3);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r6);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_r6, _tmp);
#endif // Original_Q12289
        _r6 = _mm256_sub_epi16(_r6, _m2);
#ifdef Original_Q7681
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r6);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_r6, _tmp);
#endif // Original_Q7681
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r6);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_r6, _tmp);
#endif // Original_Q12289
        _r6 = _mm256_sub_epi16(_r6, _m6);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r6);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r6 = _mm256_sub_epi16(_r6, _tmp);

        //ab7[i] = (m07[i] - m0[i] - m7[i] + m16[i] - m1[i] - m6[i] + m25[i] - m2[i] - m5[i] + m34[i] - m3[i] - m4[i]) % q
        //       = (m07[i] + m16[i] + m25[i] + m34[i] - (m0[i] + m1[i]) - (m2[i] + m3[i]) - (m4[i] + m5[i]) - (m6[i] + m7[i])) % q;
        _r7 = _mm256_add_epi16(_m07, _m16);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r7);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r7 = _mm256_sub_epi16(_r7, _tmp);
#endif // Original_Q12289
        _r7 = _mm256_add_epi16(_r7, _m25);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r7);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r7 = _mm256_sub_epi16(_r7, _tmp);
#endif // Original_Q12289
        _r7 = _mm256_add_epi16(_r7, _m34);
#ifdef Original_Q7681
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r7);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r7 = _mm256_sub_epi16(_r7, _tmp);
#endif // Original_Q7681
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r7);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r7 = _mm256_sub_epi16(_r7, _tmp);
#endif // Original_Q12289
        _r7 = _mm256_sub_epi16(_r7, _s4);
#ifdef Original_Q12289
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r7);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r7 = _mm256_sub_epi16(_r7, _tmp);
#endif // Original_Q12289
        _r7 = _mm256_sub_epi16(_r7, _s1);
        //Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _r7);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _r7 = _mm256_sub_epi16(_r7, _tmp);
    /************************************************************************/
        _mm256_store_si256((__m256i*)(s0 + i), _r0);
        _mm256_store_si256((__m256i*)(s1 + i), _r1);
        _mm256_store_si256((__m256i*)(s2 + i), _r2);
        _mm256_store_si256((__m256i*)(s3 + i), _r3);
        _mm256_store_si256((__m256i*)(s4 + i), _r4);
        _mm256_store_si256((__m256i*)(s5 + i), _r5);
        _mm256_store_si256((__m256i*)(s6 + i), _r6);
        _mm256_store_si256((__m256i*)(s7 + i), _r7);
    }
    /**/
    invntt(s0);
    invntt(s1);
    invntt(s2);
    invntt(s3);
    invntt(s4);
    invntt(s5);
    invntt(s6);
    invntt(s7);
}