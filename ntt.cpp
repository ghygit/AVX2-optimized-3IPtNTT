#include <stdio.h>
#include <math.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "params.h"
#include "brv.h"
#include "getw.h"

extern int invn2;
extern int invq;
extern int n2modq;
extern int vv;

void ntt(short *p)
{
    short level, start, h, i, j, jl, k, zn, n;
    short zn15, zn14, zn13, zn12, zn11, zn10, zn9, zn8, zn7, zn6, zn5, zn4, zn3, zn2, zn1, zn0, 
        jl2, jl3, jl4, jl5, jl6, jl7, jl8, jl9, jl10, jl11, jl12, jl13, jl14, jl15,
        jl16, jl17, jl18, jl19, jl20, jl21, jl22, jl23, jl24, jl25, jl26, jl27, jl28, jl29, jl30, jl31;
    __m256i _p, _p2, z_p, z_n, modq, modinvq, _p_t, _p2_t, _p_t_t, _p2_t_t, _tmp, _tmp1;
    __m256i z_ph, z_pl, _VV;

    _VV = _mm256_set1_epi16(vv);
    n = Pt3_N;
    k = 1;
    modq = _mm256_set1_epi16(Original_Q);
    modinvq= _mm256_set1_epi16(invq);
    for (level = n / 2; level >= 16; level = level >> 1)
    {
        for (start = 0; start < n; start = j + level)
        {
            zn = zetas[k++];
            z_n = _mm256_set1_epi16(zn);

            for (j = start; j < start + level; j += 16)
            {
                jl = j + level;
                _p2 = _mm256_load_si256((__m256i*)(p + jl));
                _p = _mm256_load_si256((__m256i*)(p + j));
                //NTT==========================================
                //Montgomery red
                z_pl = _mm256_mullo_epi16(_p2, z_n);
                z_ph = _mm256_mulhi_epi16(_p2, z_n);
                _tmp = _mm256_mullo_epi16(z_pl, modinvq);
                _tmp = _mm256_mulhi_epi16(_tmp, modq);
                z_p = _mm256_sub_epi16(z_ph, _tmp);

                _p2 = _mm256_sub_epi16(_p, z_p);
                _p = _mm256_add_epi16(_p, z_p);
                //Barret red
                /**/
                _tmp = _mm256_mulhi_epi16(_VV, _p);
                _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
                _tmp = _mm256_mullo_epi16(_tmp, modq);
                _p = _mm256_sub_epi16(_p, _tmp);

                _tmp = _mm256_mulhi_epi16(_VV, _p2);
                _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
                _tmp = _mm256_mullo_epi16(_tmp, modq);
                _p2 = _mm256_sub_epi16(_p2, _tmp);
                //==================================================
                _mm256_store_si256((__m256i*)(p + j), _p);
                _mm256_store_si256((__m256i*)(p + jl), _p2);
            }
        }
    }
    //4th floor
    for (start = 0; start < n; start = j + level)
    {
        zn0 = zetas[k++];
        zn1 = zetas[k++];
        z_n = _mm256_set_epi16(zn1, zn1, zn1, zn1, zn1, zn1, zn1, zn1, zn0, zn0, zn0, zn0, zn0, zn0, zn0, zn0);
        for (j = start; j < start + level; j += 24)
        {
            jl = j + level * 2;
            _p2 = _mm256_load_si256((__m256i*)(p + jl));
            _p = _mm256_load_si256((__m256i*)(p + j));
            _p_t = _mm256_permute4x64_epi64(_p, 0b11011000);
            _p2_t = _mm256_permute4x64_epi64(_p2, 0b11011000);
            _p2_t_t = _mm256_unpackhi_epi64(_p_t, _p2_t);
            _p_t_t = _mm256_unpacklo_epi64(_p_t, _p2_t);
            _p = _mm256_permute4x64_epi64(_p_t_t, 0b11011000);
            _p2 = _mm256_permute4x64_epi64(_p2_t_t, 0b11011000);
            /**/
            //NTT==========================================
            //Montgomery red
            z_pl = _mm256_mullo_epi16(_p2, z_n);
            z_ph = _mm256_mulhi_epi16(_p2, z_n);
            _tmp = _mm256_mullo_epi16(z_pl, modinvq);
            _tmp = _mm256_mulhi_epi16(_tmp, modq);
            z_p = _mm256_sub_epi16(z_ph, _tmp);

            _p2 = _mm256_sub_epi16(_p, z_p);
            _p = _mm256_add_epi16(_p, z_p);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p = _mm256_sub_epi16(_p, _tmp);
            _tmp = _mm256_mulhi_epi16(_VV, _p2);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2, _tmp);
            //==================================================
            ////
            _mm256_store_si256((__m256i*)(p + j), _p);
            _mm256_store_si256((__m256i*)(p + jl), _p2);
        }
    }
    level = level >> 1;
    //3rd floor
    for (start = 0; start < n; start = j + level)
    {
        zn0 = zetas[k++];
        zn1 = zetas[k++];
        zn2 = zetas[k++];
        zn3 = zetas[k++];
        z_n = _mm256_set_epi16(zn3, zn3, zn3, zn3, zn2, zn2, zn2, zn2, zn1, zn1, zn1, zn1, zn0, zn0, zn0, zn0);
        for (j = start; j < start + level; j += 28)
        {
            jl = j + level * 4;
            _p2_t = _mm256_load_si256((__m256i*)(p + jl));
            _p_t = _mm256_load_si256((__m256i*)(p + j));
            _p2 = _mm256_unpackhi_epi64(_p_t, _p2_t);
            _p = _mm256_unpacklo_epi64(_p_t, _p2_t);
            //NTT==========================================
            //Montgomery red
            z_pl = _mm256_mullo_epi16(_p2, z_n);
            z_ph = _mm256_mulhi_epi16(_p2, z_n);
            _tmp = _mm256_mullo_epi16(z_pl, modinvq);
            _tmp = _mm256_mulhi_epi16(_tmp, modq);
            z_p = _mm256_sub_epi16(z_ph, _tmp);

            _p2 = _mm256_sub_epi16(_p, z_p);
            _p = _mm256_add_epi16(_p, z_p);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p = _mm256_sub_epi16(_p, _tmp);
            _tmp = _mm256_mulhi_epi16(_VV, _p2);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2, _tmp);
            //==================================================
            _mm256_store_si256((__m256i*)(p + j), _p);
            _mm256_store_si256((__m256i*)(p + jl), _p2);
        }
    }
    level = level >> 1;
    //2nd floor
    for (start = 0; start < n; start = j + level)
    {
        zn0 = zetas[k++]; zn1 = zetas[k++]; zn2 = zetas[k++]; zn3 = zetas[k++];
        zn4 = zetas[k++]; zn5 = zetas[k++]; zn6 = zetas[k++]; zn7 = zetas[k++];
        z_n = _mm256_set_epi16(zn7, zn7, zn6, zn6, zn5, zn5, zn4, zn4, zn3, zn3, zn2, zn2, zn1, zn1, zn0, zn0);
        for (j = start; j < start + level; j += 30)
        {
            jl = j + level * 8;
            _p2 = _mm256_load_si256((__m256i*)(p + jl));
            _p = _mm256_load_si256((__m256i*)(p + j));
            _p2_t = _mm256_unpackhi_epi32(_p, _p2);
            _p_t = _mm256_unpacklo_epi32(_p, _p2);
            _p2 = _mm256_unpackhi_epi64(_p_t, _p2_t);
            _p = _mm256_unpacklo_epi64(_p_t, _p2_t);
            //NTT==========================================
            //Montgomery red
            z_pl = _mm256_mullo_epi16(_p2, z_n);
            z_ph = _mm256_mulhi_epi16(_p2, z_n);
            _tmp = _mm256_mullo_epi16(z_pl, modinvq);
            _tmp = _mm256_mulhi_epi16(_tmp, modq);
            z_p = _mm256_sub_epi16(z_ph, _tmp);

            _p2 = _mm256_sub_epi16(_p, z_p);
            _p = _mm256_add_epi16(_p, z_p);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p = _mm256_sub_epi16(_p, _tmp);
            _tmp = _mm256_mulhi_epi16(_VV, _p2);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2, _tmp);
            //==================================================
            _mm256_store_si256((__m256i*)(p + j), _p);
            _mm256_store_si256((__m256i*)(p + jl), _p2);
        }
    }
    level = level >> 1;
    //1st floor
    for (start = 0; start < n; start = j + level)
    {
        zn0 = zetas[k++]; zn1 = zetas[k++]; zn2 = zetas[k++]; zn3 = zetas[k++];
        zn4 = zetas[k++]; zn5 = zetas[k++]; zn6 = zetas[k++]; zn7 = zetas[k++];
        zn8 = zetas[k++]; zn9 = zetas[k++]; zn10 = zetas[k++]; zn11 = zetas[k++];
        zn12 = zetas[k++]; zn13 = zetas[k++]; zn14 = zetas[k++]; zn15 = zetas[k++];
        z_n = _mm256_set_epi16(zn15, zn14, zn13, zn12, zn11, zn10, zn9, zn8, zn7, zn6, zn5, zn4, zn3, zn2, zn1, zn0);
        for (j = start; j < start + level; j += 31)
        {
            jl = j + level * 16;
            _p2_t = _mm256_load_si256((__m256i*)(p + jl));
            _p_t = _mm256_load_si256((__m256i*)(p + j));

            _p2 = _mm256_unpackhi_epi16(_p_t, _p2_t);
            _p = _mm256_unpacklo_epi16(_p_t, _p2_t);

            _p2_t = _mm256_unpackhi_epi32(_p, _p2);
            _p_t = _mm256_unpacklo_epi32(_p, _p2);

            _p2 = _mm256_unpackhi_epi32(_p_t, _p2_t);
            _p = _mm256_unpacklo_epi32(_p_t, _p2_t);
            //NTT==========================================
            //Montgomery red
            z_pl = _mm256_mullo_epi16(_p2, z_n);
            z_ph = _mm256_mulhi_epi16(_p2, z_n);
            _tmp = _mm256_mullo_epi16(z_pl, modinvq);
            _tmp = _mm256_mulhi_epi16(_tmp, modq);
            z_p = _mm256_sub_epi16(z_ph, _tmp);

            _p2 = _mm256_sub_epi16(_p, z_p);
            _p = _mm256_add_epi16(_p, z_p);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p = _mm256_sub_epi16(_p, _tmp);
            _tmp = _mm256_mulhi_epi16(_VV, _p2);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2, _tmp);
            //==================================================
            _mm256_store_si256((__m256i*)(p + j), _p);
            _mm256_store_si256((__m256i*)(p + jl), _p2);
        }
    }
}

void invntt(short *p)
{
    short level, start, i, j, jl, k, zn, n, q;
    short zn15, zn14, zn13, zn12, zn11, zn10, zn9, zn8, zn7, zn6, zn5, zn4, zn3, zn2, zn1, zn0,
        jl2, jl3, jl4, jl5, jl6, jl7, jl8, jl9, jl10, jl11, jl12, jl13, jl14, jl15,
        jl16, jl17, jl18, jl19, jl20, jl21, jl22, jl23, jl24, jl25, jl26, jl27, jl28, jl29, jl30, jl31;
    __m256i _p, _pl, _ph, _p2, _p2l, _p2h, _t0, _t0_t, _t0_t_t, modq, modinvq, modinvn2, _VV, z_n, _p_t, _p2_t, _p_t_t, _p2_t_t, _tmp, _tmp1, _1;

    n = Pt3_N;
    q = Original_Q;
    /**/
    modq = _mm256_set1_epi16(q);
    modinvq = _mm256_set1_epi16(invq);
    modinvn2 = _mm256_set1_epi16(invn2);
    _VV = _mm256_set1_epi16(vv);
    _1 = _mm256_set1_epi16(1);

    k = 0;
    //1st floor
    level = 1;
    for (start = 0; start < n; start = j + level)
    {
        zn0 = inv_zetas[k++]; zn1 = inv_zetas[k++]; zn2 = inv_zetas[k++]; zn3 = inv_zetas[k++];
        zn4 = inv_zetas[k++]; zn5 = inv_zetas[k++]; zn6 = inv_zetas[k++]; zn7 = inv_zetas[k++];
        zn8 = inv_zetas[k++]; zn9 = inv_zetas[k++]; zn10 = inv_zetas[k++]; zn11 = inv_zetas[k++];
        zn12 = inv_zetas[k++]; zn13 = inv_zetas[k++]; zn14 = inv_zetas[k++]; zn15 = inv_zetas[k++];
        z_n = _mm256_set_epi16(zn15, zn14, zn13, zn12, zn11, zn10, zn9, zn8, zn7, zn6, zn5, zn4, zn3, zn2, zn1, zn0);
        for (j = start; j < start + level; j += 31)
        {
            jl = j + level * 16;
            _p2 = _mm256_load_si256((__m256i*)(p + jl));
            _t0 = _mm256_load_si256((__m256i*)(p + j));
            //ntt=========================================
            _p = _mm256_add_epi16(_t0, _p2);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p = _mm256_sub_epi16(_p, _tmp);

            _p2 = _mm256_sub_epi16(_t0, _p2);

            //Montgomery red
            _p2l = _mm256_mullo_epi16(z_n, _p2);
            _p2h = _mm256_mulhi_epi16(z_n, _p2);
            _tmp = _mm256_mullo_epi16(_p2l, modinvq);
            _tmp = _mm256_mulhi_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2h, _tmp);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p2);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2, _tmp);
            //==================================================
            _mm256_store_si256((__m256i*)(p + j), _p);
            _mm256_store_si256((__m256i*)(p + jl), _p2);
        }
    }
    level <<= 1;
    //2nd floor
    for (start = 0; start < n; start = j + level)
    {
        zn0 = inv_zetas[k++]; zn1 = inv_zetas[k++]; zn2 = inv_zetas[k++]; zn3 = inv_zetas[k++];
        zn4 = inv_zetas[k++]; zn5 = inv_zetas[k++]; zn6 = inv_zetas[k++]; zn7 = inv_zetas[k++];
        z_n = _mm256_set_epi16(zn7, zn7, zn6, zn6, zn5, zn5, zn4, zn4, zn3, zn3, zn2, zn2, zn1, zn1, zn0, zn0);
        for (j = start; j < start + level; j += 30)
        {
            jl = j + level * 8;
            _p2_t = _mm256_load_si256((__m256i*)(p + jl));
            _t0_t = _mm256_load_si256((__m256i*)(p + j));

            _p2 = _mm256_unpackhi_epi16(_t0_t, _p2_t);
            _t0 = _mm256_unpacklo_epi16(_t0_t, _p2_t);

            _p2_t = _mm256_unpackhi_epi32(_t0, _p2);
            _t0_t = _mm256_unpacklo_epi32(_t0, _p2);

            _p2 = _mm256_unpackhi_epi32(_t0_t, _p2_t);
            _t0 = _mm256_unpacklo_epi32(_t0_t, _p2_t);
            //ntt=========================================
            _p = _mm256_add_epi16(_t0, _p2);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p = _mm256_sub_epi16(_p, _tmp);

            _p2 = _mm256_sub_epi16(_t0, _p2);

            //Montgomery red
            _p2l = _mm256_mullo_epi16(z_n, _p2);
            _p2h = _mm256_mulhi_epi16(z_n, _p2);
            _tmp = _mm256_mullo_epi16(_p2l, modinvq);
            _tmp = _mm256_mulhi_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2h, _tmp);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p2);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2, _tmp);
            //==================================================
            _mm256_store_si256((__m256i*)(p + j), _p);
            _mm256_store_si256((__m256i*)(p + jl), _p2);
        }
    }
    level <<= 1;
    /**/
    //3rd floor
    for (start = 0; start < n; start = j + level)
    {
        zn0 = inv_zetas[k++];
        zn1 = inv_zetas[k++];
        zn2 = inv_zetas[k++];
        zn3 = inv_zetas[k++];
        z_n = _mm256_set_epi16(zn3, zn3, zn3, zn3, zn2, zn2, zn2, zn2, zn1, zn1, zn1, zn1, zn0, zn0, zn0, zn0);
        for (j = start; j < start + level; j += 28)
        {
            jl = j + level * 4;
            _p2 = _mm256_load_si256((__m256i*)(p + jl));
            _t0 = _mm256_load_si256((__m256i*)(p + j));
            _p2_t = _mm256_unpackhi_epi32(_t0, _p2);
            _t0_t = _mm256_unpacklo_epi32(_t0, _p2);
            _p2 = _mm256_unpackhi_epi64(_t0_t, _p2_t);
            _t0 = _mm256_unpacklo_epi64(_t0_t, _p2_t);
            _p = _mm256_add_epi16(_t0, _p2);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p = _mm256_sub_epi16(_p, _tmp);

            _p2 = _mm256_sub_epi16(_t0, _p2);

            //Montgomery red
            _p2l = _mm256_mullo_epi16(z_n, _p2);
            _p2h = _mm256_mulhi_epi16(z_n, _p2);
            _tmp = _mm256_mullo_epi16(_p2l, modinvq);
            _tmp = _mm256_mulhi_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2h, _tmp);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p2);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2, _tmp);
            //==================================================
            _mm256_store_si256((__m256i*)(p + j), _p);
            _mm256_store_si256((__m256i*)(p + jl), _p2);
        }
    }
    level <<= 1;
    /**/
    //4th floor
    for (start = 0; start < n; start = j + level)
    {
        zn0 = inv_zetas[k++];
        zn1 = inv_zetas[k++];
        z_n = _mm256_set_epi16(zn1, zn1, zn1, zn1, zn1, zn1, zn1, zn1, zn0, zn0, zn0, zn0, zn0, zn0, zn0, zn0);
        for (j = start; j < start + level; j += 24)
        {
            jl = j + level * 2;
            _p2_t = _mm256_load_si256((__m256i*)(p + jl));
            _t0_t = _mm256_load_si256((__m256i*)(p + j));
            _p2 = _mm256_unpackhi_epi64(_t0_t, _p2_t);
            _t0 = _mm256_unpacklo_epi64(_t0_t, _p2_t);
            //ntt=========================================
            _p = _mm256_add_epi16(_t0, _p2);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p = _mm256_sub_epi16(_p, _tmp);
            
            _p2 = _mm256_sub_epi16(_t0, _p2);

            //Montgomery red
            _p2l = _mm256_mullo_epi16(z_n, _p2);
            _p2h = _mm256_mulhi_epi16(z_n, _p2);
            _tmp = _mm256_mullo_epi16(_p2l, modinvq);
            _tmp = _mm256_mulhi_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2h, _tmp);
            //Barret red
            _tmp = _mm256_mulhi_epi16(_VV, _p2);
            _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
            _tmp = _mm256_mullo_epi16(_tmp, modq);
            _p2 = _mm256_sub_epi16(_p2, _tmp);
            //==================================================
            _p_t = _mm256_permute4x64_epi64(_p, 0b11011000);
            _p2_t = _mm256_permute4x64_epi64(_p2, 0b11011000);
            _p2_t_t = _mm256_unpackhi_epi64(_p_t, _p2_t);
            _p_t_t = _mm256_unpacklo_epi64(_p_t, _p2_t);
            _p = _mm256_permute4x64_epi64(_p_t_t, 0b11011000);
            _p2 = _mm256_permute4x64_epi64(_p2_t_t, 0b11011000);

            _mm256_store_si256((__m256i*)(p + j), _p);
            _mm256_store_si256((__m256i*)(p + jl), _p2);
        }
    }
    level <<= 1;
    /**/
    for (; level < n; level = level << 1)
    {
        for (start = 0; start < n; start = j + level)
        {
            zn = inv_zetas[k++];
            z_n = _mm256_set1_epi16(zn);
            for (j = start; j < start + level; j += 16)
            {
                jl = j + level;
                _t0 = _mm256_load_si256((__m256i*)(p + j));
                _p2 = _mm256_load_si256((__m256i*)(p + jl));

                //ntt=========================================
                _p = _mm256_add_epi16(_t0, _p2);
                //Barret red
                _tmp = _mm256_mulhi_epi16(_VV, _p);
                _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
                _tmp = _mm256_mullo_epi16(_tmp, modq);
                _p = _mm256_sub_epi16(_p, _tmp);

                _p2 = _mm256_sub_epi16(_t0, _p2);

                //Montgomery red
                _p2l = _mm256_mullo_epi16(z_n, _p2);
                _p2h = _mm256_mulhi_epi16(z_n, _p2);
                _tmp = _mm256_mullo_epi16(_p2l, modinvq);
                _tmp = _mm256_mulhi_epi16(_tmp, modq);
                _p2 = _mm256_sub_epi16(_p2h, _tmp);
                //Barret red
                _tmp = _mm256_mulhi_epi16(_VV, _p2);
                _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
                _tmp = _mm256_mullo_epi16(_tmp, modq);
                _p2 = _mm256_sub_epi16(_p2, _tmp);
                //==================================================
                _mm256_store_si256((__m256i*)(p + j), _p);
                _mm256_store_si256((__m256i*)(p + jl), _p2);
            }
        }
    }
    for (i = 0; i < n; i += 16)
    {
        _p = _mm256_load_si256((__m256i*)(p + i));

        //Montgomery red
        _pl = _mm256_mullo_epi16(modinvn2, _p);
        _ph = _mm256_mulhi_epi16(modinvn2, _p);
        _tmp = _mm256_mullo_epi16(_pl, modinvq);
        _tmp = _mm256_mulhi_epi16(_tmp, modq);
        _p = _mm256_sub_epi16(_ph, _tmp);
        
        //turn - into +     Barret red
        _tmp = _mm256_mulhi_epi16(_VV, _p);
        _tmp = _mm256_srai_epi16(_tmp, Bit_mov);
        _tmp = _mm256_mullo_epi16(_tmp, modq);
        _p = _mm256_sub_epi16(_p, _tmp);

        _mm256_store_si256((__m256i*)(p + i), _p);
    }
}

void poly_mul(short *a, short *b, short *ab)
{
    int i, j;
    int temp1, temp2, temp3;

    for (i = 0; i < Original_N; i++)
    {
        temp1 = 0; temp2 = 0;
        for (j = 0; j < i + 1; j++)
            temp1 = (temp1 + (((int)a[j] * b[i - j]) % Original_Q)) % Original_Q;
        for (j = i + 1; j < Original_N; j++)
            temp2 = (temp2 + (((int)a[j] * b[Original_N + i - j]) % Original_Q)) % Original_Q;
        temp3 = (temp1 - temp2) % Original_Q;
        if (temp3 < 0)
            temp3 = Original_Q + temp3;
        ab[i] = temp3;
    }
}