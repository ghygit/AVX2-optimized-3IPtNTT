#include <stdio.h>
#include <math.h>
#include <time.h>
#include "params.h"
#include "getw.h"
#include "brv.h"

int countw = 0;

int quick_exp_mod(int a, int b, int m)
{
    int ans = 1;
    a %= m;
    while (b)
    {
        if (b & 1)
            ans = ans * a % m;
        b >>= 1;
        a = a * a % m;
    }
    return ans;
}

void findw()
{
    int w, i, q = Original_Q, n = Pt3_N;
    countw = 0;
    for (w = 2; w < q; w++)
    {
        for (i = 0; i <= n; i++)
        {
            if (quick_exp_mod(w, i, q) == q - 1)
                if (i != n) continue;
                else //printf("%4d,",wn[countw++] = w);
                    wn[countw++] = w;
        }
    }
}

int mod_invese(int d, int n)
{
    int a, b, q, r, u = 0, v = 1, t;
    a = n;
    b = d % n;
    while (b != 0)
    {
        q = a / b;
        r = a % b;
        a = b;
        b = r;
        t = v;
        v = u - q * v;
        u = t;
    }
    if (a != 1) return 0;
    return((u < 0) ? u + n : u);
}

void table_wn_chosen(int a)
{
    int i, q = Original_Q, n = Pt3_N;
    for (i = 0; i < n; i++)
        wn_chosen[i] = quick_exp_mod(a, i, q);
}

void table_invwn_chosen(int a)
{
    int i, q = Original_Q, n = Pt3_N;
    a = mod_invese(a, q);
    for (i = 0; i < n; i++)
        invwn_chosen[i] = quick_exp_mod(a, i, q) % q;
}

void table_wn_chosen_brv()
{
    int i, q = Original_Q, n = Pt3_N;
    for (i = 0; i < n; i++)
        wn_chosen_brv[i] = wn_chosen[brv[i]];
}

void table_zetas()
{
    int i, q = Original_Q, n = Pt3_N;
    for (i = 0; i < n; i++)
        zetas[i] = (wn_chosen_brv[i] << 16) % q;
}

void table_invwn_chosen_brv()
{
    int i, q = Original_Q, n = Pt3_N;
    for (i = 0; i < n; i++)
        invwn_chosen_brv[i] = invwn_chosen[brv[i] + 1];
}

void table_inv_zetas()
{
    int i, q = Original_Q, n = Pt3_N;
    for (i = 0; i < n; i++)
        inv_zetas[i] = (invwn_chosen_brv[i] << 16) % q;
}