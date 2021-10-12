#include <stdio.h>
#include <math.h>
#include "params.h"

extern int wn[Pt3_N];
extern int wn_chosen[Pt3_N];
extern int invwn_chosen[Pt3_N];
extern int wn_chosen_brv[Pt3_N];
extern int invwn_chosen_brv[Pt3_N];
extern int zetas[Pt3_N];
extern int inv_zetas[Pt3_N];

int quick_exp_mod(int a, int b, int m);
void findw();
int mod_invese(int d, int n);
void table_wn_chosen(int a);
void table_invwn_chosen(int a);
void table_wn_chosen_brv();
void table_invwn_chosen_brv();
void table_zetas();
void table_inv_zetas();