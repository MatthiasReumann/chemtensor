#include <stdio.h>
#include "aligned_memory.h"
#include "mpo.h"


struct g_pair {
    struct mpo p;
    struct mpo q;
};


struct g {
    struct g_pair* fake_dict;  // List of MPO pairs accessible like an dictionary with tuple key via get_value(...)
    long N;               // THC Rank `N`. Length of `fake_dict` is `2N`.
};


void allocate_g(struct g* g, const long N)
{
    assert(N > 0);
    g->N = N;
    g->fake_dict = ct_malloc(2 * N * sizeof(struct g_pair));
}


struct g_pair* get_value(const struct g* g, const int i, const int s)
{
    assert(i + g->N * s < 2 * g->N);
    return &(g->fake_dict[i + g->N * s]);
}


void construct_elementary_mpo(const int L, const dcomplex* chi_row, const bool is_creation, struct mpo* mpo)
{
    const long d = 2;
    const qnumber qd[2] = {0, 1};

    mpo->d = d;
    mpo->nsites = L;
    mpo->qsite = ct_malloc(d * sizeof(qnumber));
    memcpy(mpo->qsite, &qd, d * sizeof(qnumber));

    mpo->a = ct_calloc(L, sizeof(struct block_sparse_tensor));
    
    const enum tensor_axis_direction axis_dir[4] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN, TENSOR_AXIS_IN };
    
    struct dense_tensor dt;

    // first site. dimensions: 1 x d x d x 2
    {
        const long dim[4] = { 1, d, d, 2 };
        allocate_dense_tensor(CT_DOUBLE_COMPLEX, 4, dim, &dt);
        
        if (is_creation) {
            const qnumber qD[1] = { 0 };
            const qnumber qD_next[2] = { 0, 1 };
            const qnumber* qnums[4] = { qD, qd, qd, qD_next};
            const dcomplex data[8] = { 1, 0, 0, 0, 0, chi_row[0], -1, 0};
            
            memcpy(dt.data, &data, 8 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[0]);
        } else {
            const qnumber qD[1] = { 0 };
            const qnumber qD_next[2] = { 0, -1 };
            const qnumber* qnums[4] = { qD, qd, qd, qD_next};
            const dcomplex data[8] = { 1, 0, 0, chi_row[0], 0, 0, -1, 0};
            
            memcpy(dt.data, &data, 8 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[0]);
        }
    }

    // intermediate sites: dimensions: 2 x d x d x 2
    for (int i = 1; i < L - 1; i++)
	{
        const long dim[4] = { 2, d, d, 2 };
        allocate_dense_tensor(CT_DOUBLE_COMPLEX, 4, dim, &dt);

        if (is_creation) {
            const qnumber qD[2] = { 0, 1 };
		    const qnumber* qnums[4] = { qD, qd, qd, qD };
            const dcomplex data[16] = { 1, 0, 0, 0, 0, chi_row[i], -1, 0, 0, 1, 0, 0, 0, 0, 0, 1 };
            memcpy(dt.data, data, 16 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[i]);

        } else {
            const qnumber qD[2] = { 0, -1 };
		    const qnumber* qnums[4] = { qD, qd, qd, qD };
            const dcomplex data[16] = { 1, 0, 0, chi_row[i], 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1 };
            
            memcpy(dt.data, data, 16 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[i]);
        }
	}

    // last site. dimensions: 2 x d x d x 1
    {
        const long dim[4] = { 2, d, d, 1 };
        allocate_dense_tensor(CT_DOUBLE_COMPLEX, 4, dim, &dt);

        if (is_creation) {
            const qnumber qD[2] = { 0, 1 };
            const qnumber qD_next[1] = { 1 };
            const qnumber* qnums[4] = { qD, qd, qd, qD_next};
            const dcomplex data[8] = { 0, 0, chi_row[L - 1], 0, 1, 0, 0, 1};

            memcpy(dt.data, data, 8 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[L - 1]);
        } else {
            const qnumber qD[2] = { 0, -1 };
            const qnumber qD_next[1] = { -1 };
            const qnumber* qnums[4] = { qD, qd, qd, qD_next};
            const dcomplex data[8] = { 0, chi_row[L - 1], 0, 0, 1, 0, 0, 1};

            memcpy(dt.data, data, 8 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[L - 1]);
        }
    }
}


void interleave_zero(const dcomplex* a, const long n, const long offset, dcomplex** ret)
{
    *ret = ct_calloc(2 * n, sizeof(dcomplex));
    for (int i = 0; i < n; i++) {
        (*ret)[offset + 2 * i] = a[i];
    }
}


void construct_g(dcomplex** chi, const long N, const long L, struct g* g){
    
    // chi.shape = (N, L/2)
    dcomplex* chi_row;   
    
    allocate_g(g, N);
    for (int i = 0; i < N; i++) {
        // spin up
        {
            struct g_pair* pair = get_value(g, i, 0);   
            interleave_zero(chi[i], L / 2, 0, &chi_row);
            construct_elementary_mpo(L, chi_row, false, &pair->p);
            construct_elementary_mpo(L, chi_row, true, &pair->q);
        }

        // spin down
        {
            struct g_pair* pair = get_value(g, i, 1);
            interleave_zero(chi[i], L / 2, 1, &chi_row);
            construct_elementary_mpo(L, chi_row, false, &pair->p);
            construct_elementary_mpo(L, chi_row, true, &pair->q);
        }
    }
}


void print_mpo(struct mpo mpo)
{
    printf("MPO: L=%d, d=%ld, qsite=(%d, %d)\n", mpo.nsites, mpo.d, mpo.qsite[0], mpo.qsite[1]);

    for(int i = 0; i < mpo.nsites; i++) {
        struct dense_tensor dt;
        block_sparse_to_dense_tensor(&mpo.a[i], &dt);

        dcomplex* data = (dcomplex*)(dt.data);
        const long nblocks = integer_product(dt.dim, dt.ndim);
        printf("[");
        for (long k = 0; k < nblocks; k++)
		{
			printf(" %.2f", creal(data[k]));
		}
        printf("]\n");
    }
}


int main()
{
    const long N = 4;
    const long L = 6;

    const dcomplex lst[N][L/2] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
    };

    dcomplex** chi = ct_malloc(N * sizeof(dcomplex*));
    for (int i = 0; i < N; i++){
        chi[i] = ct_malloc(L/2 * sizeof(dcomplex));
        for (int j = 0; j < L/2; j++) {
            chi[i][j] = lst[i][j];
        }
    }
    
    struct g g;
    construct_g(chi, N, L, &g);
    struct g_pair* pair_ret = get_value(&g, 3, 1);
    print_mpo(pair_ret->p);
    print_mpo(pair_ret->q);

    // struct mpo p, q;
    // const int nsites = 3;
    // const dcomplex chi_row[3] = { 42, 1337, 72 };
    // construct_elementary_mpo(nsites, chi_row, true, &p);
    // construct_elementary_mpo(nsites, chi_row, false, &q);
    // allocate_g(&g, 7);
    // struct g_pair pair = { .p = &p, .q = &q };
    // set_value(&g, 0, 0, &pair);
    // struct g_pair* pair_ret = get_value(&g, 0, 0);
    // print_mpo(*pair_ret->p);
    // print_mpo(*pair_ret->q);
}