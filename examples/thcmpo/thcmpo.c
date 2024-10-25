#include <stdio.h>
#include "mps.h"
#include "mpo.h"
#include "operation.h"
#include "aligned_memory.h"


void THCMPO_print_mpo(struct mpo mpo)
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


void THCMPO_print_bst(struct block_sparse_tensor bst)
{
    printf("bst=[ndim=%d, dim=(%ld, %ld, %ld)]\n", 
            bst.ndim, 
            bst.dim_blocks[0], bst.dim_blocks[1], bst.dim_blocks[2]);
}


void THCMPO_print_dt(struct dense_tensor dt)
{
    printf("dt=[ndim=%d, dim=(%ld, %ld, %ld)]\n",
        dt.ndim,
        dt.dim[0],dt.dim[1],dt.dim[2]);
}

struct g_pair {
    struct mpo p;
    struct mpo q;
};


struct g {
    struct g_pair* fake_dict;   // List of MPO pairs accessible like an dictionary with tuple key via get_value(...)
    long N;                     // THC Rank `N`. Length of `fake_dict` is `2N`.
};


void THCMPO_g_allocate(struct g* g, const long N)
{
    assert(N > 0);
    g->N = N;
    g->fake_dict = ct_malloc(2 * N * sizeof(struct g_pair));
}


void THCMPO_g_get(const struct g* g, const int i, const int s, struct g_pair** g_pair)
{
    assert(i + g->N * s < 2 * g->N);
    *g_pair = &(g->fake_dict[i + g->N * s]);
}


void THCMPO_construct_elementary_mpo(const int L, const dcomplex* chi_row, const bool is_creation, struct mpo* mpo)
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
    for (long i = 1; i < L - 1; i++)
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


void THCMPO_interleave_zero(const dcomplex* a, const long n, const long offset, dcomplex** ret)
{
    *ret = ct_calloc(2 * n, sizeof(dcomplex));
    for (long i = 0; i < n; i++) {
        (*ret)[offset + 2 * i] = a[i];
    }
}


void THCMPO_g_construct_from_chi(dcomplex** chi, const long N, const long L, struct g* g){
    
    // chi.shape = (N, L/2)
    dcomplex* chi_row;
    
    THCMPO_g_allocate(g, N);
    for (long i = 0; i < N; i++) {
        // spin up
        {
            struct g_pair* pair;
            THCMPO_g_get(g, i, 0, &pair);   
            THCMPO_interleave_zero(chi[i], L / 2, 0, &chi_row);
            THCMPO_construct_elementary_mpo(L, chi_row, false, &pair->p);
            THCMPO_construct_elementary_mpo(L, chi_row, true, &pair->q);
        }

        // spin down
        {
            struct g_pair* pair;
            THCMPO_g_get(g, i, 1, &pair);
            THCMPO_interleave_zero(chi[i], L / 2, 1, &chi_row);
            THCMPO_construct_elementary_mpo(L, chi_row, false, &pair->p);
            THCMPO_construct_elementary_mpo(L, chi_row, true, &pair->q);
        }
    }
}


void THCMPO_calculate_phi(const struct mps *psi, struct g* g, const long N, const double tol, struct mps* phi)
{
    const int S = 2; // |{UP, DOWN}| = 2
    for(long n = 0; n < N; n++) {
        for(int s1 = 0; s1 < S; s1++) {
            
            struct mps a;
            struct mps b;
            struct g_pair* pair_n_s1;
            THCMPO_g_get(g, n, s1, &pair_n_s1);


            apply_operator(&pair_n_s1->p, psi, &a);
            // TODO: Compress
            apply_operator(&pair_n_s1->q, &a, &b);
            // TODO: Compress

            for(long m = 0; m < N; m++) {
                for(int s2 = 0; s2 < S; s2++) {
                    struct mps c;
                    struct mps d;
                    struct g_pair* pair_m_s2;
                    THCMPO_g_get(g, m, s2, &pair_m_s2);

                    apply_operator(&pair_m_s2->p, &b, &c);
                    // TODO: Compress
                    apply_operator(&pair_m_s2->q, &c, &d);
                    // TODO: Compress
                }
            }
        }
    }
}


void construct_computational_basis_mps(
    const enum numeric_type dtype,
    const int nsites,
    const int basis_state,
    struct mps* mps
) 
{
    const long d = 2;
    const qnumber qsite[2] = { 0, 1 };
    
    const dcomplex state_zero[2] = { 1, 0 };
    const dcomplex state_one[2] = { 0, 1 };

    const int ndim = 3;
	const long dim[3] = { 1, d, 1 };

    const enum tensor_axis_direction axis_dir[3] = { 
        TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN
    };

    int acc = 0;
    
    qnumber qbond_curr[1] = { acc };
    qnumber qbond_next[1];

    allocate_empty_mps(nsites, d, qsite, mps);
    
    for (int i = 0; i < nsites; i++) {        
        const int ith = ((basis_state & (1 << (nsites - i - 1))) >> (nsites - i - 1));

        struct dense_tensor dt;
        allocate_dense_tensor(dtype, ndim, dim, &dt);
    
        if (ith == 0) {
            memcpy(dt.data, &state_zero, 2 * sizeof(dcomplex));
        } else if (ith == 1) {
            memcpy(dt.data, &state_one, 2 * sizeof(dcomplex));
            acc++;
        }
        qbond_next[0] = acc;
        
        const qnumber* qnums[3] = { qbond_curr, qsite, qbond_next };
        dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mps->a[i]);

        delete_dense_tensor(&dt);

        qbond_curr[0] = qbond_next[0];
	}
}

int main()
{
    const long N = 28;
    const long L = 14;

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
            chi[i][j] = 1.3f;
        }
    }
    
    struct g g;
    struct mps psi;
    struct mps phi;
    THCMPO_g_construct_from_chi(chi, N, L, &g);
    construct_computational_basis_mps(CT_DOUBLE_COMPLEX, L, 0b11111111110000, &psi);

    struct dense_tensor dt;
    struct block_sparse_tensor bst;
    mps_to_statevector(&psi, &bst);

    block_sparse_to_dense_tensor(&bst, &dt);

    THCMPO_print_dt(dt);
}