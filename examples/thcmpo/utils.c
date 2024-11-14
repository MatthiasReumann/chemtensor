#include "utils.h"

void print_dt(struct dense_tensor dt)
{
    printf("dt=[dtype=%d, ndim=%d, dim=(", dt.dtype, dt.ndim);
    for(int i = 0; i < dt.ndim; i++) {
        printf(" %ld ", dt.dim[i]);
    }
    printf(")]\n");
}

void print_mpo(struct mpo mpo)
{
    printf("MPO: L=%d, d=%ld, qsite=(%d, %d)\n", mpo.nsites, mpo.d, mpo.qsite[0], mpo.qsite[1]);

    for (int i = 0; i < mpo.nsites; i++)
    {
        struct dense_tensor dt;
        block_sparse_to_dense_tensor(&mpo.a[i], &dt);

        // for(int k = 0; k < mpo.a[i].ndim; k++) {
        //     for (int j = 0; j < mpo.a[i].dim_blocks[k]; j++) {
        //         printf(" %d ", mpo.a[i].qnums_blocks[k][j]);
        //     }
        // }

        double *data = (double *)(dt.data);
        // print_dt(dt);
        const long nblocks = integer_product(dt.dim, dt.ndim);
        printf("[");
        for (long k = 0; k < nblocks; k++)
        {
            printf(" %.1f ", data[k]);
        }
        printf("]\n");
    }
}

void print_bst(struct block_sparse_tensor bst)
{
    printf("bst=[ndim=%d, dim=(%ld, %ld, %ld)]\n",
           bst.ndim,
           bst.dim_blocks[0], bst.dim_blocks[1], bst.dim_blocks[2]);
}