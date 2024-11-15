#include "utils.h"

void print_dt(struct dense_tensor dt)
{
    printf("dt=[dtype=%d, ndim=%d, dim=(", dt.dtype, dt.ndim);
    for (int i = 0; i < dt.ndim; i++)
    {
        printf(" %ld ", dt.dim[i]);
    }
    printf(")]\n");
}

void print_bst(struct block_sparse_tensor bst)
{
    printf("bst=[ndim=%d, dim=(%ld, %ld, %ld)]\n",
           bst.ndim,
           bst.dim_blocks[0], bst.dim_blocks[1], bst.dim_blocks[2]);
}

void print_mpo(struct mpo mpo)
{
    printf("MPO: L=%d, d=%ld, qsite=(", mpo.nsites, mpo.d);

    for (size_t i = 0; i < mpo.d - 1; i++)
    {
        printf("%d, ", mpo.qsite[i]);
    }
    printf("%d)\n", mpo.qsite[mpo.d - 1]);

    for (size_t i = 0; i < mpo.nsites; i++)
    {
        struct dense_tensor dt;
        block_sparse_to_dense_tensor(&mpo.a[i], &dt);

        printf("%zu: ", i);
        print_dt(dt);
    }
}

void print_mps(struct mps mps)
{
    printf("MPS: L=%d, d=%ld, qsite=(", mps.nsites, mps.d);

    for (size_t i = 0; i < mps.d - 1; i++)
    {
        printf("%d, ", mps.qsite[i]);
    }
    printf("%d)\n", mps.qsite[mps.d - 1]);

    for (size_t i = 0; i < mps.nsites; i++)
    {
        struct dense_tensor dt;
        block_sparse_to_dense_tensor(&mps.a[i], &dt);

        printf("%zu: ", i);
        print_dt(dt);
    }
}