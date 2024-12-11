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

void read_water(double *zeta, double *chi, double *H, double *tkin)
{
    hid_t file = H5Fopen("../examples/thcmpo/data/water.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    
    assert(file >= 0);
    assert(read_hdf5_dataset(file, "zeta", H5T_NATIVE_DOUBLE, zeta) >= 0);
    assert(read_hdf5_dataset(file, "chi", H5T_NATIVE_DOUBLE, chi) >= 0);

    if (H != NULL)
    {
        assert(read_hdf5_dataset(file, "H", H5T_NATIVE_DOUBLE, H) >= 0);
    }

    if (tkin != NULL)
    {
        assert(read_hdf5_dataset(file, "tkin", H5T_NATIVE_DOUBLE, tkin) >= 0);
    }
}