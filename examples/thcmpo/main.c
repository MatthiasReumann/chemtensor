#include <stdio.h>
#include <time.h>
#include "mps.h"
#include "mpo.h"
#include "hamiltonian.h"

#include "utils.h"
#include "states.h"
#include "thcmpo.h"

void read_data(double *zeta, double *chi)
{
    hid_t file = H5Fopen("../examples/thcmpo/data/water_chi_zeta.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        printf("'H5Fopen' failed\n");
    }

    if (read_hdf5_dataset(file, "zeta", H5T_NATIVE_DOUBLE, zeta) < 0)
    {
        printf("can not read zeta\n");
    }

    if (read_hdf5_dataset(file, "chi", H5T_NATIVE_DOUBLE, chi) < 0)
    {
        printf("can not read chi\n");
    }
}

void read_hamiltonian(double *H)
{
    hid_t file = H5Fopen("../examples/thcmpo/data/water_h.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        printf("'H5Fopen' failed\n");
    }

    if (read_hdf5_dataset(file, "H", H5T_NATIVE_DOUBLE, H) < 0)
    {
        printf("can not read H\n");
    }
}

void read_kinetic(double *tkin)
{
    hid_t file = H5Fopen("../examples/thcmpo/data/water_t.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        printf("'H5Fopen' failed\n");
    }

    if (read_hdf5_dataset(file, "T", H5T_NATIVE_DOUBLE, tkin) < 0)
    {
        printf("can not read H\n");
    }
}


int main()
{
    // TODO: Create one large hdf5 dataset instead of multiple ones.

    const long N = 28;
    const long L = 7;

    const double TOL = 1e-20;
    const long MAX_VDIM = LONG_MAX;

    // ζ, χ
    struct dense_tensor zeta;
    struct dense_tensor chi;
    {
        const long zeta_dim[2] = {N, N};
        allocate_dense_tensor(CT_DOUBLE_REAL, 2, zeta_dim, &zeta);

        const long chi_dim[2] = {N, L};
        allocate_dense_tensor(CT_DOUBLE_REAL, 2, chi_dim, &chi);

        read_data((double *)zeta.data, (double *)chi.data);
    }

    // G_{nu, sigma}
    struct mpo **g;
    g = ct_malloc(2 * N * sizeof(struct mpo *));
    for (size_t i = 0; i < 2 * N; i++)
    {
        g[i] = ct_malloc(2 * sizeof(struct mpo));
    }
    construct_g_4d(chi, N, L, g);

    // water hamiltonian 'H'
    struct dense_tensor H;
    {
        const long dim[2] = {16384, 16384};
        allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &H);
        read_hamiltonian((double *)H.data);
    }

    // t_{pq}
    struct dense_tensor tkin;
    {
        const long dim[2] = {L, L};
        allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &tkin);
        read_kinetic((double *)tkin.data);
    }

    // hartree fock state
    struct mps hfs;
    {
        const unsigned spin_state[] = {3, 3, 3, 3, 3, 0, 0};
        construct_spin_basis_mps(L, spin_state, &hfs);
    }

    // hartree fock as dense tensor ~ vector
    struct dense_tensor hfs_vec;
    {
        struct block_sparse_tensor bst;
        mps_to_statevector(&hfs, &bst);
        block_sparse_to_dense_tensor(&bst, &hfs_vec);
        const long dim[] = {16384};
        reshape_dense_tensor(1, dim, &hfs_vec);
        delete_block_sparse_tensor(&bst);
    }

    struct dense_tensor h_hfs;
    {
        const int i_ax = 1;
        dense_tensor_multiply_axis(&H, i_ax, &hfs_vec, TENSOR_AXIS_RANGE_LEADING, &h_hfs);
    }

    struct mpo t_mpo;
    {
        struct dense_tensor vint;
        struct mpo_assembly assembly;
        const long dim[] = {L, L, L, L};
        allocate_dense_tensor(CT_DOUBLE_REAL, 4, dim, &vint);
        construct_spin_molecular_hamiltonian_mpo_assembly(&tkin, &vint, false, &assembly);
        mpo_from_assembly(&assembly, &t_mpo);
        delete_mpo_assembly(&assembly);
    }

    // v|ᴪ>
    struct mps v_psi;
    {
        clock_t start = clock();
        compute_phi(&hfs, g, zeta, N, TOL, MAX_VDIM, &v_psi);
        clock_t end = clock();

        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("compute v_psi [duration]=%fs\n", time_spent);
    }

    // t|ᴪ>
    struct mps t_psi;
    {
        clock_t start = clock();
        apply_and_compress(&hfs, &t_mpo, TOL, MAX_VDIM, &t_psi);
        clock_t end = clock();

        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("compute t_psi [duration]=%fs\n", time_spent);
    }

    // t|ᴪ> + v|ᴪ>
    struct mps h_psi;
    {
        clock_t start = clock();
        add_and_compress(&t_psi, &v_psi, TOL, MAX_VDIM, &h_psi);
        clock_t end = clock();

        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("compute h_psi [duration]=%fs\n", time_spent);
    }

    struct dense_tensor h_psi_vec;
    {
        struct block_sparse_tensor bst;
        mps_to_statevector(&h_psi, &bst);
        block_sparse_to_dense_tensor(&bst, &h_psi_vec);
        const long dim[] = {16384};
        reshape_dense_tensor(1, dim, &h_psi_vec);
        delete_block_sparse_tensor(&bst);
    }

    printf("norm1: %f\n", dense_tensor_norm2(&h_hfs));
    printf("norm2: %f\n", dense_tensor_norm2(&h_psi_vec));
    printf("close: %d\n", dense_tensor_allclose(&h_hfs, &h_psi_vec, 1e-6));

    // teardown
    delete_mps(&h_psi);
    delete_mps(&t_psi);
    delete_mps(&v_psi);
    delete_mpo(&t_mpo);
    delete_dense_tensor(&h_hfs);
    delete_dense_tensor(&hfs_vec);
    delete_mps(&hfs);
    delete_dense_tensor(&chi);
    delete_dense_tensor(&zeta);
}