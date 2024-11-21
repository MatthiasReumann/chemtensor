#include <stdio.h>
#include <time.h>
#include "mps.h"
#include "mpo.h"

#include "utils.h"
#include "states.h"
#include "thcmpo.h"

void read_data(double *zeta, double *chi)
{
    hid_t file = H5Fopen("../examples/thcmpo/data/water.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
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

void read_ref_vector(double *phi)
{
    hid_t file = H5Fopen("../examples/thcmpo/data/ref4d.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
    {
        printf("'H5Fopen' failed\n");
    }

    if (read_hdf5_dataset(file, "phi", H5T_NATIVE_DOUBLE, phi) < 0)
    {
        printf("can not read zeta\n");
    }
}

void validate(struct mps *phi)
{
    struct block_sparse_tensor phi_comp_bst;
    struct dense_tensor phi_comp;
    struct dense_tensor phi_val;

    const long phi_dim[3] = {1, 16384, 1};
    allocate_dense_tensor(CT_DOUBLE_REAL, 3, phi_dim, &phi_val);
    read_ref_vector((double *)phi_val.data);

    mps_to_statevector(phi, &phi_comp_bst);
    block_sparse_to_dense_tensor(&phi_comp_bst, &phi_comp);

    assert(dense_tensor_allclose(&phi_val, &phi_comp, 1e-13));
}

int main()
{
    const long N = 28;
    const long L = 7;

    // ζ
    struct dense_tensor zeta;
    const long zeta_dim[2] = {N, N};
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, zeta_dim, &zeta);

    // χ
    struct dense_tensor chi;
    const long chi_dim[2] = {N, L};
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, chi_dim, &chi);

    read_data((double *)zeta.data, (double *)chi.data);

    // G_{nu, sigma}
    struct mpo **g;
    g = ct_malloc(2 * N * sizeof(struct mpo *));
    for (size_t i = 0; i < 2 * N; i++)
    {
        g[i] = ct_malloc(2 * sizeof(struct mpo));
    }
    construct_gmap_4d(chi, N, L, g);

    // struct mps psi;
    // const unsigned basis_state[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
    // construct_computational_basis_mps(L, basis_state, &psi);

    // hartree fock state
    struct mps hfs;
    const unsigned spin_state[] = {3, 3, 3, 3, 3, 0, 0};
    construct_spin_basis_mps(L, spin_state, &hfs);

    // phi
    struct mps phi;
    clock_t start = clock();

    compute_phi(&hfs, g, zeta, N, 1e-3, LONG_MAX, &phi);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("compute_phi[duration]=%fs\n", time_spent);

    validate(&phi);

    delete_mps(&phi);
    delete_mps(&hfs);
    delete_dense_tensor(&chi);
    delete_dense_tensor(&zeta);
}