#include <stdio.h>
#include <time.h>
#include "mps.h"
#include "mpo.h"

#include "thcmpo.h"
#include "gmap.h"

void read_data(double* zeta, double *chi)
{
    hid_t file = H5Fopen("../examples/thcmpo/data/water.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		printf("'H5Fopen' failed\n");
	}

    if (read_hdf5_dataset(file, "zeta", H5T_NATIVE_DOUBLE, zeta) < 0) {
        printf("can not read zeta\n");
	}

    if (read_hdf5_dataset(file, "chi", H5T_NATIVE_DOUBLE, chi) < 0) {
        printf("can not read chi\n");
	}
}


void read_validation_vector(double *phi)
{
    hid_t file = H5Fopen("../examples/thcmpo/data/validation.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		printf("'H5Fopen' failed\n");
	}

    if (read_hdf5_dataset(file, "phi", H5T_NATIVE_DOUBLE, phi) < 0) {
        printf("can not read zeta\n");
    }
}


void validate(struct mps *phi) 
{
    struct block_sparse_tensor phi_comp_bst;
    struct dense_tensor phi_comp;
    struct dense_tensor phi_val;
    
    const long phi_dim[3] = { 1, 16384, 1 };
    allocate_dense_tensor(CT_DOUBLE_REAL, 3, phi_dim, &phi_val);
    read_validation_vector((double*)phi_val.data);

    mps_to_statevector(phi, &phi_comp_bst);
    block_sparse_to_dense_tensor(&phi_comp_bst, &phi_comp);

    assert(dense_tensor_allclose(&phi_val, &phi_comp, 1e-13));
}

int main()
{
    const long N = 28;
    const long L = 14;

    // ζ
    struct dense_tensor zeta;
    const long zeta_dim[2] = { N, N };
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, zeta_dim, &zeta);

    // χ
    struct dense_tensor chi;
    const long chi_dim[2] = { N, L / 2 }; // #spin_orbitals = L / 2
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, chi_dim, &chi);

    read_data((double*)zeta.data, (double*)chi.data);

    // G_{nu, sigma}
    struct gmap gmap;
    construct_gmap(chi, N, L, &gmap);

    // hartree fock state
    struct mps psi;
    construct_computational_basis_mps_2d(L, 0b11111111110000, &psi);

    // phi
    struct mps phi;
    clock_t start = clock();
    compute_phi(&psi, &gmap, zeta, N, 1e-3, LONG_MAX, &phi);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("compute_phi[duration]=%fs\n", time_spent);

    validate(&phi);

    delete_mps(&psi);
    delete_mps(&phi);
    delete_dense_tensor(&chi);
    delete_dense_tensor(&zeta);
}