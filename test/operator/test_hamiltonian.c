#include "hamiltonian.h"


char* test_ising_1d_mpo()
{
	hid_t file = H5Fopen("../test/operator/data/test_ising_1d_mpo.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_ising_1d_mpo failed";
	}

	// number of lattice sites
	const int nsites = 7;

	// Hamiltonian parameters
	const double J =  5./11;
	const double h = -2./7;
	const double g = 13./8;

	struct mpo ising_1d_mpo;
	construct_ising_1d_mpo(nsites, J, h, g, &ising_1d_mpo);
	if (!mpo_is_consistent(&ising_1d_mpo)) {
		return "internal consistency check for Ising Hamiltonian MPO failed";
	}

	for (int i = 0; i <= nsites; i++) {
		const long bdim_ref = (i == 0 || i == nsites ? 1 : 3);
		if (mpo_bond_dim(&ising_1d_mpo, i) != bdim_ref) {
			return "virtual bond dimension of MPO representation of Ising Hamiltonian does not match reference";
		}
	}

	struct block_sparse_tensor ising_1d_mat;
	mpo_to_matrix(&ising_1d_mpo, &ising_1d_mat);

	// convert to dense tensor for comparison with reference
	struct dense_tensor ising_1d_mat_dns;
	block_sparse_to_dense_tensor(&ising_1d_mat, &ising_1d_mat_dns);

	// reference matrix for checking
	hsize_t dims_ref_hsize[2];
	if (get_hdf5_dataset_dims(file, "ising_1d_mat", dims_ref_hsize) < 0) {
		return "obtaining dimensions of reference Hamiltonian failed";
	}
	const long dim_ref[4] = { 1, dims_ref_hsize[0], dims_ref_hsize[1], 1 };  // include dummy virtual bond dimensions
	struct dense_tensor ising_1d_mat_ref;
	allocate_dense_tensor(DOUBLE_REAL, 4, dim_ref, &ising_1d_mat_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "ising_1d_mat", H5T_NATIVE_DOUBLE, ising_1d_mat_ref.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&ising_1d_mat_dns, &ising_1d_mat_ref, 1e-13)) {
		return "matrix representation of Ising Hamiltonian based on MPO form does not match reference";
	}

	// clean up
	delete_block_sparse_tensor(&ising_1d_mat);
	delete_dense_tensor(&ising_1d_mat_dns);
	delete_dense_tensor(&ising_1d_mat_ref);
	delete_mpo(&ising_1d_mpo);

	H5Fclose(file);

	return 0;
}


char* test_heisenberg_xxz_1d_mpo()
{
	hid_t file = H5Fopen("../test/operator/data/test_heisenberg_xxz_1d_mpo.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_heisenberg_xxz_1d_mpo failed";
	}

	// number of lattice sites
	const int nsites = 7;

	// Hamiltonian parameters
	const double J = 14./25;
	const double D = 13./8;
	const double h =  2./7;

	struct mpo heisenberg_xxz_1d_mpo;
	construct_heisenberg_xxz_1d_mpo(nsites, J, D, h, &heisenberg_xxz_1d_mpo);
	if (!mpo_is_consistent(&heisenberg_xxz_1d_mpo)) {
		return "internal consistency check for Heisenberg XXZ Hamiltonian MPO failed";
	}

	for (int i = 0; i <= nsites; i++) {
		const long bdim_ref = (i == 0 || i == nsites ? 1 : (i == 1 || i == nsites - 1 ? 4 : 5));
		if (mpo_bond_dim(&heisenberg_xxz_1d_mpo, i) != bdim_ref) {
			return "virtual bond dimension of MPO representation of Heisenberg XXZ Hamiltonian does not match reference";
		}
	}

	struct block_sparse_tensor heisenberg_xxz_1d_mat;
	mpo_to_matrix(&heisenberg_xxz_1d_mpo, &heisenberg_xxz_1d_mat);

	// convert to dense tensor for comparison with reference
	struct dense_tensor heisenberg_xxz_1d_mat_dns;
	block_sparse_to_dense_tensor(&heisenberg_xxz_1d_mat, &heisenberg_xxz_1d_mat_dns);

	// reference matrix for checking
	hsize_t dims_ref_hsize[2];
	if (get_hdf5_dataset_dims(file, "heisenberg_xxz_1d_mat", dims_ref_hsize) < 0) {
		return "obtaining dimensions of reference Hamiltonian failed";
	}
	const long dim_ref[4] = { 1, dims_ref_hsize[0], dims_ref_hsize[1], 1 };  // include dummy virtual bond dimensions
	struct dense_tensor heisenberg_xxz_1d_mat_ref;
	allocate_dense_tensor(DOUBLE_REAL, 4, dim_ref, &heisenberg_xxz_1d_mat_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "heisenberg_xxz_1d_mat", H5T_NATIVE_DOUBLE, heisenberg_xxz_1d_mat_ref.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&heisenberg_xxz_1d_mat_dns, &heisenberg_xxz_1d_mat_ref, 1e-13)) {
		return "matrix representation of Heisenberg XXZ Hamiltonian based on MPO form does not match reference";
	}

	// clean up
	delete_block_sparse_tensor(&heisenberg_xxz_1d_mat);
	delete_dense_tensor(&heisenberg_xxz_1d_mat_dns);
	delete_dense_tensor(&heisenberg_xxz_1d_mat_ref);
	delete_mpo(&heisenberg_xxz_1d_mpo);

	H5Fclose(file);

	return 0;
}


char* test_bose_hubbard_1d_mpo()
{
	hid_t file = H5Fopen("../test/operator/data/test_bose_hubbard_1d_mpo.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_bose_hubbard_1d_mpo failed";
	}

	// number of lattice sites
	const int nsites = 5;

	// physical dimension per site (maximal occupancy is d - 1)
	const long d = 3;

	// Hamiltonian parameters
	const double t  =  7./10;
	const double u  = 17./4;
	const double mu = 13./11;

	struct mpo bose_hubbard_1d_mpo;
	construct_bose_hubbard_1d_mpo(nsites, d, t, u, mu, &bose_hubbard_1d_mpo);
	if (!mpo_is_consistent(&bose_hubbard_1d_mpo)) {
		return "internal consistency check for Bose-Hubbard Hamiltonian MPO failed";
	}

	for (int i = 0; i <= nsites; i++) {
		const long bdim_ref = (i == 0 || i == nsites ? 1 : 4);
		if (mpo_bond_dim(&bose_hubbard_1d_mpo, i) != bdim_ref) {
			return "virtual bond dimension of MPO representation of Bose-Hubbard Hamiltonian does not match reference";
		}
	}

	struct block_sparse_tensor bose_hubbard_1d_mat;
	mpo_to_matrix(&bose_hubbard_1d_mpo, &bose_hubbard_1d_mat);

	// convert to dense tensor for comparison with reference
	struct dense_tensor bose_hubbard_1d_mat_dns;
	block_sparse_to_dense_tensor(&bose_hubbard_1d_mat, &bose_hubbard_1d_mat_dns);

	// reference matrix for checking
	hsize_t dims_ref_hsize[2];
	if (get_hdf5_dataset_dims(file, "bose_hubbard_1d_mat", dims_ref_hsize) < 0) {
		return "obtaining dimensions of reference Hamiltonian failed";
	}
	const long dim_ref[4] = { 1, dims_ref_hsize[0], dims_ref_hsize[1], 1 };  // include dummy virtual bond dimensions
	struct dense_tensor bose_hubbard_1d_mat_ref;
	allocate_dense_tensor(DOUBLE_REAL, 4, dim_ref, &bose_hubbard_1d_mat_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "bose_hubbard_1d_mat", H5T_NATIVE_DOUBLE, bose_hubbard_1d_mat_ref.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&bose_hubbard_1d_mat_dns, &bose_hubbard_1d_mat_ref, 1e-13)) {
		return "matrix representation of Bose-Hubbard Hamiltonian based on MPO form does not match reference";
	}

	// clean up
	delete_block_sparse_tensor(&bose_hubbard_1d_mat);
	delete_dense_tensor(&bose_hubbard_1d_mat_dns);
	delete_dense_tensor(&bose_hubbard_1d_mat_ref);
	delete_mpo(&bose_hubbard_1d_mpo);

	H5Fclose(file);

	return 0;
}


char* test_fermi_hubbard_1d_mpo()
{
	hid_t file = H5Fopen("../test/operator/data/test_fermi_hubbard_1d_mpo.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_fermi_hubbard_1d_mpo failed";
	}

	// number of lattice sites
	const int nsites = 4;

	// Hamiltonian parameters
	const double t  = 11./9;
	const double u  = 13./4;
	const double mu =  3./7;

	struct mpo fermi_hubbard_1d_mpo;
	construct_fermi_hubbard_1d_mpo(nsites, t, u, mu, &fermi_hubbard_1d_mpo);
	if (!mpo_is_consistent(&fermi_hubbard_1d_mpo)) {
		return "internal consistency check for Fermi-Hubbard Hamiltonian MPO failed";
	}

	for (int i = 0; i <= nsites; i++) {
		const long bdim_ref = (i == 0 || i == nsites ? 1 : 6);
		if (mpo_bond_dim(&fermi_hubbard_1d_mpo, i) != bdim_ref) {
			return "virtual bond dimension of MPO representation of Fermi-Hubbard Hamiltonian does not match reference";
		}
	}

	struct block_sparse_tensor fermi_hubbard_1d_mat;
	mpo_to_matrix(&fermi_hubbard_1d_mpo, &fermi_hubbard_1d_mat);

	// convert to dense tensor for comparison with reference
	struct dense_tensor fermi_hubbard_1d_mat_dns;
	block_sparse_to_dense_tensor(&fermi_hubbard_1d_mat, &fermi_hubbard_1d_mat_dns);

	// reference matrix for checking
	hsize_t dims_ref_hsize[2];
	if (get_hdf5_dataset_dims(file, "fermi_hubbard_1d_mat", dims_ref_hsize) < 0) {
		return "obtaining dimensions of reference Hamiltonian failed";
	}
	const long dim_ref[4] = { 1, dims_ref_hsize[0], dims_ref_hsize[1], 1 };  // include dummy virtual bond dimensions
	struct dense_tensor fermi_hubbard_1d_mat_ref;
	allocate_dense_tensor(DOUBLE_REAL, 4, dim_ref, &fermi_hubbard_1d_mat_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "fermi_hubbard_1d_mat", H5T_NATIVE_DOUBLE, fermi_hubbard_1d_mat_ref.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&fermi_hubbard_1d_mat_dns, &fermi_hubbard_1d_mat_ref, 1e-13)) {
		return "matrix representation of Fermi-Hubbard Hamiltonian based on MPO form does not match reference";
	}

	// clean up
	delete_block_sparse_tensor(&fermi_hubbard_1d_mat);
	delete_dense_tensor(&fermi_hubbard_1d_mat_dns);
	delete_dense_tensor(&fermi_hubbard_1d_mat_ref);
	delete_mpo(&fermi_hubbard_1d_mpo);

	H5Fclose(file);

	return 0;
}


char* test_molecular_hamiltonian_mpo()
{
	hid_t file = H5Fopen("../test/operator/data/test_molecular_hamiltonian_mpo.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_molecular_hamiltonian_mpo failed";
	}

	// number of fermionic modes (orbitals)
	const int nmodes = 7;

	// Hamiltonian coefficients
	struct dense_tensor tkin;
	const long dim_tkin[2] = { nmodes, nmodes };
	allocate_dense_tensor(DOUBLE_REAL, 2, dim_tkin, &tkin);
	if (read_hdf5_dataset(file, "tkin", H5T_NATIVE_DOUBLE, tkin.data) < 0) {
		return "reading kinetic hopping coefficients from disk failed";
	}
	struct dense_tensor vint;
	const long dim_vint[4] = { nmodes, nmodes, nmodes, nmodes };
	allocate_dense_tensor(DOUBLE_REAL, 4, dim_vint, &vint);
	if (read_hdf5_dataset(file, "vint", H5T_NATIVE_DOUBLE, vint.data) < 0) {
		return "reading interaction potential coefficients from disk failed";
	}

	struct mpo molecular_hamiltonian_mpo;
	construct_molecular_hamiltonian_mpo(&tkin, &vint, &molecular_hamiltonian_mpo);
	if (!mpo_is_consistent(&molecular_hamiltonian_mpo)) {
		return "internal consistency check for molecular Hamiltonian MPO failed";
	}

	struct block_sparse_tensor molecular_hamiltonian_mat;
	mpo_to_matrix(&molecular_hamiltonian_mpo, &molecular_hamiltonian_mat);

	// convert to dense tensor for comparison with reference
	struct dense_tensor molecular_hamiltonian_mat_dns;
	block_sparse_to_dense_tensor(&molecular_hamiltonian_mat, &molecular_hamiltonian_mat_dns);

	// reference matrix for checking
	hsize_t dims_ref_hsize[2];
	if (get_hdf5_dataset_dims(file, "molecular_hamiltonian_mat", dims_ref_hsize) < 0) {
		return "obtaining dimensions of reference Hamiltonian failed";
	}
	const long dim_ref[4] = { 1, dims_ref_hsize[0], dims_ref_hsize[1], 1 };  // include dummy virtual bond dimensions
	struct dense_tensor molecular_hamiltonian_mat_ref;
	allocate_dense_tensor(DOUBLE_REAL, 4, dim_ref, &molecular_hamiltonian_mat_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "molecular_hamiltonian_mat", H5T_NATIVE_DOUBLE, molecular_hamiltonian_mat_ref.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&molecular_hamiltonian_mat_dns, &molecular_hamiltonian_mat_ref, 1e-13)) {
		return "matrix representation of molecular Hamiltonian based on MPO form does not match reference";
	}

	// clean up
	delete_block_sparse_tensor(&molecular_hamiltonian_mat);
	delete_dense_tensor(&molecular_hamiltonian_mat_dns);
	delete_dense_tensor(&molecular_hamiltonian_mat_ref);
	delete_mpo(&molecular_hamiltonian_mpo);
	delete_dense_tensor(&vint);
	delete_dense_tensor(&tkin);

	H5Fclose(file);

	return 0;
}
