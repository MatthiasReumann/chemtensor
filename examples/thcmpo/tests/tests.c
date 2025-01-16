#include "hamiltonian.h"
#include "mpo.h"
#include "mps.h"
#include <stdio.h>
#include <time.h>

#include "states.h"
#include "thcops.h"
#include "utils.h"

void compute_reference(const struct dense_tensor* H, const struct mps* psi, struct dense_tensor* ret) {
	struct dense_tensor psi_vec;
	{
		struct block_sparse_tensor bst;
		mps_to_statevector(psi, &bst);
		block_sparse_to_dense_tensor(&bst, &psi_vec);
		const long dim[] = {16384};
		reshape_dense_tensor(1, dim, &psi_vec);
		delete_block_sparse_tensor(&bst);
	}

	const int i_ax = 1;
	dense_tensor_multiply_axis(H, i_ax, &psi_vec, TENSOR_AXIS_RANGE_LEADING, ret);

	delete_dense_tensor(&psi_vec);
}

int test_water() {
	const long N = 28;
	const long L = 7;

	const double TOL = 0;
	const long MAX_VDIM = 250;

	struct dense_tensor zeta; // ζ
	{
		const long dim[2] = {N, N};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &zeta);
	}

	struct dense_tensor chi; // χ
	{
		const long dim[2] = {N, L};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &chi);
	}

	struct dense_tensor H; // H
	{
		const long dim[2] = {16384, 16384};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &H);
	}

	struct dense_tensor tkin; // t[p, q]
	{
		const long dim[2] = {L, L};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &tkin);
	}

	read_water((double*)zeta.data, (double*)chi.data, (double*)H.data, (double*)tkin.data);

	struct thc_spin_hamiltonian hamiltonian;
	construct_thc_spin_hamiltonian(&tkin, &zeta, &chi, &hamiltonian);

	// hartree fock state
	struct mps psi;
	const unsigned spin_state[] = {3, 3, 3, 3, 3, 0, 0};
	construct_spin_basis_mps(L, spin_state, &psi);

	// hartree fock as dense tensor ~ vector
	struct dense_tensor ref;
	compute_reference(&H, &psi, &ref);

	struct mps h_psi;
	apply_thc_spin_hamiltonian(&hamiltonian, &psi, TOL, MAX_VDIM, &h_psi);

	struct dense_tensor h_psi_vec;
	{
		struct block_sparse_tensor bst;
		mps_to_statevector(&h_psi, &bst);
		block_sparse_to_dense_tensor(&bst, &h_psi_vec);
		const long dim[] = {16384};
		reshape_dense_tensor(1, dim, &h_psi_vec);
		delete_block_sparse_tensor(&bst);
	}

	if(!dense_tensor_allclose(&ref, &h_psi_vec, 1e-9)){
		const long nelem = dense_tensor_num_elements(&ref);
		printf("diff: %.15f\n", uniform_distance(ref.dtype, nelem, ref.data, h_psi_vec.data));
		printf("a: %f\n", dense_tensor_norm2(&ref));
		printf("b: %f\n", dense_tensor_norm2(&h_psi_vec));
		return 1;
	}

	// teardown
	delete_dense_tensor(&h_psi_vec);
	delete_mps(&h_psi);
	delete_dense_tensor(&ref);
	delete_mps(&psi);
	delete_dense_tensor(&tkin);
	delete_dense_tensor(&H);
	delete_dense_tensor(&chi);
	delete_dense_tensor(&zeta);

	return 0;
}

int test() {
	const long N = 7;
	const long L = 5;

	const double TOL = 0;
	const long MAX_VDIM = 250;

	hid_t file = H5Fopen("../test/algorithm/data/test_apply_thc_spin_molecular_hamiltonian.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	assert(file >= 0);

	struct dense_tensor tkin;
	{
		const long dim[2] = {L, L};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &tkin);
		assert(read_hdf5_dataset(file, "tkin", H5T_NATIVE_DOUBLE, tkin.data) >= 0);
	}

	struct dense_tensor zeta; // ζ
	{
		const long dim[2] = {N, N};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &zeta);
		assert(read_hdf5_dataset(file, "thc_kernel", H5T_NATIVE_DOUBLE, zeta.data) >= 0);
	}

	struct dense_tensor chi; // χ
	{
		struct dense_tensor tmp;
		const long dim[2] = {L, N};
		const int perm[2] = {1, 0};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &tmp);
		assert(read_hdf5_dataset(file, "thc_transform", H5T_NATIVE_DOUBLE, tmp.data) >= 0);
		transpose_dense_tensor(perm, &tmp, &chi); // required: shape(N, L)
		delete_dense_tensor(&tmp);
	}

	struct thc_spin_hamiltonian hamiltonian;
	construct_thc_spin_hamiltonian(&tkin, &zeta, &chi, &hamiltonian);

	// input statevector as MPS
	// physical particle number and spin quantum numbers (encoded as single integer)
	const qnumber qn[4] = {0, 1, 1, 2};
	const qnumber qs[4] = {0, -1, 1, 0};
	const qnumber qsite[4] = {
		encode_quantum_number_pair(qn[0], qs[0]),
		encode_quantum_number_pair(qn[1], qs[1]),
		encode_quantum_number_pair(qn[2], qs[2]),
		encode_quantum_number_pair(qn[3], qs[3]),
	};

	// virtual bond quantum numbers
	const long psi_dim_bonds[6] = {1, 19, 39, 41, 23, 1};
	qnumber** psi_qbonds = ct_malloc((L + 1) * sizeof(qnumber*));
	for (int i = 0; i < L + 1; i++) {
		psi_qbonds[i] = ct_malloc(psi_dim_bonds[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "psi_qbond%i", i);
		assert(read_hdf5_attribute(file, varname, H5T_NATIVE_INT, psi_qbonds[i]) >= 0);
	}

	struct mps psi;
	allocate_mps(CT_DOUBLE_REAL, L, 4, qsite, psi_dim_bonds, (const qnumber**)psi_qbonds, &psi);

	// read MPS tensors from disk
	for (int i = 0; i < L; i++) {
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(psi.a[i].dtype, psi.a[i].ndim, psi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "psi_a%i", i);
		assert(read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) >= 0);

		dense_to_block_sparse_tensor_entries(&a_dns, &psi.a[i]);
		delete_dense_tensor(&a_dns);
	}

	assert(mps_is_consistent(&psi));

	// reference vector
	// include dummy virtual bond dimensions
	const long dim_h_psi_ref[3] = { 1, ipow(4, L), 1 };
	struct dense_tensor h_psi_ref;
	allocate_dense_tensor(CT_DOUBLE_REAL, 3, dim_h_psi_ref, &h_psi_ref);
	assert(read_hdf5_dataset(file, "h_psi", H5T_NATIVE_DOUBLE, h_psi_ref.data) >= 0);

	struct mps h_psi;
	apply_thc_spin_hamiltonian(&hamiltonian, &psi, TOL, MAX_VDIM, &h_psi);

	struct block_sparse_tensor h_psi_vec;
	mps_to_statevector(&h_psi, &h_psi_vec);

	struct dense_tensor h_psi_vec_dns;
	block_sparse_to_dense_tensor(&h_psi_vec, &h_psi_vec_dns);

	if(!dense_tensor_allclose(&h_psi_vec_dns, &h_psi_ref, 1e-13)) {
		return 1;
	}

	// teardown
	delete_dense_tensor(&h_psi_ref);
	delete_mps(&psi);
	for (int i = 0; i < L + 1; i++) {
		ct_free(psi_qbonds[i]);
	}
	ct_free(psi_qbonds);
	delete_dense_tensor(&tkin);
	delete_dense_tensor(&chi);
	delete_dense_tensor(&zeta);

	H5Fclose(file);

	return 0;
}

int main() {
	printf("test_water: %d\n", test_water());
	printf("test_thc: %d\n", test());
}