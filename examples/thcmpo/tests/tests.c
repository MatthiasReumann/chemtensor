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

int main() {
	const long N = 28;
	const long L = 7;

	const double TOL = 1e-20;
	const long MAX_VDIM = LONG_MAX;

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
	apply_thc_hamiltonian(&hamiltonian, &psi, TOL, MAX_VDIM, &h_psi);

	struct dense_tensor h_psi_vec;
	{
		struct block_sparse_tensor bst;
		mps_to_statevector(&h_psi, &bst);
		block_sparse_to_dense_tensor(&bst, &h_psi_vec);
		const long dim[] = {16384};
		reshape_dense_tensor(1, dim, &h_psi_vec);
		delete_block_sparse_tensor(&bst);
	}

	assert(dense_tensor_allclose(&ref, &h_psi_vec, 1e-10));

	// teardown
	delete_dense_tensor(&h_psi_vec);
	delete_mps(&h_psi);
	delete_dense_tensor(&ref);
	delete_mps(&psi);
	delete_dense_tensor(&tkin);
	delete_dense_tensor(&H);
	delete_dense_tensor(&chi);
	delete_dense_tensor(&zeta);
}