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

	struct dense_tensor t; // t[p, q]
	{
		const long dim[2] = {L, L};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &t);
	}

	read_water((double*)zeta.data, (double*)chi.data, (double*)H.data, (double*)t.data);

	struct mpo** g; // G[μ, σ]
	allocate_thc_mpo_map(N, &g);
	construct_thc_mpo_map(chi, N, L, g);

	// hartree fock state
	struct mps psi;
	const unsigned spin_state[] = {3, 3, 3, 3, 3, 0, 0};
	construct_spin_basis_mps(L, spin_state, &psi);

	// hartree fock as dense tensor ~ vector
	struct dense_tensor ref;
	compute_reference(&H, &psi, &ref);

	struct mpo T;
	{
		struct dense_tensor vint;
		struct mpo_assembly assembly;
		const long dim[] = {L, L, L, L};
		allocate_dense_tensor(CT_DOUBLE_REAL, 4, dim, &vint);
		construct_spin_molecular_hamiltonian_mpo_assembly(&t, &vint, false, &assembly);
		mpo_from_assembly(&assembly, &T);
		delete_mpo_assembly(&assembly);
	}

	struct mps v_psi;
	{
		// Initialize as '0' MPS.
		const double alpha = 0.;
		copy_mps(&psi, &v_psi);
		scale_block_sparse_tensor(&alpha, &v_psi.a[0]);

		apply_thc_omp(&psi, g, zeta, N, TOL, MAX_VDIM, &v_psi);
	}

	// t|ᴪ>
	struct mps t_psi;
	apply_and_compress(&psi, &T, TOL, MAX_VDIM, &t_psi);

	// t|ᴪ> + v|ᴪ>
	struct mps h_psi;
	add_and_compress(&t_psi, &v_psi, TOL, MAX_VDIM, &h_psi);

	struct dense_tensor actual;
	{
		struct block_sparse_tensor bst;
		mps_to_statevector(&h_psi, &bst);
		block_sparse_to_dense_tensor(&bst, &actual);
		const long dim[] = {16384};
		reshape_dense_tensor(1, dim, &actual);
		delete_block_sparse_tensor(&bst);
	}

	assert(dense_tensor_allclose(&ref, &actual, 1e-8));

	// teardown
	delete_dense_tensor(&actual);
	delete_mps(&h_psi);
	delete_mps(&t_psi);
	delete_mps(&v_psi);
	delete_mpo(&T);
	delete_dense_tensor(&ref);
	delete_mps(&psi);
	// TODO: g
	delete_dense_tensor(&t);
	delete_dense_tensor(&H);
	delete_dense_tensor(&chi);
	delete_dense_tensor(&zeta);
}