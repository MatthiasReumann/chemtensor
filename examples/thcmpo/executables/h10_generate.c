#include "hamiltonian.h"
#include "mpo.h"
#include "mps.h"
#include "states.h"
#include "storage.h"
#include "thcops.h"
#include "utils.h"

int main() {
	const double TOL = 0;
	const long MAX_VDIM = 250;

	const long N = 27;
	const long L = 10;

	const size_t K = 2;

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

	struct dense_tensor tkin; // t[p, q]
	{
		const long dim[2] = {L, L};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &tkin);
	}

	read_h10((double*)zeta.data, (double*)chi.data, NULL, (double*)tkin.data);

	struct thc_spin_hamiltonian hamiltonian;
	construct_thc_spin_hamiltonian(&tkin, &zeta, &chi, &hamiltonian);

	// hartree fock state
	struct mps psi;
	const unsigned spin_state[] = {3, 3, 3, 3, 3, 0, 0, 0, 0, 0};
	construct_spin_basis_mps(L, spin_state, &psi);

	for (size_t i = 0; i < K; i++) {
		struct mps ret;
		apply_thc_spin_hamiltonian(&hamiltonian, &psi, TOL, MAX_VDIM, &ret);

		delete_mps(&psi);
		move_mps_data(&ret, &psi);
	}

	save_mps_hdf5(&psi, "h10_K20_Dim250.hdf5");

	delete_mps(&psi);
	delete_dense_tensor(&tkin);
	delete_dense_tensor(&chi);
	delete_dense_tensor(&zeta);
}