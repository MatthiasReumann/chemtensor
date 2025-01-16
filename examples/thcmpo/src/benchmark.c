#include "hamiltonian.h"
#include "mpo.h"
#include "mps.h"
#include <stdio.h>
#include <time.h>

#include "states.h"
#include "thcops.h"
#include "utils.h"

void thc_benchmark_apply_thc_run(const long N, const long L, const double tol, const long max_vdim,
								 const long K,
								 void (*read)(double*, double*, double*, double*),
								 const struct mps* start) {
	const size_t REPEATS = 10;

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

	read((double*)zeta.data, (double*)chi.data, NULL, (double*)tkin.data);

	struct thc_spin_hamiltonian hamiltonian;
	construct_thc_spin_hamiltonian(&tkin, &zeta, &chi, &hamiltonian);

	printf("setup\n");
	struct mps psi; // H^{K}|start>
	copy_mps(start, &psi);
	print_mps(psi);
	for (size_t i = 0; i < K; i++) {
		struct mps ret;
		apply_thc_spin_hamiltonian(&hamiltonian, &psi, tol, max_vdim, &ret);

		delete_mps(&psi);
		move_mps_data(&ret, &psi);
		print_mps(psi);
	}
	printf("start benchmark\n");

	double sum_t = 0.;
	for (size_t i = 0; i < REPEATS; i++) {
		struct mps v_psi; // v|ᴪ>
		{
			struct timespec t0, t1;

			clock_gettime(CLOCK_MONOTONIC, &t0);
			apply_thc_spin_coulomb(&hamiltonian, &psi, tol, max_vdim, &v_psi);
			clock_gettime(CLOCK_MONOTONIC, &t1);

			sum_t += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1000000000.0;
		}
		delete_mps(&v_psi);
	}

	printf("%f\n", sum_t / REPEATS);

	delete_dense_tensor(&chi);
	delete_dense_tensor(&zeta);
}