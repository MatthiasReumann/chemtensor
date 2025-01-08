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
								 void (*apply_thcf)(const struct mps*, struct mpo**, const struct dense_tensor, const long, const double, const long, struct mps*),
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

	struct dense_tensor t; // t[p, q]
	{
		const long dim[2] = {L, L};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &t);
	}

	read((double*)zeta.data, (double*)chi.data, NULL, (double*)t.data);

	struct mpo** g; // G[μ, σ]
	allocate_thc_mpo_map(N, &g);
	construct_thc_mpo_map(chi, N, L, g);

	struct mpo T; // T (kinetic)
	{
		struct dense_tensor vint;
		struct mpo_assembly assembly;
		const long dim[] = {L, L, L, L};
		allocate_dense_tensor(CT_DOUBLE_REAL, 4, dim, &vint);
		construct_spin_molecular_hamiltonian_mpo_assembly(&t, &vint, false, &assembly);
		mpo_from_assembly(&assembly, &T);
		delete_mpo_assembly(&assembly);
	}

	struct mps psi;
	copy_mps(start, &psi);
	for (size_t i = 0; i < K; i++) {
		const double alpha = 0.;

		struct mps v_psi;
		struct mps t_psi;
		struct mps h_psi;

		copy_mps(&psi, &v_psi);
		scale_block_sparse_tensor(&alpha, &v_psi.a[0]);

		apply_thcf(&psi, g, zeta, N, tol, max_vdim, &v_psi);     //  v|ᴪ>
		apply_and_compress(&psi, &T, tol, max_vdim, &t_psi);     // t|ᴪ>
		add_and_compress(&t_psi, &v_psi, tol, max_vdim, &h_psi); // t|ᴪ> + v|ᴪ>

		delete_mps(&t_psi);
		delete_mps(&v_psi);
		delete_mps(&psi);

		psi = h_psi;
	}

	double sum_t = 0.;
	for (size_t i = 0; i < REPEATS; i++) {
		struct mps v_psi; // v|ᴪ>
		{
			struct timespec t0, t1;

			// Initialize as '0' MPS.
			const double alpha = 0.;
			copy_mps(&psi, &v_psi);
			scale_block_sparse_tensor(&alpha, &v_psi.a[0]);

			clock_gettime(CLOCK_MONOTONIC, &t0);
			apply_thcf(&psi, g, zeta, N, tol, max_vdim, &v_psi);
			clock_gettime(CLOCK_MONOTONIC, &t1);

			double t = (t1.tv_sec - t0.tv_sec);
			t += (t1.tv_nsec - t0.tv_nsec) / 1000000000.0;
			sum_t += t;
		}
		delete_mps(&v_psi);
	}

	printf("%f\n", sum_t / REPEATS);

	// TODO: Free g
	delete_dense_tensor(&chi);
	delete_dense_tensor(&zeta);
}