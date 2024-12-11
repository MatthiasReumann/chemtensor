#include "hamiltonian.h"
#include "mpo.h"
#include "mps.h"
#include <stdio.h>
#include <time.h>

#include "states.h"
#include "thcmpo.h"
#include "utils.h"

void thc_benchmark_apply_thc_run(const long N, const long L, const double tol, const long max_vdim,
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

	read((double*)zeta.data, (double*)chi.data, NULL, NULL);

	// G_{nu, sigma}
	struct mpo** g;
	g = ct_malloc(2 * N * sizeof(struct mpo*));
	for (size_t i = 0; i < 2 * N; i++) {
		g[i] = ct_malloc(2 * sizeof(struct mpo));
	}
	construct_g_4d(chi, N, L, g);

    double sum_t = 0.;
    for(size_t i = 0; i < REPEATS; i++) {
        struct mps v_psi; // v|ᴪ>
        {
            struct timespec t0, t1;

            // Initialize as '0' MPS.
            const double alpha = 0.;
            copy_mps(start, &v_psi);
            scale_block_sparse_tensor(&alpha, &v_psi.a[0]);

            clock_gettime(CLOCK_MONOTONIC, &t0);
            apply_thcf(start, g, zeta, N, tol, max_vdim, &v_psi);
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