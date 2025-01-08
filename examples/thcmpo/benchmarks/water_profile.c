#include "benchmark.h"
#include "mps.h"
#include "states.h"
#include "thcops.h"
#include "utils.h"

int main() {
	const long N = 28;
	const long L = 7;

	const double TOL = 1e-20;
	const long MAX_VDIM = LONG_MAX;

	struct mps hfs; // hartree fock state
	const unsigned spin_state[] = {3, 3, 3, 3, 3, 0, 0};
	{
		construct_spin_basis_mps(L, spin_state, &hfs);
	}

	thc_benchmark_apply_thc_run(N, L, TOL, MAX_VDIM, 0, &read_water, &apply_thc_omp_prof, &hfs);

	delete_mps(&hfs);
}