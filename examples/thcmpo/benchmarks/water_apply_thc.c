#include "benchmark.h"
#include "mps.h"
#include "states.h"
#include "thcops.h"
#include "utils.h"

int main() {
	const long N = 28;
	const long L = 7;

	const double TOL = 0;
	const long MAX_VDIM = 250;

	struct mps hfs; // hartree fock state
	const unsigned spin_state[] = {3, 3, 3, 3, 3, 0, 0};
	{
		construct_spin_basis_mps(L, spin_state, &hfs);
	}

	thc_benchmark_apply_thc_run(N, L, TOL, MAX_VDIM, 20, &read_water, &hfs);

	delete_mps(&hfs);
}