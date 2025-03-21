#include "benchmark.h"
#include "mps.h"
#include "states.h"
#include "thcops.h"
#include "utils.h"

#define THC_NO_REDUC 1

int main() {
	const long N = 27;
	const long L = 10;

	const double TOL = 0;
	const long MAX_VDIM = 250;

	struct mps psi;
	load_mps("../examples/thcmpo/data/h10_K20_Dim250.hdf5", &psi);

	thc_benchmark_apply_thc_run(N, L, TOL, MAX_VDIM, &read_h10, &psi);

	delete_mps(&psi);
}