#include "benchmark.h"
#include "mps.h"
#include "states.h"
#include "thcops.h"
#include "utils.h"
#include "storage.h"

int main() {
	const long N = 27;
	const long L = 10;

	const double TOL = 0;
	const long MAX_VDIM = 250;

	struct mps psi;
	load_mps_hdf5("../examples/thcmpo/data/h10_K20_Dim250.h5", &psi);

	thc_benchmark_apply_thc_run(N, L, TOL, MAX_VDIM, &read_h10, &psi);

	delete_mps(&psi);
}