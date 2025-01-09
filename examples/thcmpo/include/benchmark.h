#include "mpo.h"
#include "mps.h"

void thc_benchmark_apply_thc_run(const long N, const long L, const double tol, const long max_vdim,
								 const long K,
								 void (*read)(double*, double*, double*, double*),
								 const struct mps* start);