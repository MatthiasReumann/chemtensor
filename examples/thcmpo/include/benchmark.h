#include "mps.h"
#include "mpo.h"

void thc_benchmark_apply_thc_run(const long N, const long L, const double tol, const long max_vdim,
								 void (*read)(double*, double*, double*, double*),
								 void (*apply_thcf)(const struct mps*, struct mpo**, const struct dense_tensor, const long, const double, const long, struct mps*),
								 const struct mps* start);