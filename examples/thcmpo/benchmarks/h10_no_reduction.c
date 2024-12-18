#include "benchmark.h"
#include "thcmpo.h"
#include "states.h"
#include "utils.h"
#include "mps.h"

int main()
{
    const long N = 27;
    const long L = 10;

    const double TOL = 1e-20;
    const long MAX_VDIM = LONG_MAX;

    struct mps hfs; // hartree fock state
    const unsigned spin_state[] = {3, 3, 3, 3, 3, 0, 0, 0, 0, 0};
    {
        construct_spin_basis_mps(L, spin_state, &hfs);
    }

    thc_benchmark_apply_thc_run(N, L, TOL, MAX_VDIM, &read_h10, &apply_thc_omp_no_reduc, &hfs);

    delete_mps(&hfs);
}