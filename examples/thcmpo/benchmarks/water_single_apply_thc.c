#include <stdio.h>
#include <time.h>
#include "mps.h"
#include "mpo.h"
#include "hamiltonian.h"

#include "utils.h"
#include "states.h"
#include "thcmpo.h"

int main()
{
    const long N = 28;
    const long L = 7;

    const double TOL = 1e-20;
    const long MAX_VDIM = LONG_MAX;

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

    read_water((double *)zeta.data, (double *)chi.data, NULL, NULL);

    // G_{nu, sigma}
    struct mpo **g;
    g = ct_malloc(2 * N * sizeof(struct mpo *));
    for (size_t i = 0; i < 2 * N; i++)
    {
        g[i] = ct_malloc(2 * sizeof(struct mpo));
    }
    construct_g_4d(chi, N, L, g);

    // hartree fock state
    struct mps hfs;
    const unsigned spin_state[] = {3, 3, 3, 3, 3, 0, 0};
    {
        construct_spin_basis_mps(L, spin_state, &hfs);
    }

    // v|ᴪ>
    struct mps v_psi;
    {
        struct timespec start, finish;

        construct_spin_zero_mps(L, spin_state, &v_psi);

        clock_gettime(CLOCK_MONOTONIC, &start);
        apply_thc(&hfs, g, zeta, N, TOL, MAX_VDIM, &v_psi);
        clock_gettime(CLOCK_MONOTONIC, &finish);

        double elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        printf("%f\n", elapsed);
    }

    delete_mps(&v_psi);
    delete_mps(&hfs);
    // TODO: Free g
    delete_dense_tensor(&chi);
    delete_dense_tensor(&zeta);
}