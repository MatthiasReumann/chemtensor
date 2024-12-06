#include "states.h"

void construct_computational_basis_mps(const int nsites, const unsigned *basis_state, struct mps *mps)
{
    const long d = 2;
    const qnumber qsite[2] = {0, 1};

    const double states[2][2] = {
        {1, 0}, // ket0
        {0, 1}  // ket1
    };

    const int ndim = 3;
    const long dim[3] = {1, d, 1};

    const enum tensor_axis_direction axis_dir[3] = {TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN};

    int acc = 0;
    qnumber qbond[1] = {0};

    allocate_empty_mps(nsites, d, qsite, mps);

    for (size_t i = 0; i < nsites; i++)
    {
        const unsigned ith = basis_state[i];

        struct dense_tensor dt;
        allocate_dense_tensor(CT_DOUBLE_REAL, ndim, dim, &dt);

        if (ith < 2)
        {
            memcpy(dt.data, &states[ith], sizeof(states[ith]));
            acc += qsite[ith];
        }

        const qnumber qbondn[1] = {acc};
        const qnumber *qnums[3] = {qbond, qsite, qbondn};
        dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mps->a[i]);

        delete_dense_tensor(&dt);

        qbond[0] = acc;
    }
}

void construct_spin_basis_mps(const int nsites, const unsigned *spin_state, struct mps *mps)
{
    const long d = 4;

    const qnumber qsite[] = {
        encode_quantum_number_pair(0, 0),
        encode_quantum_number_pair(1, -1),
        encode_quantum_number_pair(1, 1),
        encode_quantum_number_pair(2, 0),
    };

    const double states[4][4] = {
        {1, 0, 0, 0}, // no electron (0, 0)
        {0, 1, 0, 0}, // only spin-down (0, down)
        {0, 0, 1, 0}, // only spin-up (up, 0)
        {0, 0, 0, 1}  // spin-up and spin-down (up, down)
    };

    const int ndim = 3;
    const long dim[3] = {1, d, 1};

    const enum tensor_axis_direction axis_dir[3] = {TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN};

    int acc = 0;

    qnumber qbond[1] = {0};

    allocate_empty_mps(nsites, d, qsite, mps);

    for (size_t i = 0; i < nsites; i++)
    {
        const unsigned ith = spin_state[i];

        struct dense_tensor dt;
        allocate_dense_tensor(CT_DOUBLE_REAL, ndim, dim, &dt);

        if (ith < 4)
        {
            memcpy(dt.data, &states[ith], sizeof(states[ith]));
            acc += qsite[ith];
        }

        const qnumber qbondn[1] = {acc};
        const qnumber *qnums[3] = {qbond, qsite, qbondn};
        dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mps->a[i]);

        delete_dense_tensor(&dt);

        qbond[0] = acc;
    }
}