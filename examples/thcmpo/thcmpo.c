#include <stdio.h>
#include "aligned_memory.h"
#include "mpo.h"

// TODO: Ask Yu/Prof.Mendl if I can implement this using the opchain method. Problem: Bosonic Z.
//
//
void construct_elementary_thc_mpo(const int nsites, const dcomplex *chi_row, struct mpo_assembly *assembly)
{
    assert(nsites >= 2);

    // Setup physical quantum numbers.
    assembly->d = 2;
    assembly->dtype = CT_DOUBLE_COMPLEX;
    assembly->qsite = ct_calloc(assembly->d, sizeof(qnumber));
    assembly->qsite[1] = 1;

    // operator map
    // 0:I, 1:Z, 2:chi_0*a, 3:chi_1*a, ...
    assembly->num_local_ops = 2 + nsites; // Identity + Z + each site ann/crea op.
    assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
    for (int i = 0; i < assembly->num_local_ops; i++)
    {
        const long dim[2] = {assembly->d, assembly->d};
        allocate_dense_tensor(assembly->dtype, 2, dim, &assembly->opmap[i]);
    }

    const dcomplex pZ[4] = {1, 0, 0, -1};
    const dcomplex annihilation[4] = {0, 1, 0, 0};
    const dcomplex creation[4] = {0, 0, 1, 0};

    dense_tensor_set_identity(&assembly->opmap[0]);  // copy 'I'
    memcpy(assembly->opmap[2].data, pZ, sizeof(pZ)); // copy 'Z'
    for (int i = 2; i < nsites + 2; i++)
    { // copy 'a'
        memcpy(assembly->opmap[i].data, creation, sizeof(creation));
    }

    // coefficient map; first two entries must always be 0 and 1
    assembly->num_coeffs = 2 + nsites;
    assembly->coeffmap = (dcomplex *)ct_malloc(assembly->num_coeffs * sizeof(dcomplex));

    const double coeffmap[2] = {0, 1};
    memcpy(assembly->coeffmap, coeffmap, sizeof(2 * sizeof(dcomplex)));         // copy mandatory
    memcpy(assembly->coeffmap + 2, chi_row, sizeof(nsites * sizeof(dcomplex))); // copy prefactors

    // // local two-site and single-site terms
    // int oids_c0[] = { OID_Z, OID_Z };  qnumber qnums_c0[] = { 0, 0, 0 };
    // int oids_c1[] = { OID_Z };         qnumber qnums_c1[] = { 0, 0 };
    // int oids_c2[] = { OID_X };         qnumber qnums_c2[] = { 0, 0 };
    // struct op_chain lopchains[] = {
    // 	{ .oids = oids_c0, .qnums = qnums_c0, .cid = 2, .length = ARRLEN(oids_c0), .istart = 0 },
    // 	{ .oids = oids_c1, .qnums = qnums_c1, .cid = 3, .length = ARRLEN(oids_c1), .istart = 0 },
    // 	{ .oids = oids_c2, .qnums = qnums_c2, .cid = 4, .length = ARRLEN(oids_c2), .istart = 0 },
    // };

    // // convert to an MPO graph
    // local_opchains_to_mpo_graph(nsites, lopchains, ARRLEN(lopchains), &assembly->graph);
}


void construct_elementary_mpo(const int nsites, const dcomplex* chi_row, const bool is_creation, struct mpo* mpo)
{
    const long d = 2;
    const qnumber qd[2] = {0, 1};

    mpo->d = d;
    mpo->nsites = nsites;
    mpo->qsite = ct_malloc(d * sizeof(qnumber));
    memcpy(mpo->qsite, &qd, d * sizeof(qnumber));

    mpo->a = ct_calloc(nsites, sizeof(struct block_sparse_tensor));
    
    const enum tensor_axis_direction axis_dir[4] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN, TENSOR_AXIS_IN };
    
    struct dense_tensor dt;

    // first site. dimensions: 1 x d x d x 2
    {
        const long dim[4] = { 1, d, d, 2 };
        allocate_dense_tensor(CT_DOUBLE_COMPLEX, 4, dim, &dt);
        
        if (is_creation) {
            const qnumber qD[1] = { 0 };
            const qnumber qD_next[2] = { 0, 1 };
            const qnumber* qnums[4] = { qD, qd, qd, qD_next};
            const dcomplex data[8] = { 1, 0, 0, 0, 0, chi_row[0], -1, 0};
            
            memcpy(dt.data, &data, 8 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[0]);
        } else {
            const qnumber qD[1] = { 0 };
            const qnumber qD_next[2] = { 0, -1 };
            const qnumber* qnums[4] = { qD, qd, qd, qD_next};
            const dcomplex data[8] = { 1, 0, 0, chi_row[0], 0, 0, -1, 0};
            
            memcpy(dt.data, &data, 8 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[0]);
        }
    }

    // intermediate sites: dimensions: 2 x d x d x 2
    for (int i = 1; i < nsites - 1; i++)
	{
        const long dim[4] = { 2, d, d, 2 };
        allocate_dense_tensor(CT_DOUBLE_COMPLEX, 4, dim, &dt);

        if (is_creation) {
            const qnumber qD[2] = { 0, 1 };
		    const qnumber* qnums[4] = { qD, qd, qd, qD };
            const dcomplex data[16] = { 1, 0, 0, 0, 0, chi_row[i], -1, 0, 0, 1, 0, 0, 0, 0, 0, 1 };
            memcpy(dt.data, data, 16 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[i]);

        } else {
            const qnumber qD[2] = { 0, -1 };
		    const qnumber* qnums[4] = { qD, qd, qd, qD };
            const dcomplex data[16] = { 1, 0, 0, chi_row[i], 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1 };
            
            memcpy(dt.data, data, 16 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[i]);
        }
	}

    // last site. dimensions: 2 x d x d x 1
    {
        const long dim[4] = { 2, d, d, 1 };
        allocate_dense_tensor(CT_DOUBLE_COMPLEX, 4, dim, &dt);

        if (is_creation) {
            const qnumber qD[2] = { 0, 1 };
            const qnumber qD_next[1] = { 1 };
            const qnumber* qnums[4] = { qD, qd, qd, qD_next};
            const dcomplex data[8] = { 0, 0, chi_row[nsites - 1], 0, 1, 0, 0, 1};

            memcpy(dt.data, data, 8 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[nsites - 1]);
        } else {
            const qnumber qD[2] = { 0, -1 };
            const qnumber qD_next[1] = { -1 };
            const qnumber* qnums[4] = { qD, qd, qd, qD_next};
            const dcomplex data[8] = { 0, chi_row[nsites - 1], 0, 0, 1, 0, 0, 1};

            memcpy(dt.data, data, 8 * sizeof(dcomplex));
            dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mpo->a[nsites - 1]);
        }
    }
}

void print_mpo(struct mpo mpo)
{
    printf("MPO: nsites=%d, d=%ld, qsite=(%d, %d)\n", mpo.nsites, mpo.d, mpo.qsite[0], mpo.qsite[1]);

    for(int i = 0; i < mpo.nsites; i++) {
        struct dense_tensor dt;
        block_sparse_to_dense_tensor(&mpo.a[i], &dt);

        dcomplex* data = (dcomplex*)(dt.data);
        const long nblocks = integer_product(dt.dim, dt.ndim);
        printf("[");
        for (long k = 0; k < nblocks; k++)
		{
			printf(" %.2f", creal(data[k]));
		}
        printf("]\n");
    }
}


int main()
{
    struct mpo mpo;
    const int nsites = 3;
    const dcomplex chi_row[3] = { 42, 1337, 72 };
    construct_elementary_mpo(nsites, chi_row, false, &mpo);
    print_mpo(mpo);
    printf("is_conssitent=%d\n", mpo_is_consistent(&mpo));
}