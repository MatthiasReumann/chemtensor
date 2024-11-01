#include <stdio.h>
#include "mps.h"
#include "mpo.h"
#include "operation.h"
#include "aligned_memory.h"

void print_dt(struct dense_tensor dt)
{
    printf("dt=[ndim=%d, dim=(", dt.ndim);
    for(int i = 0; i < dt.ndim; i++) {
        printf(" %ld ", dt.dim[i]);
    }
    printf(")]\n");
}

void print_mpo(struct mpo mpo)
{
    printf("MPO: L=%d, d=%ld, qsite=(%d, %d)\n", mpo.nsites, mpo.d, mpo.qsite[0], mpo.qsite[1]);

    for (int i = 0; i < mpo.nsites; i++)
    {
        struct dense_tensor dt;
        block_sparse_to_dense_tensor(&mpo.a[i], &dt);

        // for(int k = 0; k < mpo.a[i].ndim; k++) {
        //     for (int j = 0; j < mpo.a[i].dim_blocks[k]; j++) {
        //         printf(" %d ", mpo.a[i].qnums_blocks[k][j]);
        //     }
        // }

        dcomplex *data = (dcomplex *)(dt.data);
        // print_dt(dt);
        const long nblocks = integer_product(dt.dim, dt.ndim);
        printf("[");
        for (long k = 0; k < nblocks; k++)
        {
            printf(" %.1f ", creal(data[k]));
        }
        printf("]\n");
    }
}

void print_bst(struct block_sparse_tensor bst)
{
    printf("bst=[ndim=%d, dim=(%ld, %ld, %ld)]\n",
           bst.ndim,
           bst.dim_blocks[0], bst.dim_blocks[1], bst.dim_blocks[2]);
}

// TODO: Maybe remote g_pair and replace with mpo[2]?
struct g_pair
{
    struct mpo p;
    struct mpo q;
};

struct gmap
{
    struct g_pair *fake_dict; // List of MPO pairs accessible like an dictionary with tuple key via get_value(...)
    long N;                   // THC Rank `N`. Length of `fake_dict` is `2N`.
};

void allocate_gmap(struct gmap *g, const long N)
{
    assert(N > 0);
    g->N = N;
    g->fake_dict = ct_malloc(2 * N * sizeof(struct g_pair));
}


void get_gmap_entry(const struct gmap *g, const int i, const int s, struct g_pair **g_pair)
{
    assert(i + g->N * s < 2 * g->N);
    *g_pair = &(g->fake_dict[i + g->N * s]);
}


void interleave_zero(const dcomplex *a, const long n, const long offset, dcomplex **ret)
{
    *ret = ct_calloc(2 * n, sizeof(dcomplex));
    for (long i = 0; i < n; i++)
    {
        (*ret)[offset + 2 * i] = a[i];
    }
}


void construct_thc_mpo_edge(const int oid, const int cid, const int vids[2], struct mpo_graph_edge* edge) 
{
    edge->nopics = 1;
    edge->opics = ct_malloc(sizeof(struct local_op_ref));
    edge->opics->oid = oid;
    edge->opics->cid = cid;
    edge->vids[0] = vids[0];
    edge->vids[1] = vids[1];
}


void construct_thc_mpo_assembly(const int nsites, const dcomplex *chi_row, const bool is_creation, struct mpo_assembly *assembly)
{
    // physical quantum numbers (particle number)
    const long d = 2;
    const qnumber qsite[2] = {0, 1};

    // operator map
    const int OID_Id = 0; // identity
    const int OID_Z = 1;  // Pauli-Z
    const int OID_B = 2;  // bosonic creation or annihilation depending on `is_creation`

    const dcomplex z[4] = {1., 0., 0., -1.};            // Z
    const dcomplex creation[4] = {0., 0., 1., 0.};     // bosonic creation
    const dcomplex annihilation[4] = {0., 1., 0., 0.}; // bosonic annihilation

	const dcomplex coeffmap[] = { 0, 1 }; // first two entries must always be 0 and 1

    struct mpo_graph graph; // graph for MPO construction

    // allocate and set memory for physical quantum numbers
    assembly->d = d;
    assembly->dtype = CT_DOUBLE_COMPLEX;
    assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
    memcpy(assembly->qsite, &qsite, d * sizeof(qnumber));

    // allocate memory for operators
    assembly->num_local_ops = 3;
    assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
    for (int i = 0; i < assembly->num_local_ops; i++) {
        const long dim[2] = {assembly->d, assembly->d};
        allocate_dense_tensor(assembly->dtype, 2, dim, &assembly->opmap[i]);
    }

    // copy operators into memory
    dense_tensor_set_identity(&assembly->opmap[OID_Id]);
    memcpy(assembly->opmap[OID_Z].data, z, sizeof(z));
    if (is_creation) {
        memcpy(assembly->opmap[OID_B].data, creation, sizeof(creation));
    } else {
        memcpy(assembly->opmap[OID_B].data, annihilation, sizeof(annihilation));
    }

    // copy coefficents; first 2 entries must always be 0 and 1
    assembly->num_coeffs = 2 + nsites;
    assembly->coeffmap = ct_calloc(assembly->num_coeffs, sizeof(dcomplex));
    memcpy(assembly->coeffmap, coeffmap, 2 * sizeof(dcomplex));
    memcpy(((dcomplex*)assembly->coeffmap) + 2, chi_row, nsites * sizeof(dcomplex));

    // setup MPO graph
    assembly->graph.nsites = nsites;
    
    assembly->graph.num_edges = ct_malloc(nsites * sizeof(int));
    assembly->graph.edges = ct_malloc(nsites * sizeof(struct mpo_graph_edge*));

    assembly->graph.num_verts = ct_malloc((nsites + 1) * sizeof(int));
    assembly->graph.verts = ct_malloc((nsites + 1) * sizeof(struct mpo_graph_vertex*));

    {
        // left-most site
        // [v0 -e0(Z)- v1]
        // [ \ -e1(b)- v2]

        // (v0)
        assembly->graph.verts[0] = ct_malloc(1 * sizeof(struct mpo_graph_vertex));
        assembly->graph.num_verts[0] = 1;
        assembly->graph.verts[0]->qnum = 0;
        
        // (e0, e1)
        assembly->graph.edges[0] = ct_malloc(2 * sizeof(struct mpo_graph_vertex));
        assembly->graph.num_edges[0] = 2;

        // (v1, v2)
        assembly->graph.verts[1] = ct_malloc(2 * sizeof(struct mpo_graph_vertex));
        assembly->graph.num_verts[1] = 2;
        assembly->graph.verts[1][0].qnum = 0;
        assembly->graph.verts[1][1].qnum = is_creation ? 1 : -1;
        
        // e0
        {
            const int vids[] = {0, 0};
            construct_thc_mpo_edge(OID_Z, CID_ONE, vids, &assembly->graph.edges[0][0]);

            // v0 -> e0
            mpo_graph_vertex_add_edge(1, 0, &assembly->graph.verts[0][0]);

            // e0 <- v1
            mpo_graph_vertex_add_edge(0, 0, &assembly->graph.verts[1][0]);
        }

        // e1
        {
            const int vids[] = {0, 1};
            construct_thc_mpo_edge(OID_B, 2, vids, &assembly->graph.edges[0][1]);

            // v0 -> e1
            mpo_graph_vertex_add_edge(1, 1, &assembly->graph.verts[0][0]);

            // e1 <- v2
            mpo_graph_vertex_add_edge(0, 1, &assembly->graph.verts[1][1]);
        }
    }

    // intermediate sites [2, L-2]
    // v1 [ -e2(Z)- v3]
    //    [\-e3(b)- v4]
    // v2 [ -e4(I)/ ]
    for (int i = 1; i < nsites - 1; i++) {
        assembly->graph.edges[i] = ct_malloc(3 * sizeof(struct mpo_graph_vertex));
        assembly->graph.num_edges[i] = 3;

        assembly->graph.verts[i + 1] = ct_malloc(2 * sizeof(struct mpo_graph_vertex));
        assembly->graph.num_verts[i + 1] = 2;
        assembly->graph.verts[i + 1][0].qnum = 0;
        assembly->graph.verts[i + 1][1].qnum = is_creation ? 1 : -1;

        // v1 -> e2
        //  \ -> e3
        mpo_graph_vertex_add_edge(1, 0, &assembly->graph.verts[i][0]);
        mpo_graph_vertex_add_edge(1, 1, &assembly->graph.verts[i][0]);
        
        // v2 -> e4
        mpo_graph_vertex_add_edge(1, 2, &assembly->graph.verts[i][1]);

        // e2
        {
            const int vids[] = {0, 0};
            construct_thc_mpo_edge(OID_Z, CID_ONE, vids, &assembly->graph.edges[i][0]);

            // e2 <- v3
            mpo_graph_vertex_add_edge(0, 0, &assembly->graph.verts[i + 1][0]);
        }

        // e3
        {
            const int vids[] = {0, 1};
            construct_thc_mpo_edge(OID_B, 2 + i, vids, &assembly->graph.edges[i][1]);

            // e3 <- v4
            mpo_graph_vertex_add_edge(0, 1, &assembly->graph.verts[i + 1][1]);
        }

        // e4
        {
            const int vids[] = {1, 1};
            construct_thc_mpo_edge(OID_Id, CID_ONE, vids, &assembly->graph.edges[i][2]);

            // e4 <- v4
            mpo_graph_vertex_add_edge(0, 2, &assembly->graph.verts[i + 1][1]);
        }
    }

    {
        // right-most site
        // v3 [ -e5- v5 ]
        // v4 [ -e6-/   ]
        assembly->graph.edges[nsites - 1] = ct_malloc(2 * sizeof(struct mpo_graph_vertex));
        assembly->graph.num_edges[nsites - 1] = 2;

        assembly->graph.verts[nsites] = ct_malloc(1 * sizeof(struct mpo_graph_vertex));
        assembly->graph.num_verts[nsites] = 1;
        assembly->graph.verts[nsites][0].qnum = is_creation ? 1 : -1;

        // v3 -> e5
        // v4 -> e6
        mpo_graph_vertex_add_edge(1, 0, &assembly->graph.verts[nsites - 1][0]);
        mpo_graph_vertex_add_edge(1, 1, &assembly->graph.verts[nsites - 1][1]);

        // e5
        {
            const int vids[] = {0, 0};
            construct_thc_mpo_edge(OID_B, 2 + (nsites - 1), vids, &assembly->graph.edges[nsites - 1][0]);

            // e5 <- v5
            mpo_graph_vertex_add_edge(0, 0, &assembly->graph.verts[nsites][0]);
        }

        // e6
        {
            const int vids[] = {1, 0};
            construct_thc_mpo_edge(OID_Id, CID_ONE, vids, &assembly->graph.edges[nsites - 1][1]);

            // e6 <- v5
            mpo_graph_vertex_add_edge(0, 1, &assembly->graph.verts[nsites][0]);
        }
    }

    // TODO: Annihilation operator is probably wrong. Should be 0. See paper.
    //       Yu probably used the chi_row vector with 0s to achieve this.
    assert(mpo_graph_is_consistent(&assembly->graph));
}


void construct_computational_basis_mps_2d(const enum numeric_type dtype, const int nsites, const int basis_state, struct mps *mps)
{
    const long d = 2;
    const qnumber qsite[2] = {0, 1};

    const dcomplex state_zero[2] = {1, 0};
    const dcomplex state_one[2] = {0, 1};

    const int ndim = 3;
    const long dim[3] = {1, d, 1};

    const enum tensor_axis_direction axis_dir[3] = {
        TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN};

    int acc = 0;

    qnumber qbond_curr[1] = {acc};
    qnumber qbond_next[1];

    allocate_empty_mps(nsites, d, qsite, mps);

    for (int i = 0; i < nsites; i++)
    {
        const int ith = ((basis_state & (1 << (nsites - i - 1))) >> (nsites - i - 1));

        struct dense_tensor dt;
        allocate_dense_tensor(dtype, ndim, dim, &dt);

        if (ith == 0)
        {
            memcpy(dt.data, &state_zero, 2 * sizeof(dcomplex));
        }
        else if (ith == 1)
        {
            memcpy(dt.data, &state_one, 2 * sizeof(dcomplex));
            acc++;
        }
        qbond_next[0] = acc;

        const qnumber *qnums[3] = {qbond_curr, qsite, qbond_next};
        dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mps->a[i]);

        delete_dense_tensor(&dt);

        qbond_curr[0] = qbond_next[0];
    }
}


void construct_gmap(dcomplex **chi, const long N, const long L, struct gmap *g)
{

    // chi.shape = (N, L/2)
    dcomplex *chi_row;

    allocate_gmap(g, N);
    for (long i = 0; i < N; i++)
    {
        // spin up
        {
            struct g_pair *pair;
            get_gmap_entry(g, i, 0, &pair);
            interleave_zero(chi[i], L / 2, 0, &chi_row);

            struct mpo_assembly assembly_p, assembly_q;
            construct_thc_mpo_assembly(L, chi_row, false, &assembly_p);
            construct_thc_mpo_assembly(L, chi_row, true, &assembly_q);
            
            
            mpo_from_assembly(&assembly_p, &pair->p);
            mpo_from_assembly(&assembly_q, &pair->q);
            
            delete_mpo_assembly(&assembly_p);
            delete_mpo_assembly(&assembly_q);
        }

        // spin down
        {
            struct g_pair *pair;
            struct mpo_assembly assembly_p, assembly_q;
            get_gmap_entry(g, i, 1, &pair);
            interleave_zero(chi[i], L / 2, 1, &chi_row);
            
            construct_thc_mpo_assembly(L, chi_row, false, &assembly_p);
            construct_thc_mpo_assembly(L, chi_row, true, &assembly_q);
            
            mpo_from_assembly(&assembly_p, &pair->p);
            mpo_from_assembly(&assembly_q, &pair->q);
            
            delete_mpo_assembly(&assembly_p);
            delete_mpo_assembly(&assembly_q);
        }
    }
}


void THCMPO_calculate_phi(const struct mps *psi, struct gmap *g, const long N, const double tol, struct mps *phi)
{
    const int S = 2; // |{UP, DOWN}| = 2
    for (long n = 0; n < N; n++)
    {
        for (int s1 = 0; s1 < S; s1++)
        {

            struct mps a;
            struct mps b;
            struct g_pair *pair_n_s1;
            get_gmap_entry(g, n, s1, &pair_n_s1);

            apply_operator(&pair_n_s1->p, psi, &a);
            // TODO: Compress
            apply_operator(&pair_n_s1->q, &a, &b);
            // TODO: Compress

            for (long m = 0; m < N; m++)
            {
                for (int s2 = 0; s2 < S; s2++)
                {
                    struct mps c;
                    struct mps d;
                    struct g_pair *pair_m_s2;
                    get_gmap_entry(g, m, s2, &pair_m_s2);

                    apply_operator(&pair_m_s2->p, &b, &c);
                    // TODO: Compress
                    apply_operator(&pair_m_s2->q, &c, &d);
                    // TODO: Compress
                }
            }
        }
    }
}

int main()
{
    const long N = 4;
    const long L = 3;

    dcomplex **chi = ct_malloc(N * sizeof(dcomplex *));
    for (int i = 0; i < N; i++)
    {
        chi[i] = ct_malloc(L * sizeof(dcomplex));
        for (int j = 0; j < L; j++)
        {
            chi[i][j] = 33.0;
        }
    }

    struct gmap gmap;
    construct_gmap(chi, N, L, &gmap);
    
    // const long nblocks = integer_product(dt.dim, dt.ndim);
    // printf("[");
    // for (long k = 0; k < nblocks; k++)
    // {
    //     printf(" %.2f", creal(data[k]));
    // }
    // printf("]\n");

    // struct mps psi;
    // construct_computational_basis_mps_2d(CT_DOUBLE_COMPLEX, L, 0b11111111110000, &psi);

    // struct dense_tensor dt;
    // struct block_sparse_tensor bst;
    // mps_to_statevector(&psi, &bst);

    // block_sparse_to_dense_tensor(&bst, &dt);

    // dcomplex *data = dt.data;
    // for (int i = 0; i < dt.ndim; i++)
    // {
    //     if (i == 16368)
    //     {
    //         assert(creal(data[i] == 1.));
    //     }
    //     else
    //     {
    //         assert(creal(data[i] == 0.));
    //     }
    // }

    // print_dt(dt);
}