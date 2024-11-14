#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mps.h"
#include "mpo.h"
#include "chain_ops.h"
#include "aligned_memory.h"

#include "gmap.h"


void construct_thc_mpo_edge(const int oid, const int cid, const int vids[2], struct mpo_graph_edge* edge) 
{
    edge->nopics = 1;
    edge->opics = ct_malloc(sizeof(struct local_op_ref));
    edge->opics->oid = oid;
    edge->opics->cid = cid;
    edge->vids[0] = vids[0];
    edge->vids[1] = vids[1];
}

///             ╭ for s in [2, L-1]  ╮
///             ┊       ┌────┐       ┊
///             ┊   ┌───│ Z  │────┐  ┊
///      ┌────┐ ┊   │   └────┘    │  ┊ ┌────┐
///  ┌───│ Z  │────● 0         0 ●─────│ χb │────┐
///  │   └────┘ ┊   │   ┌────┐       ┊ └────┘    │
///  ● 0        ┊   └───│ χb │────┐  ┊           ● 0 
///  │   ┌────┐ ┊       └────┘    │  ┊ ┌────┐    │
///  └───│ χb │────● 1  ┌────┐  1 ●────│ I  │────┘
///      └────┘ ┊  └────│ I  │────┘  ┊ └────┘
///             ┊       └────┘       ┊
///             ╰                    ╯
void construct_thc_mpo_assembly(const int nsites, const double *chi_row, const bool is_creation, struct mpo_assembly *assembly)
{
    // physical quantum numbers (particle number)
    const long d = 2;
    const qnumber qsite[2] = {0, 1};

    // operator map
    const int OID_Id = 0; // identity
    const int OID_Z = 1;  // Pauli-Z
    const int OID_B = 2;  // bosonic creation or annihilation depending on `is_creation`

    const double z[4] = {1., 0., 0., -1.};            // Z
    const double creation[4] = {0., 0., 1., 0.};     // bosonic creation
    const double annihilation[4] = {0., 1., 0., 0.}; // bosonic annihilation
    
    const double coeffmap[] = { 0., 1. }; // first two entries must always be 0 and 1

    struct mpo_graph graph; // graph for MPO construction

    // allocate and set memory for physical quantum numbers
    assembly->d = d;
    assembly->dtype = CT_DOUBLE_REAL;
    assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
    memcpy(assembly->qsite, &qsite, d * sizeof(qnumber));

    // allocate memory for operators
    assembly->num_local_ops = 3;
    assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
    for (size_t i = 0; i < assembly->num_local_ops; i++) {
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
    assembly->coeffmap = ct_calloc(assembly->num_coeffs, sizeof(double));
    memcpy(assembly->coeffmap, coeffmap, 2 * sizeof(double));
    memcpy(((double*)assembly->coeffmap) + 2, chi_row, nsites * sizeof(double));

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
    for (size_t i = 1; i < nsites - 1; i++) {
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

    assert(mpo_graph_is_consistent(&assembly->graph));
}


void interleave_zero(const double *a, const long n, const long offset, double **ret)
{
    *ret = ct_calloc(2 * n, sizeof(double));
    for (size_t i = 0; i < n; i++)
    {
        (*ret)[offset + 2 * i] = a[i];
    }
}


void construct_gmap(const struct dense_tensor chi, const long N, const long L, struct gmap *g)
{
    allocate_gmap(g, N);
    
    for (size_t i = 0; i < N; i++) {
        double *chi_row;
        // spin up
        {
            struct mpo *pair;
            struct mpo_assembly assembly_p, assembly_q;
            
            const long index[2] = { i, 0 };
            const long offset = tensor_index_to_offset(chi.ndim, chi.dim, index);
            interleave_zero(&((double*)chi.data)[offset], L / 2, 0, &chi_row);

            construct_thc_mpo_assembly(L, chi_row, false, &assembly_p);
            construct_thc_mpo_assembly(L, chi_row, true, &assembly_q);
            
            get_gmap_pair(g, i, 0, &pair);
            mpo_from_assembly(&assembly_p, &pair[0]);
            mpo_from_assembly(&assembly_q, &pair[1]);
            
            delete_mpo_assembly(&assembly_p);
            delete_mpo_assembly(&assembly_q);
        }

        // spin down
        {
            struct mpo *pair;
            struct mpo_assembly assembly_p, assembly_q;
            
            const long index[2] = { i, 0 };
            const long offset = tensor_index_to_offset(chi.ndim, chi.dim, index);
            interleave_zero(&((double*)chi.data)[offset], L / 2, 1, &chi_row);
            
            construct_thc_mpo_assembly(L, chi_row, false, &assembly_p);
            construct_thc_mpo_assembly(L, chi_row, true, &assembly_q);
            
            get_gmap_pair(g, i, 1, &pair);
            mpo_from_assembly(&assembly_p, &pair[0]);
            mpo_from_assembly(&assembly_q, &pair[1]);
            
            delete_mpo_assembly(&assembly_p);
            delete_mpo_assembly(&assembly_q);
        }
    }
}


void contract_layer(const struct mps *psi, const struct mpo *mpo, const double tol, const long max_vdim, struct mps *ret) 
{
    double norm;
    double scale;
    struct trunc_info *info = ct_calloc(psi->nsites, sizeof(struct trunc_info));
    apply_mpo(mpo, psi, ret);
    mps_compress(tol, max_vdim, MPS_ORTHONORMAL_LEFT, ret, &norm, &scale, info);

    ct_free(info);
}


void add_partial(const struct mps *phi, const struct mps *psi, const double tol, const long max_vdim, struct mps *ret)
{
    double norm;
    double scale;
    struct trunc_info *info = ct_calloc(psi->nsites, sizeof(struct trunc_info));
    mps_add(phi, psi, ret);
    mps_compress(tol, max_vdim, MPS_ORTHONORMAL_LEFT, ret, &norm, &scale, info);

    ct_free(info);
}


void copy_mps(const struct mps *mps, struct mps* ret)
{
    ret->d = mps->d;
    ret->nsites = mps->nsites;
    ret->qsite = ct_malloc(mps->nsites * sizeof(qnumber));
    ret->a = ct_malloc(mps->nsites * sizeof(struct block_sparse_tensor));
    for (size_t i = 0; i < mps->nsites; i++) {
        ret->qsite[i] = mps->qsite[i]; 
        copy_block_sparse_tensor(&mps->a[i], &ret->a[i]);
    }
}


void compute_phi(const struct mps *psi, const struct gmap *gmap, const struct dense_tensor zeta, const long N, const double tol, const long max_vdim, struct mps *phi)
{
    const int S = 2; // |{UP, DOWN}| = 2
    
    for (size_t n = 0; n < N; n++) {
        for (size_t s1 = 0; s1 < S; s1++) {
            struct mps a;
            struct mps b;
            struct mpo *pair;

            get_gmap_pair(gmap, n, s1, &pair);

            contract_layer(psi, &pair[0], tol, max_vdim, &a); // a = compress(p10@psi)
            contract_layer(&a, &pair[1], tol, max_vdim, &b);  // b = compress(p11@a)

            for (size_t m = 0; m < N; m++) {
                for (size_t s2 = 0; s2 < S; s2++) {
                    double alpha;
                    struct mps b_copy;
                    struct mps c;
                    struct mps d;
                    struct mpo *pair2;
                    
                    const long index[2] = { m, n };
                    const long offset = tensor_index_to_offset(zeta.ndim, zeta.dim, index);
                    alpha = 0.5 * ((double*)zeta.data)[offset];

                    copy_mps(&b, &b_copy);
                    scale_block_sparse_tensor(&alpha, &(b_copy.a[0]));

                    get_gmap_pair(gmap, m, s2, &pair2);
                    contract_layer(&b_copy, &pair2[0], tol, max_vdim, &c); // c = compress(p20@b)
                    contract_layer(&c, &pair2[1], tol, max_vdim, &d); // d = compress(p21@c)
                    
                    delete_mps(&c);
                    delete_mps(&b_copy);
                    
                    if (n == 0 && s1 == 0 && m == 0 && s2 == 0) {
                        *phi = d;
                    } else {
                        struct mps new_phi;

                        add_partial(phi, &d, tol, max_vdim, &new_phi);
                        
                        delete_mps(phi);
                        delete_mps(&d);

                        *phi = new_phi;
                    }
                }
            }

            delete_mps(&a);
            delete_mps(&b);
        }
    }
}
