#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mps.h"
#include "mpo.h"
#include "operation.h"
#include "aligned_memory.h"

void print_dt(struct dense_tensor dt)
{
    printf("dt=[dtype=%d, ndim=%d, dim=(", dt.dtype, dt.ndim);
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

        double *data = (double *)(dt.data);
        // print_dt(dt);
        const long nblocks = integer_product(dt.dim, dt.ndim);
        printf("[");
        for (long k = 0; k < nblocks; k++)
        {
            printf(" %.1f ", data[k]);
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


struct gmap
{
    struct mpo **data;    // List of MPO pairs accessible like an dictionary with tuple key via get_value(...)
    long N;                 // THC Rank `N`. Length of `data` is `2N`
};


void allocate_gmap(struct gmap *g, const long N)
{
    assert(N > 0);
    g->N = N;
    g->data = ct_malloc(2 * N * sizeof(struct mpo*));
    for(size_t i = 0; i < 2 * N; i++) {
        g->data[i] = ct_malloc(2 * sizeof(struct mpo));
    }
}


void get_gmap_pair(const struct gmap *g, const int i, const int s, struct mpo **pair)
{
    assert(i + g->N * s < 2 * g->N);
    *(pair) = g->data[i + g->N * s];
}


void interleave_zero(const double *a, const long n, const long offset, double **ret)
{
    *ret = ct_calloc(2 * n, sizeof(double));
    for (size_t i = 0; i < n; i++)
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


void construct_computational_basis_mps_2d(const int nsites, const int basis_state, struct mps *mps)
{
    const long d = 2;
    const qnumber qsite[2] = {0, 1};

    const double state_zero[2] = {1, 0};
    const double state_one[2] = {0, 1};

    const int ndim = 3;
    const long dim[3] = {1, d, 1};

    const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };

    int acc = 0;

    qnumber qbond_curr[1] = {acc};
    qnumber qbond_next[1];

    allocate_empty_mps(nsites, d, qsite, mps);

    for (size_t i = 0; i < nsites; i++)
    {
        const int ith = ((basis_state & (1 << (nsites - i - 1))) >> (nsites - i - 1));

        struct dense_tensor dt;
        allocate_dense_tensor(CT_DOUBLE_REAL, ndim, dim, &dt);

        if (ith == 0)
        {
            memcpy(dt.data, &state_zero, 2 * sizeof(double));
        }
        else if (ith == 1)
        {
            memcpy(dt.data, &state_one, 2 * sizeof(double));
            acc++;
        }
        qbond_next[0] = acc;

        const qnumber *qnums[3] = {qbond_curr, qsite, qbond_next};
        dense_to_block_sparse_tensor(&dt, axis_dir, qnums, &mps->a[i]);

        delete_dense_tensor(&dt);

        qbond_curr[0] = qbond_next[0];
    }
}


void construct_gmap(const struct dense_tensor chi, const long N, const long L, struct gmap *g)
{
    allocate_gmap(g, N);
    
    for (size_t i = 0; i < N; i++)
    {
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
    apply_operator(mpo, psi, ret);
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


void read_data(double* zeta, double *chi)
{
    hid_t file = H5Fopen("../examples/thcmpo/data/water.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		printf("'H5Fopen' failed\n");
	}

    if (read_hdf5_dataset(file, "zeta", H5T_NATIVE_DOUBLE, zeta) < 0) {
        printf("can not read zeta\n");
	}

    if (read_hdf5_dataset(file, "chi", H5T_NATIVE_DOUBLE, chi) < 0) {
        printf("can not read chi\n");
	}
}


void read_validation_vector(double *phi)
{
    hid_t file = H5Fopen("../examples/thcmpo/data/validation.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		printf("'H5Fopen' failed\n");
	}

    if (read_hdf5_dataset(file, "phi", H5T_NATIVE_DOUBLE, phi) < 0) {
        printf("can not read zeta\n");
    }
}


void validate(struct mps *phi) 
{
    struct block_sparse_tensor phi_comp_bst;
    struct dense_tensor phi_comp;
    struct dense_tensor phi_val;
    
    const long phi_dim[3] = { 1, 16384, 1 };
    allocate_dense_tensor(CT_DOUBLE_REAL, 3, phi_dim, &phi_val);
    read_validation_vector((double*)phi_val.data);

    mps_to_statevector(phi, &phi_comp_bst);
    block_sparse_to_dense_tensor(&phi_comp_bst, &phi_comp);

    assert(dense_tensor_allclose(&phi_val, &phi_comp, 1e-13));
}


int main()
{
    const long N = 28;
    const long L = 14;

    // ζ
    struct dense_tensor zeta;
    const long zeta_dim[2] = { N, N };
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, zeta_dim, &zeta);

    // χ
    struct dense_tensor chi;
    const long chi_dim[2] = { N, L / 2 };
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, chi_dim, &chi);

    read_data((double*)zeta.data, (double*)chi.data);

    // G_{nu, sigma}
    struct gmap gmap;
    construct_gmap(chi, N, L, &gmap);

    // hartree fock state
    struct mps psi;
    construct_computational_basis_mps_2d(L, 0b11111111110000, &psi);
    printf("psi[nsites=%d, d=%ld, is_cons=%d]\n", psi.nsites, psi.d, mps_is_consistent(&psi));

    // phi
    struct mps phi;
    compute_phi(&psi, &gmap, zeta, N, 1e-3, LONG_MAX, &phi);
    printf("phi[nsites=%d, d=%ld, is_cons=%d]\n", phi.nsites, phi.d, mps_is_consistent(&phi));

    validate(&phi);

    delete_mps(&psi);
    delete_mps(&phi);
    delete_dense_tensor(&chi);
    delete_dense_tensor(&zeta);
}