#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#include "aligned_memory.h"
#include "chain_ops.h"
#include "mpo.h"
#include "mps.h"
#include "states.h"
#include "utils.h"

void construct_thc_mpo_edge(const int oid, const int cid, const int vids[2], struct mpo_graph_edge* edge) {
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
void construct_thc_mpo_assembly(const int nsites, const double* chi_row, const bool is_creation, struct mpo_assembly* assembly) {
	// physical quantum numbers (particle number)
	const long d = 2;
	const qnumber qsite[2] = {0, 1};

	// operator map
	const int OID_Id = 0; // identity
	const int OID_Z = 1;  // Pauli-Z
	const int OID_B = 2;  // bosonic creation or annihilation depending on `is_creation`

	const double z[4] = {1., 0., 0., -1.};           // Z
	const double creation[4] = {0., 0., 1., 0.};     // bosonic creation
	const double annihilation[4] = {0., 1., 0., 0.}; // bosonic annihilation

	const double coeffmap[] = {0., 1.}; // first two entries must always be 0 and 1

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
		assembly->graph.edges[0] = ct_malloc(2 * sizeof(struct mpo_graph_edge));
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
		assembly->graph.edges[i] = ct_malloc(3 * sizeof(struct mpo_graph_edge));
		assembly->graph.num_edges[i] = 3;

		assembly->graph.verts[i + 1] = ct_malloc(2 * sizeof(struct mpo_graph_vertex));
		assembly->graph.num_verts[i + 1] = 2;
		assembly->graph.verts[i + 1][0].qnum = 0;
		assembly->graph.verts[i + 1][1].qnum = is_creation ? 1 : -1;

		// e2
		{
			const int vids[] = {0, 0};
			construct_thc_mpo_edge(OID_Z, CID_ONE, vids, &assembly->graph.edges[i][0]);

			mpo_graph_vertex_add_edge(1, 0, &assembly->graph.verts[i][0]);
			mpo_graph_vertex_add_edge(0, 0, &assembly->graph.verts[i + 1][0]); // e2 <- v3
		}

		// e3
		{
			const int vids[] = {0, 1};
			construct_thc_mpo_edge(OID_B, 2 + i, vids, &assembly->graph.edges[i][1]);

			mpo_graph_vertex_add_edge(1, 1, &assembly->graph.verts[i][0]);
			mpo_graph_vertex_add_edge(0, 1, &assembly->graph.verts[i + 1][1]); // e3 <- v4
		}

		// e4
		{
			const int vids[] = {1, 1};
			construct_thc_mpo_edge(OID_Id, CID_ONE, vids, &assembly->graph.edges[i][2]);

			mpo_graph_vertex_add_edge(1, 2, &assembly->graph.verts[i][1]);
			mpo_graph_vertex_add_edge(0, 2, &assembly->graph.verts[i + 1][1]); // e4 <- v4
		}
	}

	{
		// right-most site
		// v3 [ -e5- v5 ]
		// v4 [ -e6-/   ]
		assembly->graph.edges[nsites - 1] = ct_malloc(2 * sizeof(struct mpo_graph_edge));
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

void construct_thc_mpo_assembly_4d(const int nsites, const double* chi_row, const bool is_creation, const bool is_spin_up, struct mpo_assembly* assembly) {
	// physical quantum numbers (particle number)
	const long d = 4;

	const qnumber qsite[] = {
		encode_quantum_number_pair(0, 0),
		encode_quantum_number_pair(1, -1),
		encode_quantum_number_pair(1, 1),
		encode_quantum_number_pair(2, 0),
	};

	qnumber qeff;

	// operator map
	const int OID_I4 = 0;
	const int OID_ZZ = 1;
	const int OID_PQ = 2;

	const double coeffmap[] = {0., 1.}; // first two entries must always be 0 and 1

	struct mpo_graph graph; // graph for MPO construction

	// allocate and set memory for physical quantum numbers
	assembly->d = d;
	assembly->dtype = CT_DOUBLE_REAL;
	assembly->qsite = ct_malloc(assembly->d * sizeof(qnumber));
	memcpy(assembly->qsite, &qsite, d * sizeof(qnumber));

	struct dense_tensor id;
	{
		const long dim[2] = {2, 2};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &id);
		dense_tensor_set_identity(&id);
	}

	struct dense_tensor z;
	{
		const long dim[2] = {2, 2};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &z);
		const double data[4] = {1., 0., 0., -1.};
		memcpy(z.data, data, sizeof(data));
	}

	struct dense_tensor creation;
	{
		const long dim[2] = {2, 2};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &creation);
		const double data[4] = {0., 0., 1., 0.};
		memcpy(creation.data, data, sizeof(data));
	}

	struct dense_tensor annihilation;
	{
		const long dim[2] = {2, 2};
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &annihilation);
		const double data[4] = {0., 1., 0., 0.};
		memcpy(annihilation.data, data, sizeof(data));
	}

	// allocate memory for operators
	assembly->num_local_ops = 3;
	assembly->opmap = ct_malloc(assembly->num_local_ops * sizeof(struct dense_tensor));
	dense_tensor_kronecker_product(&id, &id, &assembly->opmap[OID_I4]);
	dense_tensor_kronecker_product(&z, &z, &assembly->opmap[OID_ZZ]);

	if (is_creation) {
		if (is_spin_up) {
			dense_tensor_kronecker_product(&creation, &id, &assembly->opmap[OID_PQ]);
			qeff = qsite[2];
		} else {
			dense_tensor_kronecker_product(&z, &creation, &assembly->opmap[OID_PQ]);
			qeff = qsite[1];
		}
	} else {
		if (is_spin_up) {
			dense_tensor_kronecker_product(&annihilation, &id, &assembly->opmap[OID_PQ]);
			qeff = -qsite[2];
		} else {
			dense_tensor_kronecker_product(&z, &annihilation, &assembly->opmap[OID_PQ]);
			qeff = -qsite[1];
		}
	}

	// copy coefficents; first 2 entries must always be 0 and 1
	assembly->num_coeffs = 2 + nsites;
	assembly->coeffmap = ct_calloc(assembly->num_coeffs, sizeof(double));
	memcpy(assembly->coeffmap, coeffmap, 2 * sizeof(double));
	memcpy(((double*)assembly->coeffmap) + 2, chi_row, nsites * sizeof(double));

	// setup MPO graph
	assembly->graph.nsites = nsites;

	assembly->graph.num_edges = ct_calloc(nsites, sizeof(int));
	assembly->graph.edges = ct_calloc(nsites, sizeof(struct mpo_graph_edge*));

	assembly->graph.verts = ct_calloc(nsites + 1, sizeof(struct mpo_graph_vertex*));
	assembly->graph.num_verts = ct_calloc(nsites + 1, sizeof(int));

	// left-most site
	{
		// v0
		assembly->graph.verts[0] = ct_calloc(1, sizeof(struct mpo_graph_vertex));
		assembly->graph.num_verts[0] = 1;
		assembly->graph.verts[0]->qnum = 0;

		// (e0, e1)
		assembly->graph.edges[0] = ct_malloc(2 * sizeof(struct mpo_graph_edge));
		assembly->graph.num_edges[0] = 2;

		// (v2, v3)
		assembly->graph.verts[1] = ct_calloc(2, sizeof(struct mpo_graph_vertex));
		assembly->graph.num_verts[1] = 2;

		assembly->graph.verts[1][0].qnum = 0;
		assembly->graph.verts[1][1].qnum = qeff;

		{
			const int vids[] = {0, 0};
			construct_thc_mpo_edge(OID_ZZ, CID_ONE, vids, &assembly->graph.edges[0][0]);
			mpo_graph_vertex_add_edge(1, 0, &assembly->graph.verts[0][0]); // v0 -> e0
			mpo_graph_vertex_add_edge(0, 0, &assembly->graph.verts[1][0]); // e0 <- v2
		}

		{
			const int vids[] = {0, 1};
			construct_thc_mpo_edge(OID_PQ, 2, vids, &assembly->graph.edges[0][1]);
			mpo_graph_vertex_add_edge(1, 1, &assembly->graph.verts[0][0]); // v1 -> e1
			mpo_graph_vertex_add_edge(0, 1, &assembly->graph.verts[1][1]); // e1 <- v3
		}
	}

	// intermediate sites [2, L-2]
	for (size_t i = 1; i < nsites - 1; i++) {
		assembly->graph.edges[i] = ct_malloc(3 * sizeof(struct mpo_graph_edge));
		assembly->graph.num_edges[i] = 3;

		assembly->graph.verts[i + 1] = ct_calloc(2, sizeof(struct mpo_graph_vertex));
		assembly->graph.num_verts[i + 1] = 2;

		assembly->graph.verts[i + 1][0].qnum = 0;
		assembly->graph.verts[i + 1][1].qnum = qeff;

		{
			const int vids[] = {0, 0};
			construct_thc_mpo_edge(OID_ZZ, CID_ONE, vids, &assembly->graph.edges[i][0]);

			mpo_graph_vertex_add_edge(1, 0, &assembly->graph.verts[i][0]);
			mpo_graph_vertex_add_edge(0, 0, &assembly->graph.verts[i + 1][0]);
		}

		{
			const int vids[] = {0, 1};
			construct_thc_mpo_edge(OID_PQ, 2 + i, vids, &assembly->graph.edges[i][1]);

			mpo_graph_vertex_add_edge(1, 1, &assembly->graph.verts[i][0]);
			mpo_graph_vertex_add_edge(0, 1, &assembly->graph.verts[i + 1][1]);
		}

		{
			const int vids[] = {1, 1};
			construct_thc_mpo_edge(OID_I4, CID_ONE, vids, &assembly->graph.edges[i][2]);

			mpo_graph_vertex_add_edge(1, 2, &assembly->graph.verts[i][1]);
			mpo_graph_vertex_add_edge(0, 2, &assembly->graph.verts[i + 1][1]);
		}
	}

	// right-most site
	{
		assembly->graph.edges[nsites - 1] = ct_malloc(2 * sizeof(struct mpo_graph_edge));
		assembly->graph.num_edges[nsites - 1] = 2;

		assembly->graph.verts[nsites] = ct_calloc(1, sizeof(struct mpo_graph_vertex));
		assembly->graph.num_verts[nsites] = 1;
		assembly->graph.verts[nsites]->qnum = qeff;

		{
			const int vids[] = {0, 0};
			construct_thc_mpo_edge(OID_PQ, 2 + (nsites - 1), vids, &assembly->graph.edges[nsites - 1][0]);

			mpo_graph_vertex_add_edge(1, 0, &assembly->graph.verts[nsites - 1][0]);
			mpo_graph_vertex_add_edge(0, 0, &assembly->graph.verts[nsites][0]);
		}

		{
			const int vids[] = {1, 0};
			construct_thc_mpo_edge(OID_I4, CID_ONE, vids, &assembly->graph.edges[nsites - 1][1]);

			mpo_graph_vertex_add_edge(1, 1, &assembly->graph.verts[nsites - 1][1]);
			mpo_graph_vertex_add_edge(0, 1, &assembly->graph.verts[nsites][0]);
		}
	}

	assert(mpo_graph_is_consistent(&assembly->graph));

	delete_dense_tensor(&annihilation);
	delete_dense_tensor(&creation);
	delete_dense_tensor(&z);
	delete_dense_tensor(&id);
}

void interleave_zero(const double* a, const long n, const long offset, double** ret) {
	*ret = ct_calloc(2 * n, sizeof(double));
	for (size_t i = 0; i < n; i++) {
		(*ret)[offset + 2 * i] = a[i];
	}
}

long index_to_g_offset(const long N, const long i, const long s) {
	assert(i + N * s < 2 * N);
	return i + N * s;
}

void allocate_thc_mpo_map(const long N, struct mpo*** g) {
	(*g) = ct_malloc(2 * N * sizeof(struct mpo*));
	for (size_t i = 0; i < 2 * N; i++) {
		(*g)[i] = ct_malloc(2 * sizeof(struct mpo));
	}
}

void construct_g(const struct dense_tensor chi, const long N, const long L, struct mpo** g) {
	for (size_t i = 0; i < N; i++) {
		double* chi_row;

		// spin up
		{
			struct mpo_assembly assembly_p, assembly_q;

			const long g_off = index_to_g_offset(N, i, 0);

			const long index[2] = {i, 0};
			const long offset = tensor_index_to_offset(chi.ndim, chi.dim, index);
			interleave_zero(&((double*)chi.data)[offset], L, 0, &chi_row);

			construct_thc_mpo_assembly(L, chi_row, false, &assembly_p);
			construct_thc_mpo_assembly(L, chi_row, true, &assembly_q);

			mpo_from_assembly(&assembly_p, &g[g_off][0]);
			mpo_from_assembly(&assembly_q, &g[g_off][1]);

			delete_mpo_assembly(&assembly_p);
			delete_mpo_assembly(&assembly_q);
		}

		// spin down
		{
			struct mpo_assembly assembly_p, assembly_q;

			const long g_off = index_to_g_offset(N, i, 1);

			const long index[2] = {i, 0};
			const long offset = tensor_index_to_offset(chi.ndim, chi.dim, index);
			interleave_zero(&((double*)chi.data)[offset], L / 2, 1, &chi_row);

			construct_thc_mpo_assembly(L, chi_row, false, &assembly_p);
			construct_thc_mpo_assembly(L, chi_row, true, &assembly_q);

			mpo_from_assembly(&assembly_p, &g[g_off][0]);
			mpo_from_assembly(&assembly_q, &g[g_off][1]);

			delete_mpo_assembly(&assembly_p);
			delete_mpo_assembly(&assembly_q);
		}
	}
}

void construct_thc_mpo_map(const struct dense_tensor chi, const long N, const long L, struct mpo** g) {
	for (size_t i = 0; i < N; i++) {
		const long index[2] = {i, 0};
		const long chi_off = tensor_index_to_offset(chi.ndim, chi.dim, index);
		const double* chi_row = &((double*)chi.data)[chi_off];

		// spin up
		{
			struct mpo_assembly assembly_p, assembly_q;

			const long g_off = index_to_g_offset(N, i, 0);

			construct_thc_mpo_assembly_4d(L, chi_row, false, true, &assembly_p);
			construct_thc_mpo_assembly_4d(L, chi_row, true, true, &assembly_q);

			mpo_from_assembly(&assembly_p, &g[g_off][0]);
			mpo_from_assembly(&assembly_q, &g[g_off][1]);

			delete_mpo_assembly(&assembly_p);
			delete_mpo_assembly(&assembly_q);
		}

		// spin down
		{
			struct mpo_assembly assembly_p, assembly_q;

			const long g_off = index_to_g_offset(N, i, 1);

			construct_thc_mpo_assembly_4d(L, chi_row, false, false, &assembly_p);
			construct_thc_mpo_assembly_4d(L, chi_row, true, false, &assembly_q);

			mpo_from_assembly(&assembly_p, &g[g_off][0]);
			mpo_from_assembly(&assembly_q, &g[g_off][1]);

			delete_mpo_assembly(&assembly_p);
			delete_mpo_assembly(&assembly_q);
		}
	}
}

void apply_and_compress(const struct mps* psi, const struct mpo* mpo, const double tol, const long max_vdim, struct mps* ret) {
	double norm;
	double scaling;
	struct trunc_info* info = ct_calloc(psi->nsites, sizeof(struct trunc_info));

	apply_mpo(mpo, psi, ret);
	mps_compress(tol, max_vdim, MPS_ORTHONORMAL_LEFT, ret, &norm, &scaling, info);
	scale_block_sparse_tensor(&norm, &ret->a[0]);

	ct_free(info);
}

void add_and_compress(const struct mps* phi, const struct mps* psi, const double tol, const long max_vdim, struct mps* ret) {
	double norm;
	double scaling;
	struct trunc_info* info = ct_calloc(psi->nsites, sizeof(struct trunc_info));

	mps_add(phi, psi, ret);
	mps_compress(tol, max_vdim, MPS_ORTHONORMAL_LEFT, ret, &norm, &scaling, info);
	scale_block_sparse_tensor(&norm, &ret->a[0]);

	ct_free(info);
}

void apply_thc(const struct mps* psi, struct mpo** g, const struct dense_tensor zeta, const long N, const double tol, const long max_vdim, struct mps* phi) {
	for (size_t n = 0; n < N; n++) {
		for (size_t s1 = 0; s1 < 2; s1++) {
			struct mps a;
			struct mps b;

			// G_{ν, σ'}
			{
				const long g_off = index_to_g_offset(N, n, s1);
				apply_and_compress(psi, &g[g_off][0], tol, max_vdim, &a); // a = compress(p10@psi)
				apply_and_compress(&a, &g[g_off][1], tol, max_vdim, &b);  // b = compress(p11@a)
			}

			for (size_t m = 0; m < N; m++) {
				double alpha;
				{
					const long index[2] = {m, n};
					const long offset = tensor_index_to_offset(zeta.ndim, zeta.dim, index);
					alpha = 0.5 * ((double*)zeta.data)[offset];
				}

				struct mps b_copy;
				{
					copy_mps(&b, &b_copy);
					scale_block_sparse_tensor(&alpha, &(b_copy.a[0]));
				}

				for (size_t s2 = 0; s2 < 2; s2++) {
					struct mps c;
					struct mps d;
					struct mps phi_nxt;

					// G_{μ, σ}
					{
						const long g_off2 = index_to_g_offset(N, m, s2);
						apply_and_compress(&b_copy, &g[g_off2][0], tol, max_vdim, &c); // c = compress(p20@b)
						apply_and_compress(&c, &g[g_off2][1], tol, max_vdim, &d);      // d = compress(p21@c)
					}

					add_and_compress(phi, &d, tol, max_vdim, &phi_nxt);

					delete_mps(phi);
					delete_mps(&d);
					delete_mps(&c);

					*phi = phi_nxt;
				}

				delete_mps(&b_copy);
			}

			delete_mps(&a);
			delete_mps(&b);
		}
	}
}

void mps_add_combiner(struct mps* out, struct mps* in) {
	struct mps ret;
	add_and_compress(out, in, 1e-20, 75, &ret); // TODO: Specify parameters via preprocessor.
	// TODO: Delete MPS?
	*out = ret;
}

void mps_add_initializer(struct mps* priv, struct mps* orig) {
	copy_mps(orig, priv);
}

void apply_thc_omp(const struct mps* psi, struct mpo** g, const struct dense_tensor zeta, const long N, const double tol, const long max_vdim, struct mps* phi) {
	struct mps acc = *phi; // openmp reduction requires non-pointer type

#pragma omp declare reduction(mpsReduceAdd : struct mps : mps_add_combiner(&omp_out, &omp_in)) \
	initializer(mps_add_initializer(&omp_priv, &omp_orig))

#pragma omp parallel for collapse(4) shared(psi) reduction(mpsReduceAdd : acc)
	for (size_t n = 0; n < N; n++) {
		for (size_t s1 = 0; s1 < 2; s1++) {
			for (size_t m = 0; m < N; m++) {
				for (size_t s2 = 0; s2 < 2; s2++) {
					struct mps G_psi; // (.5 * ζ_{μ,ν}) * (G_{μ, σ}G_{ν, σ'})|ᴪ>
					{
						struct mps b;
						{
							// G_{ν, σ'}
							struct mps tmp;
							const long off = index_to_g_offset(N, n, s1);
							apply_and_compress(psi, &g[off][0], tol, max_vdim, &tmp);
							apply_and_compress(&tmp, &g[off][1], tol, max_vdim, &b);
							delete_mps(&tmp);
						}

						double alpha; // ɑ = .5 * ζ_{μ,ν}
						{
							const long index[2] = {m, n};
							const long offset = tensor_index_to_offset(zeta.ndim, zeta.dim, index);
							alpha = 0.5 * ((double*)zeta.data)[offset];
						}

						scale_block_sparse_tensor(&alpha, &(b.a[0]));

						{
							// G_{μ, σ}
							struct mps tmp;
							const long off = index_to_g_offset(N, m, s2);
							apply_and_compress(&b, &g[off][0], tol, max_vdim, &tmp);
							apply_and_compress(&tmp, &g[off][1], tol, max_vdim, &G_psi);
							delete_mps(&tmp);
						}

						delete_mps(&b);
					}

					struct mps acc_nxt;
					add_and_compress(&acc, &G_psi, tol, max_vdim, &acc_nxt);
					delete_mps(&acc); // delete old acc

					acc = acc_nxt;

					delete_mps(&G_psi);
				}
			}
		}
	} // implicit barrier

	*phi = acc;
}

void apply_thc_omp_no_reduc(const struct mps* psi, struct mpo** g, const struct dense_tensor zeta, const long N, const double tol, const long max_vdim, struct mps* phi) {
#pragma omp parallel for collapse(4) shared(psi)
	for (size_t n = 0; n < N; n++) {
		for (size_t s1 = 0; s1 < 2; s1++) {
			for (size_t m = 0; m < N; m++) {
				for (size_t s2 = 0; s2 < 2; s2++) {
					struct mps G_psi; // (.5 * ζ_{μ,ν}) * (G_{μ, σ}G_{ν, σ'})|ᴪ>
					{
						struct mps b;
						{
							// G_{ν, σ'}
							struct mps tmp;
							const long off = index_to_g_offset(N, n, s1);
							apply_and_compress(psi, &g[off][0], tol, max_vdim, &tmp);
							apply_and_compress(&tmp, &g[off][1], tol, max_vdim, &b);
							delete_mps(&tmp);
						}

						double alpha; // ɑ = .5 * ζ_{μ,ν}
						{
							const long index[2] = {m, n};
							const long offset = tensor_index_to_offset(zeta.ndim, zeta.dim, index);
							alpha = 0.5 * ((double*)zeta.data)[offset];
						}

						scale_block_sparse_tensor(&alpha, &(b.a[0]));

						{
							// G_{μ, σ}
							struct mps tmp;
							const long off = index_to_g_offset(N, m, s2);
							apply_and_compress(&b, &g[off][0], tol, max_vdim, &tmp);
							apply_and_compress(&tmp, &g[off][1], tol, max_vdim, &G_psi);
							delete_mps(&tmp);
						}

						delete_mps(&b);
					}

					delete_mps(&G_psi);
				}
			}
		}
	} // implicit barrier
}

void mps_add_combiner_prof(struct mps* out, struct mps* in) {
	double dur;
	struct timespec t0, t1;
	struct mps ret;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	add_and_compress(out, in, 1e-20, LONG_MAX, &ret);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	dur = (t1.tv_sec - t0.tv_sec);
	dur += (t1.tv_nsec - t0.tv_nsec) / 1000000000.0;

#pragma omp critical
	{
		printf("reduction;%f\n", dur);
	}
	*out = ret;
}

void apply_thc_omp_prof(const struct mps* psi, struct mpo** g, const struct dense_tensor zeta, const long N, const double tol, const long max_vdim, struct mps* phi) {
	struct mps acc = *phi; // openmp reduction requires non-pointer type

#pragma omp declare reduction(mpsReduceAdd : struct mps : mps_add_combiner_prof(&omp_out, &omp_in)) \
	initializer(mps_add_initializer(&omp_priv, &omp_orig))

#pragma omp parallel for collapse(4) shared(psi) reduction(mpsReduceAdd : acc)
	for (size_t n = 0; n < N; n++) {
		for (size_t s1 = 0; s1 < 2; s1++) {
			for (size_t m = 0; m < N; m++) {
				for (size_t s2 = 0; s2 < 2; s2++) {
					double dur;
					struct timespec t0, t1;

					struct mps G_psi; // (.5 * ζ_{μ,ν}) * (G_{μ, σ}G_{ν, σ'})|ᴪ>
					{
						struct mps b;

						clock_gettime(CLOCK_MONOTONIC, &t0);
						{
							// G_{ν, σ'}
							struct mps tmp;
							const long off = index_to_g_offset(N, n, s1);
							apply_and_compress(psi, &g[off][0], tol, max_vdim, &tmp);
							apply_and_compress(&tmp, &g[off][1], tol, max_vdim, &b);
							delete_mps(&tmp);
						}
						clock_gettime(CLOCK_MONOTONIC, &t1);
						dur = (t1.tv_sec - t0.tv_sec);
						dur += (t1.tv_nsec - t0.tv_nsec) / 1000000000.0;

#pragma omp critical
						{
							printf("apply;%f\n", dur);
						}

						double alpha; // ɑ = .5 * ζ_{μ,ν}
						{
							const long index[2] = {m, n};
							const long offset = tensor_index_to_offset(zeta.ndim, zeta.dim, index);
							alpha = 0.5 * ((double*)zeta.data)[offset];
						}

						clock_gettime(CLOCK_MONOTONIC, &t0);
						scale_block_sparse_tensor(&alpha, &(b.a[0]));
						clock_gettime(CLOCK_MONOTONIC, &t1);
						dur = (t1.tv_sec - t0.tv_sec);
						dur += (t1.tv_nsec - t0.tv_nsec) / 1000000000.0;

#pragma omp critical
						{
							printf("scale;%f\n", dur);
						}

						clock_gettime(CLOCK_MONOTONIC, &t0);
						{
							// G_{μ, σ}
							struct mps tmp;
							const long off = index_to_g_offset(N, m, s2);
							apply_and_compress(&b, &g[off][0], tol, max_vdim, &tmp);
							apply_and_compress(&tmp, &g[off][1], tol, max_vdim, &G_psi);
							delete_mps(&tmp);
						}
						clock_gettime(CLOCK_MONOTONIC, &t1);
						dur = (t1.tv_sec - t0.tv_sec);
						dur += (t1.tv_nsec - t0.tv_nsec) / 1000000000.0;

#pragma omp critical
						{
							printf("apply;%f\n", dur);
						}

						delete_mps(&b);
					}

					struct mps acc_nxt;
					clock_gettime(CLOCK_MONOTONIC, &t0);
					add_and_compress(&acc, &G_psi, tol, max_vdim, &acc_nxt);
					clock_gettime(CLOCK_MONOTONIC, &t1);
					dur = (t1.tv_sec - t0.tv_sec);
					dur += (t1.tv_nsec - t0.tv_nsec) / 1000000000.0;

#pragma omp critical
					{
						printf("add;%f\n", dur);
					}

					delete_mps(&acc); // delete old acc

					acc = acc_nxt;

					delete_mps(&G_psi);
				}
			}
		}
	} // implicit barrier

	*phi = acc;
}