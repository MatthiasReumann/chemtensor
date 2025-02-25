#include <math.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <stdio.h>
#include <time.h>

#include "aligned_memory.h"
#include "chain_ops.h"
#include "hamiltonian.h"
#include "mpo.h"
#include "mps.h"
#include "states.h"
#include "thcops.h"
#include "utils.h"

/// Helper declarations

void mps_add_combiner(struct mps* out, struct mps* in);
void mps_add_initializer(struct mps* priv, struct mps* orig);
void add_and_compress(const struct mps* phi, const struct mps* psi, const double tol, const long max_vdim, struct mps* ret);
void construct_thc_spin_mpos(const struct dense_tensor* chi, struct mpo* thc_mpos);
long index_to_g_offset(const long N, const size_t i, const size_t s);

/// Public Interface

void construct_thc_spin_hamiltonian(const struct dense_tensor* tkin, struct dense_tensor* zeta, const struct dense_tensor* chi, struct thc_spin_hamiltonian* hamiltonian) {
	const long N = chi->dim[0];
	const long L = chi->dim[1];

	// Setup
	hamiltonian->zeta = zeta;
	hamiltonian->thc_rank = N;
	hamiltonian->thc_mpos = ct_malloc(4 * hamiltonian->thc_rank * sizeof(struct mpo));

	// Kinetic MPO
	struct dense_tensor vint;
	struct mpo_assembly T_assembly;
	const long T_dim[] = {L, L, L, L};

	allocate_dense_tensor(CT_DOUBLE_REAL, 4, T_dim, &vint);
	construct_spin_molecular_hamiltonian_mpo_assembly(tkin, &vint, false, &T_assembly);
	mpo_from_assembly(&T_assembly, &hamiltonian->T);
	delete_mpo_assembly(&T_assembly);
	delete_dense_tensor(&vint);

	// THC MPOs
	construct_thc_spin_mpos(chi, hamiltonian->thc_mpos);
}

/// @brief Compute H|psi> = T|psi> + V|psi> using THC MPOs
/// @todo: T_psi and V_psi could be computed in parallel.
void apply_thc_spin_hamiltonian(const struct thc_spin_hamiltonian* hamiltonian, const struct mps* psi, const double tol, const long max_vdim, struct mps* ret) {
	double trunc_scale;
	struct trunc_info* info = ct_calloc(psi->nsites, sizeof(struct trunc_info));

	struct mps T_psi;
	struct mps V_psi;

	apply_mpo(&hamiltonian->T, psi, &T_psi); // Kinetic term
	mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, &T_psi, &trunc_scale, info);

	apply_thc_spin_coulomb(hamiltonian, psi, tol, max_vdim, &V_psi); // Coulomb term

	// Add individual terms
	add_and_compress(&T_psi, &V_psi, tol, max_vdim, ret);
}

void apply_thc_spin_coulomb(const struct thc_spin_hamiltonian* hamiltonian, const struct mps* psi, const double tol, const long max_vdim, struct mps* ret) {
	const long N = hamiltonian->thc_rank;
	const struct dense_tensor* zeta = hamiltonian->zeta;
#if defined(_OPENMP)
	struct mps acc = {.nsites = -1};

#pragma omp declare reduction(mpsReduceAdd : struct mps : mps_add_combiner(&omp_out, &omp_in)) \
	initializer(mps_add_initializer(&omp_priv, &omp_orig))
#pragma omp parallel for collapse(4) shared(psi, hamiltonian) reduction(mpsReduceAdd : acc)
	for (size_t n = 0; n < N; n++) {
		for (size_t tau = 0; tau < 2; tau++) {
			for (size_t m = 0; m < N; m++) {
				for (size_t sigma = 0; sigma < 2; sigma++) {
					struct mps full_layer_psi; // (.5 * ζ[μ, ν] * G[μ, σ, ν, τ])|ᴪ>
					struct mps half_layer_psi; // (.5 * ζ[μ, ν] * G[ν, τ])|ᴪ>
					struct mps tmp;

					double trunc_scale;
					struct trunc_info* info = ct_calloc(psi->nsites, sizeof(struct trunc_info));

					{
						const long off = index_to_g_offset(N, n, tau);
						apply_mpo(&hamiltonian->thc_mpos[off], psi, &tmp);
						apply_mpo(&hamiltonian->thc_mpos[off + 1], &tmp, &half_layer_psi);
						mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, &half_layer_psi, &trunc_scale, info);
						delete_mps(&tmp);
					}

					{
						const long off = index_to_g_offset(N, m, sigma);
						apply_mpo(&hamiltonian->thc_mpos[off], &half_layer_psi, &tmp);
						apply_mpo(&hamiltonian->thc_mpos[off + 1], &tmp, &full_layer_psi);
						mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, &full_layer_psi, &trunc_scale, info);
						delete_mps(&tmp);
					}

					const long alpha_idx[2] = {m, n};
					const long alpha_off = tensor_index_to_offset(zeta->ndim, zeta->dim, alpha_idx);
					const double alpha = 0.5 * ((double*)zeta->data)[alpha_off]; // ɑ = .5 * ζ[μ, ν]
					scale_block_sparse_tensor(&alpha, &(full_layer_psi.a[0]));

					mps_add_combiner(&acc, &full_layer_psi);

					ct_free(info);
					delete_mps(&half_layer_psi);
				}
			}
		}
	} // implicit barrier

	move_mps_data(&acc, ret);
#else
	double trunc_scale;
	struct trunc_info* info = ct_calloc(psi->nsites, sizeof(struct trunc_info));

	for (size_t n = 0; n < N; n++) {
		for (size_t tau = 0; tau < 2; tau++) {
			struct mps half_layer_psi; // (.5 * ζ[μ, ν] * G[ν, τ])|ᴪ>

			{
				struct mps tmp;
				const long off = index_to_g_offset(N, n, tau);
				apply_mpo(&hamiltonian->thc_mpos[off], psi, &tmp);
				apply_mpo(&hamiltonian->thc_mpos[off + 1], &tmp, &half_layer_psi);
				mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, &half_layer_psi, &trunc_scale, info);
				delete_mps(&tmp);
			}

			for (size_t m = 0; m < N; m++) {
				struct mps half_layer_psi_copy;

				const long alpha_idx[2] = {m, n};
				const long alpha_off = tensor_index_to_offset(zeta->ndim, zeta->dim, alpha_idx);
				const double alpha = 0.5 * ((double*)zeta->data)[alpha_off]; // ɑ = .5 * ζ[μ, ν]

				copy_mps(&half_layer_psi, &half_layer_psi_copy);
				scale_block_sparse_tensor(&alpha, &(half_layer_psi_copy.a[0]));

				for (size_t sigma = 0; sigma < 2; sigma++) {
					struct mps full_layer_psi; // (.5 * ζ[μ, ν] * G[μ, σ, ν, τ])|ᴪ>

					{
						struct mps tmp;
						const long off = index_to_g_offset(N, m, sigma);
						apply_mpo(&hamiltonian->thc_mpos[off], &half_layer_psi, &tmp);
						apply_mpo(&hamiltonian->thc_mpos[off + 1], &tmp, &full_layer_psi);
						mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, &full_layer_psi, &trunc_scale, info);
						delete_mps(&tmp);
					}

					struct mps acc_nxt;
					add_and_compress(ret, &full_layer_psi, tol, max_vdim, &acc_nxt);

					delete_mps(&full_layer_psi);
					delete_mps(ret);

					move_mps_data(&acc_nxt, ret);
				}

				delete_mps(&half_layer_psi_copy);
			}

			delete_mps(&half_layer_psi);
		}
	}
#endif
}

/// Helper implementations

void mps_add_combiner(struct mps* out, struct mps* in) {
	if (out->nsites == -1) { // uninitialized state
		move_mps_data(in, out);
	} else {
		struct mps ret;
		add_and_compress(out, in, 0, 250, &ret); // TODO: Specify parameters via preprocessor

		delete_mps(out);
		delete_mps(in);

		move_mps_data(&ret, out);
	}
}

void mps_add_initializer(struct mps* priv, struct mps* orig) {
	priv->nsites = orig->nsites;
}

/// @brief Compress and orthonormalize an MPS by site-local SVDs and singular value truncations.
/// @note Copy of 'mps_compress' but qr decomposition removed.
int mps_compress_no_qr(const double tol, const long max_vdim,
					   struct mps* mps, double* restrict trunc_scale, struct trunc_info* info) {
	const bool renormalize = false;

	for (int i = 0; i < mps->nsites - 1; i++) {
		int ret = mps_local_orthonormalize_left_svd(tol, max_vdim, renormalize, &mps->a[i], &mps->a[i + 1], &info[i]);
		if (ret < 0) {
			return ret;
		}
	}

	// last tensor
	const int i = mps->nsites - 1;
	assert(mps->a[i].dim_logical[2] == 1);

	// create a dummy "tail" tensor
	const long dim_tail[3] = {mps->a[i].dim_logical[2], 1, mps->a[i].dim_logical[2]};
	const enum tensor_axis_direction axis_dir_tail[3] = {TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN};
	qnumber qsite_tail[1] = {0};
	const qnumber* qnums_tail[3] = {mps->a[i].qnums_logical[2], qsite_tail, mps->a[i].qnums_logical[2]};
	struct block_sparse_tensor a_tail;
	allocate_block_sparse_tensor(mps->a[i].dtype, 3, dim_tail, axis_dir_tail, qnums_tail, &a_tail);
	assert(a_tail.dim_blocks[0] == 1 && a_tail.dim_blocks[1] == 1 && a_tail.dim_blocks[2] == 1);
	// set single entry to 1
	assert(a_tail.blocks[0] != NULL);
	memcpy(a_tail.blocks[0]->data, numeric_one(a_tail.blocks[0]->dtype), sizeof_numeric_type(a_tail.blocks[0]->dtype));

	// orthonormalize last MPS tensor
	int ret = mps_local_orthonormalize_left_svd(tol, max_vdim, renormalize, &mps->a[i], &a_tail, &info[i]);
	if (ret < 0) {
		return ret;
	}

	assert(a_tail.dtype == mps->a[i].dtype);
	assert(a_tail.dim_logical[0] == 1 && a_tail.dim_logical[1] == 1 && a_tail.dim_logical[2] == 1);
	// quantum numbers for 'a_tail' should match due to preceeding QR orthonormalization // <--- (???)

	assert(a_tail.blocks[0] != NULL);
	
	double d = *((double*)a_tail.blocks[0]->data);
	// absorb potential phase factor into MPS tensor
	if (d < 0) {
		scale_block_sparse_tensor(numeric_neg_one(CT_DOUBLE_REAL), &mps->a[i]);
	}
	(*trunc_scale) = fabs(d);

	delete_block_sparse_tensor(&a_tail);

	return 0;
}

void add_and_compress(const struct mps* phi, const struct mps* psi, const double tol, const long max_vdim, struct mps* ret) {
	double trunc_scale;
	struct trunc_info* info = ct_calloc(psi->nsites, sizeof(struct trunc_info));

	mps_add(phi, psi, ret);
	mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, ret, &trunc_scale, info);
	// mps_compress_no_qr(tol, max_vdim, ret, &trunc_scale, info);
	// rscale_block_sparse_tensor(&trunc_scale, &ret->a[ret->nsites - 1]);

	ct_free(info);
}

long index_to_g_offset(const long N, const size_t i, const size_t s) {
	return 2 * i + 2 * N * s;
}

void construct_thc_mpo_edge(const int oid, const int cid, const int vids[2], struct mpo_graph_edge* edge) {
	edge->nopics = 1;
	edge->opics = ct_malloc(sizeof(struct local_op_ref));
	edge->opics->oid = oid;
	edge->opics->cid = cid;
	edge->vids[0] = vids[0];
	edge->vids[1] = vids[1];
}

void interleave_zero(const double* a, const long n, const long offset, double** ret) {
	*ret = ct_calloc(2 * n, sizeof(double));
	for (size_t i = 0; i < n; i++) {
		(*ret)[offset + 2 * i] = a[i];
	}
}

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

void construct_thc_spin_mpo_assembly(const int nsites, const double* chi_row, const bool is_creation, const bool is_spin_up, struct mpo_assembly* assembly) {
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

void construct_thc_spin_mpos(const struct dense_tensor* chi, struct mpo* thc_mpos) {
	const long N = chi->dim[0];
	const long L = chi->dim[1];

	for (size_t i = 0; i < N; i++) {
		const long index[2] = {i, 0};
		const long chi_off = tensor_index_to_offset(chi->ndim, chi->dim, index);
		const double* chi_row = &((double*)chi->data)[chi_off];

		// spin up (0)
		{
			struct mpo_assembly assembly_p, assembly_q;

			const long g_off = index_to_g_offset(N, i, 0);

			construct_thc_spin_mpo_assembly(L, chi_row, false, true, &assembly_p);
			construct_thc_spin_mpo_assembly(L, chi_row, true, true, &assembly_q);

			mpo_from_assembly(&assembly_p, &thc_mpos[g_off]);
			mpo_from_assembly(&assembly_q, &thc_mpos[g_off + 1]);

			delete_mpo_assembly(&assembly_p);
			delete_mpo_assembly(&assembly_q);
		}

		// spin down (1)
		{
			struct mpo_assembly assembly_p, assembly_q;

			const long g_off = index_to_g_offset(N, i, 1);

			construct_thc_spin_mpo_assembly(L, chi_row, false, false, &assembly_p);
			construct_thc_spin_mpo_assembly(L, chi_row, true, false, &assembly_q);

			mpo_from_assembly(&assembly_p, &thc_mpos[g_off]);
			mpo_from_assembly(&assembly_q, &thc_mpos[g_off + 1]);

			delete_mpo_assembly(&assembly_p);
			delete_mpo_assembly(&assembly_q);
		}
	}
}