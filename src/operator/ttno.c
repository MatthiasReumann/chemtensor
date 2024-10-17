/// \file ttno.c
/// \brief Tree tensor network operator (TTNO) data structure and functions.

#include "ttno.h"
#include "abstract_graph.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Construct a TTNO from a TTNO assembly.
///
void ttno_from_assembly(const struct ttno_assembly* assembly, struct ttno* ttno)
{
	assert(assembly->graph.nsites_physical  >= 1);
	assert(assembly->graph.nsites_branching >= 0);
	assert(assembly->d >= 1);
	const long d = assembly->d;
	ttno->d = d;
	ttno->nsites_physical  = assembly->graph.nsites_physical;
	ttno->nsites_branching = assembly->graph.nsites_branching;

	assert(coefficient_map_is_valid(assembly->dtype, assembly->coeffmap));

	// overall number of sites
	const int nsites = assembly->graph.nsites_physical + assembly->graph.nsites_branching;

	// tree topology
	copy_abstract_graph(&assembly->graph.topology, &ttno->topology);

	ttno->qsite = ct_malloc(d * sizeof(qnumber));
	memcpy(ttno->qsite, assembly->qsite, d * sizeof(qnumber));

	ttno->a = ct_calloc(nsites, sizeof(struct block_sparse_tensor));

	for (int l = 0; l < nsites; l++)
	{
		const int offset_phys = (l < assembly->graph.nsites_physical ? 2 : 0);

		// accumulate entries in a dense tensor first
		const int ndim_a_loc = assembly->graph.topology.num_neighbors[l] + offset_phys;
		long* dim_a_loc = ct_malloc(ndim_a_loc * sizeof(long));
		for (int i = 0; i < ndim_a_loc; i++) {
			dim_a_loc[i] = d;
		}
		long stride_trail = 1;
		// overwrite with actual virtual bond dimensions
		for (int i = 0; i < assembly->graph.topology.num_neighbors[l]; i++) {
			int k = assembly->graph.topology.neighbor_map[l][i];
			assert(k != l);
			if (i > 0) {
				assert(assembly->graph.topology.neighbor_map[l][i - 1] < k);
			}
			if (k < l) {
				dim_a_loc[i] = assembly->graph.num_verts[k*nsites + l];
			}
			else {
				dim_a_loc[i + offset_phys] = assembly->graph.num_verts[l*nsites + k];
				stride_trail *= dim_a_loc[i + offset_phys];
			}
		}
		struct dense_tensor a_loc;
		allocate_dense_tensor(assembly->dtype, ndim_a_loc, dim_a_loc, &a_loc);
		ct_free(dim_a_loc);

		for (int n = 0; n < assembly->graph.num_edges[l]; n++)
		{
			const struct ttno_graph_hyperedge* edge = &assembly->graph.edges[l][n];
			assert(edge->order == assembly->graph.topology.num_neighbors[l]);
			struct dense_tensor op;
			construct_local_operator(edge->opics, edge->nopics, assembly->opmap, assembly->coeffmap, &op);

			assert(op.ndim == 2);
			assert(op.dim[0] == op.dim[1]);
			assert(op.dim[0] == (l < assembly->graph.nsites_physical ? d : 1));
			assert(op.dtype == a_loc.dtype);

			// add entries of local operator 'op' to 'a_loc' (supporting multiple hyperedges between same nodes)
			long* index_start = ct_calloc(a_loc.ndim, sizeof(long));
			for (int i = 0; i < edge->order; i++) {
				int k = assembly->graph.topology.neighbor_map[l][i];
				assert(k != l);
				index_start[k < l ? i : i + offset_phys] = edge->vids[i];
				assert(0 <= edge->vids[i] && edge->vids[i] < a_loc.dim[k < l ? i : i + offset_phys]);
			}
			long offset = tensor_index_to_offset(a_loc.ndim, a_loc.dim, index_start);
			ct_free(index_start);
			switch (a_loc.dtype)
			{
				case CT_SINGLE_REAL:
				{
					float* al_data = a_loc.data;
					const float* op_data = op.data;
					for (long j = 0; j < op.dim[0]*op.dim[1]; j++, offset += stride_trail)
					{
						al_data[offset] += op_data[j];
					}
					break;
				}
				case CT_DOUBLE_REAL:
				{
					double* al_data = a_loc.data;
					const double* op_data = op.data;
					for (long j = 0; j < op.dim[0]*op.dim[1]; j++, offset += stride_trail)
					{
						al_data[offset] += op_data[j];
					}
					break;
				}
				case CT_SINGLE_COMPLEX:
				{
					scomplex* al_data = a_loc.data;
					const scomplex* op_data = op.data;
					for (long j = 0; j < op.dim[0]*op.dim[1]; j++, offset += stride_trail)
					{
						al_data[offset] += op_data[j];
					}
					break;
				}
				case CT_DOUBLE_COMPLEX:
				{
					dcomplex* al_data = a_loc.data;
					const dcomplex* op_data = op.data;
					for (long j = 0; j < op.dim[0]*op.dim[1]; j++, offset += stride_trail)
					{
						al_data[offset] += op_data[j];
					}
					break;
				}
				default:
				{
					// unknown data type
					assert(false);
				}
			}

			delete_dense_tensor(&op);
		}

		// note: entries not adhering to the quantum number sparsity pattern are ignored
		qnumber** qnums = ct_calloc(assembly->graph.topology.num_neighbors[l] + offset_phys, sizeof(qnumber*));
		// virtual bond axis is oriented towards smaller site index
		enum tensor_axis_direction* axis_dir = ct_malloc((assembly->graph.topology.num_neighbors[l] + offset_phys) * sizeof(enum tensor_axis_direction));
		for (int i = 0; i < assembly->graph.topology.num_neighbors[l]; i++)
		{
			int k = assembly->graph.topology.neighbor_map[l][i];
			if (k < l) {
				const int iv = k*nsites + l;
				qnums[i] = ct_malloc(assembly->graph.num_verts[iv] * sizeof(qnumber));
				for (int n = 0; n < assembly->graph.num_verts[iv]; n++)
				{
					qnums[i][n] = assembly->graph.verts[iv][n].qnum;
				}
				axis_dir[i] = TENSOR_AXIS_OUT;
			}
			else {
				const int iv = l*nsites + k;
				qnums[i + offset_phys] = ct_malloc(assembly->graph.num_verts[iv] * sizeof(qnumber));
				for (int n = 0; n < assembly->graph.num_verts[iv]; n++)
				{
					qnums[i + offset_phys][n] = assembly->graph.verts[iv][n].qnum;
				}
				axis_dir[i + offset_phys] = TENSOR_AXIS_IN;
			}
		}
		// physical axes at current site
		if (l < assembly->graph.nsites_physical)
		{
			bool site_info_set = false;
			for (int i = 0; i < assembly->graph.topology.num_neighbors[l] + offset_phys; i++)
			{
				if (qnums[i] == NULL) {
					assert(qnums[i + 1] == NULL);
					qnums[i]     = ttno->qsite;
					qnums[i + 1] = ttno->qsite;
					axis_dir[i]     = TENSOR_AXIS_OUT;
					axis_dir[i + 1] = TENSOR_AXIS_IN;
					site_info_set = true;
					break;
				}
			}
			assert(site_info_set);
		}
		dense_to_block_sparse_tensor(&a_loc, axis_dir, (const qnumber**)qnums, &ttno->a[l]);
		for (int i = 0; i < assembly->graph.topology.num_neighbors[l]; i++)
		{
			int k = assembly->graph.topology.neighbor_map[l][i];
			ct_free(qnums[k < l ? i : i + offset_phys]);
		}
		ct_free(qnums);
		ct_free(axis_dir);

		#ifdef DEBUG
		struct dense_tensor a_loc_conv;
		block_sparse_to_dense_tensor(&ttno->a[l], &a_loc_conv);
		if (!dense_tensor_allclose(&a_loc_conv, &a_loc, 0.)) {
			fprintf(stderr, "Warning: ignoring non-zero tensor entries due to the quantum number sparsity pattern in 'ttno_from_graph', site %i\n", l);
		}
		delete_dense_tensor(&a_loc_conv);
		#endif

		delete_dense_tensor(&a_loc);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a tree tensor network operator (free memory).
///
void delete_ttno(struct ttno* ttno)
{
	// overall number of sites
	const int nsites = ttno->nsites_physical + ttno->nsites_branching;

	for (int l = 0; l < nsites; l++)
	{
		delete_block_sparse_tensor(&ttno->a[l]);
	}
	ct_free(ttno->a);
	ttno->a = NULL;

	delete_abstract_graph(&ttno->topology);

	ttno->nsites_physical  = 0;
	ttno->nsites_branching = 0;

	ct_free(ttno->qsite);
	ttno->qsite = NULL;
	ttno->d = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill the axis descriptions of a TTNO tensor; 'desc' must point to an array of the same length as the degree of the tensor at 'i_site'.
///
void ttno_tensor_get_axis_desc(const struct ttno* ttno, const int i_site, struct ttno_tensor_axis_desc* desc)
{
	// overall number of sites
	const int nsites = ttno->nsites_physical + ttno->nsites_branching;

	const int offset_phys = (i_site < ttno->nsites_physical ? 2 : 0);

	assert(0 <= i_site && i_site < nsites);
	assert(ttno->a[i_site].ndim == ttno->topology.num_neighbors[i_site] + offset_phys);

	// set to default type
	for (int i = 0; i < ttno->a[i_site].ndim; i++) {
		desc[i].type = TTNO_TENSOR_AXIS_PHYS_OUT;
	}

	// virtual bonds to neighbors
	for (int i = 0; i < ttno->topology.num_neighbors[i_site]; i++)
	{
		if (i > 0) {
			assert(ttno->topology.neighbor_map[i_site][i - 1] < ttno->topology.neighbor_map[i_site][i]);
		}
		int k = ttno->topology.neighbor_map[i_site][i];
		assert(k != i_site);
		desc[k < i_site ? i : i + offset_phys].type  = TTNO_TENSOR_AXIS_VIRTUAL;
		desc[k < i_site ? i : i + offset_phys].index = k;
	}
	// physical axes at current site
	if (i_site < ttno->nsites_physical)
	{
		bool site_info_set = false;
		for (int i = 0; i < ttno->a[i_site].ndim - 1; i++)
		{
			if (desc[i].type == TTNO_TENSOR_AXIS_PHYS_OUT) {
				assert(desc[i + 1].type == TTNO_TENSOR_AXIS_PHYS_OUT);
				desc[i + 1].type = TTNO_TENSOR_AXIS_PHYS_IN;
				desc[i    ].index = i_site;
				desc[i + 1].index = i_site;
				site_info_set = true;
				break;
			}
		}
		assert(site_info_set);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the TTNO data structure.
///
bool ttno_is_consistent(const struct ttno* ttno)
{
	if (ttno->nsites_physical <= 0) {
		return false;
	}
	if (ttno->nsites_branching < 0) {
		return false;
	}
	// overall number of sites
	const int nsites = ttno->nsites_physical + ttno->nsites_branching;

	if (ttno->d <= 0) {
		return false;
	}

	// topology
	if (ttno->topology.num_nodes != nsites) {
		return false;
	}
	if (!abstract_graph_is_consistent(&ttno->topology)) {
		return false;
	}
	// verify tree topology
	if (!abstract_graph_is_connected_tree(&ttno->topology)) {
		return false;
	}

	struct ttno_tensor_axis_desc** axis_desc = ct_malloc(nsites * sizeof(struct ttno_tensor_axis_desc*));
	for (int l = 0; l < nsites; l++)
	{
		axis_desc[l] = ct_malloc(ttno->a[l].ndim * sizeof(struct ttno_tensor_axis_desc));
		ttno_tensor_get_axis_desc(ttno, l, axis_desc[l]);
	}

	for (int l = 0; l < nsites; l++)
	{
		const int offset_phys = (l < ttno->nsites_physical ? 2 : 0);

		if (ttno->a[l].ndim != ttno->topology.num_neighbors[l] + offset_phys) {
			return false;
		}

		// quantum numbers for physical legs of individual tensors must agree with 'qsite'
		for (int i = 0; i < ttno->a[l].ndim; i++)
		{
			if (axis_desc[l][i].type == TTNO_TENSOR_AXIS_PHYS_OUT || axis_desc[l][i].type == TTNO_TENSOR_AXIS_PHYS_IN)
			{
				if (ttno->a[l].dim_logical[i] != ttno->d) {
					return false;
				}
				if (!qnumber_all_equal(ttno->d, ttno->a[l].qnums_logical[i], ttno->qsite)) {
					return false;
				}
				if (ttno->a[l].axis_dir[i] != (axis_desc[l][i].type == TTNO_TENSOR_AXIS_PHYS_OUT ? TENSOR_AXIS_OUT : TENSOR_AXIS_IN)) {
					return false;
				}
			}
		}

		// virtual bond quantum numbers and axis directions must match
		for (int n = 0; n < ttno->topology.num_neighbors[l]; n++)
		{
			const int k = ttno->topology.neighbor_map[l][n];
			assert(k != l);

			// find respective axis indices
			int i_ax_neigh_lk = -1;
			for (int i = 0; i < ttno->a[l].ndim; i++) {
				if (axis_desc[l][i].type == TTNO_TENSOR_AXIS_VIRTUAL && axis_desc[l][i].index == k) {
					i_ax_neigh_lk = i;
					break;
				}
			}
			if (i_ax_neigh_lk == -1) {
				return false;
			}
			int i_ax_neigh_kl = -1;
			for (int i = 0; i < ttno->a[k].ndim; i++) {
				if (axis_desc[k][i].type == TTNO_TENSOR_AXIS_VIRTUAL && axis_desc[k][i].index == l) {
					i_ax_neigh_kl = i;
					break;
				}
			}
			if (i_ax_neigh_kl == -1) {
				return false;
			}

			if (ttno->a[l].dim_logical[i_ax_neigh_lk] != ttno->a[k].dim_logical[i_ax_neigh_kl]) {
				return false;
			}
			if (!qnumber_all_equal(ttno->a[l].dim_logical[i_ax_neigh_lk],
					ttno->a[l].qnums_logical[i_ax_neigh_lk],
					ttno->a[k].qnums_logical[i_ax_neigh_kl])) {
				return false;
			}
			if (ttno->a[l].axis_dir[i_ax_neigh_lk] != (k < l ? TENSOR_AXIS_OUT : TENSOR_AXIS_IN)) {
				return false;
			}
			if (ttno->a[k].axis_dir[i_ax_neigh_kl] != (l < k ? TENSOR_AXIS_OUT : TENSOR_AXIS_IN)) {
				return false;
			}
		}
	}

	for (int l = 0; l < nsites; l++)
	{
		ct_free(axis_desc[l]);
	}
	ct_free(axis_desc);

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Contracted subtree of a TTNO (auxiliary data structure used for contraction).
///
struct ttno_contracted_subtree
{
	struct block_sparse_tensor tensor;        //!< subtree tensor
	struct ttno_tensor_axis_desc* axis_desc;  //!< axis descriptions
};


//________________________________________________________________________________________________________________________
///
/// \brief In-place transpose (permute) tensor axes of a contracted subtree.
///
static void transpose_ttno_contracted_subtree(const int* perm, struct ttno_contracted_subtree* subtree)
{
	struct block_sparse_tensor t;
	transpose_block_sparse_tensor(perm, &subtree->tensor, &t);
	delete_block_sparse_tensor(&subtree->tensor);
	move_block_sparse_tensor_data(&t, &subtree->tensor);

	// update axis descriptions
	struct ttno_tensor_axis_desc* new_axis_desc = ct_malloc(subtree->tensor.ndim * sizeof(struct ttno_tensor_axis_desc));
	for (int i = 0; i < subtree->tensor.ndim; i++) {
		new_axis_desc[i] = subtree->axis_desc[perm[i]];
	}
	ct_free(subtree->axis_desc);
	subtree->axis_desc = new_axis_desc;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a contracted subtree (free memory).
///
static void delete_ttno_contracted_subtree(struct ttno_contracted_subtree* subtree)
{
	delete_block_sparse_tensor(&subtree->tensor);
	ct_free(subtree->axis_desc);
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure for sorting axis descriptions by site index and type.
///
struct ttno_indexed_tensor_axis_desc
{
	struct ttno_tensor_axis_desc axis_desc;
	int index;
};

//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_ttno_indexed_tensor_axis_desc(const void* a, const void* b)
{
	const struct ttno_indexed_tensor_axis_desc* x = (const struct ttno_indexed_tensor_axis_desc*)a;
	const struct ttno_indexed_tensor_axis_desc* y = (const struct ttno_indexed_tensor_axis_desc*)b;

	if (x->axis_desc.index < y->axis_desc.index) {
		return -1;
	}
	if (x->axis_desc.index > y->axis_desc.index) {
		return 1;
	}
	// x->axis_desc.index == y->axis_desc.index

	if (x->axis_desc.type < y->axis_desc.type) {
		return -1;
	}
	if (x->axis_desc.type > y->axis_desc.type) {
		return 1;
	}
	// x->axis_desc.type == y->axis_desc.type

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Recursively contract a subtree of a TTNO starting from 'i_site'.
///
static void ttno_contract_subtree(const struct ttno* ttno, const int i_site, const int i_parent, struct ttno_contracted_subtree* contracted)
{
	copy_block_sparse_tensor(&ttno->a[i_site], &contracted->tensor);
	contracted->axis_desc = ct_malloc(contracted->tensor.ndim * sizeof(struct ttno_tensor_axis_desc));
	ttno_tensor_get_axis_desc(ttno, i_site, contracted->axis_desc);

	// merge child subtrees into current subtree
	for (int i = 0; i < ttno->topology.num_neighbors[i_site]; i++)
	{
		const int k = ttno->topology.neighbor_map[i_site][i];

		if (k == i_parent) {
			continue;
		}

		// recursive function call
		struct ttno_contracted_subtree child;
		ttno_contract_subtree(ttno, k, i_site, &child);

		// find axis index of 'contracted' connecting to 'child'
		int i_ax_c = -1;
		for (int j = 0; j < contracted->tensor.ndim; j++)
		{
			if (contracted->axis_desc[j].type  == TTNO_TENSOR_AXIS_VIRTUAL &&
			    contracted->axis_desc[j].index == k) {
				i_ax_c = j;
				break;
			}
		}
		assert(i_ax_c != -1);

		// find axis index of 'child' connecting to 'contracted'
		int i_ax_p = -1;
		for (int j = 0; j < child.tensor.ndim; j++)
		{
			if (child.axis_desc[j].type  == TTNO_TENSOR_AXIS_VIRTUAL &&
			    child.axis_desc[j].index == i_site) {
				i_ax_p = j;
				break;
			}
		}
		assert(i_ax_p != -1);

		enum tensor_axis_range axrange_c;
		if (i_ax_p == 0) {
			axrange_c = TENSOR_AXIS_RANGE_LEADING;
		}
		else if (i_ax_p == child.tensor.ndim - 1) {
			axrange_c = TENSOR_AXIS_RANGE_TRAILING;
		}
		else {
			axrange_c = TENSOR_AXIS_RANGE_TRAILING;

			// transpose child tensor such that to-be contracted axis is the trailing axis
			int* perm = ct_malloc(child.tensor.ndim * sizeof(int));
			for (int j = 0; j < child.tensor.ndim - 1; j++)
			{
				perm[j] = (j < i_ax_p ? j : j + 1);
			}
			perm[child.tensor.ndim - 1] = i_ax_p;

			transpose_ttno_contracted_subtree(perm, &child);

			ct_free(perm);
		}

		// contract current tensor with child
		struct block_sparse_tensor t;
		block_sparse_tensor_multiply_axis(&contracted->tensor, i_ax_c, &child.tensor, axrange_c, &t);

		// update axis descriptions
		struct ttno_tensor_axis_desc* new_axis_desc = ct_malloc(t.ndim * sizeof(struct ttno_tensor_axis_desc));
		memcpy( new_axis_desc, contracted->axis_desc, i_ax_c * sizeof(struct ttno_tensor_axis_desc));
		memcpy(&new_axis_desc[i_ax_c], &child.axis_desc[axrange_c == TENSOR_AXIS_RANGE_LEADING ? 1 : 0], (child.tensor.ndim - 1) * sizeof(struct ttno_tensor_axis_desc));
		memcpy(&new_axis_desc[i_ax_c + child.tensor.ndim - 1], &contracted->axis_desc[i_ax_c + 1], (contracted->tensor.ndim - i_ax_c - 1) * sizeof(struct ttno_tensor_axis_desc));

		delete_block_sparse_tensor(&contracted->tensor);
		move_block_sparse_tensor_data(&t, &contracted->tensor);
		ct_free(contracted->axis_desc);
		contracted->axis_desc = new_axis_desc;

		delete_ttno_contracted_subtree(&child);

		// sort new axes
		// determine corresponding permutation
		struct ttno_indexed_tensor_axis_desc* indexed_axis_desc = ct_malloc(contracted->tensor.ndim * sizeof(struct ttno_indexed_tensor_axis_desc));
		for (int j = 0; j < contracted->tensor.ndim; j++) {
			indexed_axis_desc[j].axis_desc = contracted->axis_desc[j];
			indexed_axis_desc[j].index = j;
		}
		qsort(indexed_axis_desc, contracted->tensor.ndim, sizeof(struct ttno_indexed_tensor_axis_desc), compare_ttno_indexed_tensor_axis_desc);
		int* perm = ct_malloc(contracted->tensor.ndim * sizeof(int));
		for (int j = 0; j < contracted->tensor.ndim; j++) {
			perm[j] = indexed_axis_desc[j].index;
		}
		ct_free(indexed_axis_desc);
		// skip permutation operations in case of an identity permutation
		bool is_identity_perm = true;
		for (int j = 0; j < contracted->tensor.ndim; j++) {
			if (perm[j] != j) {
				is_identity_perm = false;
				break;
			}
		}
		if (!is_identity_perm) {
			transpose_ttno_contracted_subtree(perm, contracted);
		}
		ct_free(perm);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Merge all tensors of a TTNO to obtain the matrix representation on the full Hilbert space.
///
void ttno_to_matrix(const struct ttno* ttno, struct block_sparse_tensor* mat)
{
	assert(ttno->nsites_physical  >= 1);
	assert(ttno->nsites_branching >= 0);

	// overall number of sites
	const int nsites = ttno->nsites_physical + ttno->nsites_branching;

	// select site with maximum number of neighbors as root for contraction
	int i_root = 0;
	for (int l = 1; l < nsites; l++) {
		if (ttno->topology.num_neighbors[l] > ttno->topology.num_neighbors[i_root]) {
			i_root = l;
		}
	}

	// contract full tree
	// set parent index to -1 for root node
	struct ttno_contracted_subtree contracted;
	ttno_contract_subtree(ttno, i_root, -1, &contracted);
	assert(contracted.tensor.ndim == 2 * ttno->nsites_physical);
	for (int l = 0; l < ttno->nsites_physical; l++) {
		// sites must be ordered after subtree contraction
		assert(contracted.axis_desc[2*l    ].index == l && contracted.axis_desc[2*l    ].type == TTNO_TENSOR_AXIS_PHYS_OUT);
		assert(contracted.axis_desc[2*l + 1].index == l && contracted.axis_desc[2*l + 1].type == TTNO_TENSOR_AXIS_PHYS_IN);
	}
	ct_free(contracted.axis_desc);

	// iteratively merge physical axes
	int* perm = ct_malloc(contracted.tensor.ndim * sizeof(int));
	for (int i = 0; i < contracted.tensor.ndim; i++) {
		perm[i] = i;
	}
	if (contracted.tensor.ndim > 2) {
		perm[1] = 2;
		perm[2] = 1;
	}
	while (contracted.tensor.ndim > 2)
	{
		// group input and output axes together
		struct block_sparse_tensor t;
		transpose_block_sparse_tensor(perm, &contracted.tensor, &t);
		delete_block_sparse_tensor(&contracted.tensor);

		// flatten pairs of physical input and output axes
		struct block_sparse_tensor s;
		flatten_block_sparse_tensor_axes(&t, 0, TENSOR_AXIS_OUT, &s);
		delete_block_sparse_tensor(&t);
		flatten_block_sparse_tensor_axes(&s, 1, TENSOR_AXIS_IN, &contracted.tensor);
		delete_block_sparse_tensor(&s);
	}
	assert(contracted.tensor.ndim == 2);
	ct_free(perm);

	move_block_sparse_tensor_data(&contracted.tensor, mat);
}
