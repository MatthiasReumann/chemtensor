#pragma once
#include "mps.h"
#include "dense_tensor.h"

#include "gmap.h"

void construct_thc_mpo_edge(const int oid, const int cid, const int vids[2], struct mpo_graph_edge* edge);

void construct_thc_mpo_assembly(const int nsites, const double *chi_row, const bool is_creation, struct mpo_assembly *assembly);

void construct_thc_mpo_assembly_4d(const int nsites, const double *chi_row, struct mpo_assembly *assembly);

void interleave_zero(const double *a, const long n, const long offset, double **ret);

void construct_gmap(const struct dense_tensor chi, const long N, const long L, struct gmap *g);

void contract_layer(const struct mps *psi, const struct mpo *mpo, const double tol, const long max_vdim, struct mps *ret);

void add_partial(const struct mps *phi, const struct mps *psi, const double tol, const long max_vdim, struct mps *ret);

void copy_mps(const struct mps *mps, struct mps* ret); // TODO: Move to mps.h

void compute_phi(const struct mps *psi, const struct gmap *gmap, const struct dense_tensor zeta, const long N, const double tol, const long max_vdim, struct mps *phi);