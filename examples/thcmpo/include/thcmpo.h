#pragma once
#include "mps.h"
#include "dense_tensor.h"

void construct_thc_mpo_edge(const int oid, const int cid, const int vids[2], struct mpo_graph_edge *edge);

void construct_thc_mpo_assembly(const int nsites, const double *chi_row, const bool is_creation, struct mpo_assembly *assembly);

void construct_thc_mpo_assembly_4d(const int nsites, const double *chi_row, const bool is_creation, const bool is_spin_up, struct mpo_assembly *assembly);

void interleave_zero(const double *a, const long n, const long offset, double **ret);

long index_to_g_offset(const long N, const long i, const long s);

void construct_g(const struct dense_tensor chi, const long N, const long L, struct mpo **g);

void construct_g_4d(const struct dense_tensor chi, const long N, const long L, struct mpo **g);

void apply_and_compress(const struct mps *psi, const struct mpo *mpo, const double tol, const long max_vdim, struct mps *ret);

void add_and_compress(const struct mps *phi, const struct mps *psi, const double tol, const long max_vdim, struct mps *ret);

void mps_deep_copy(const struct mps *mps, struct mps *ret); // TODO: Move to mps.h

void apply_thc(const struct mps *psi, struct mpo **g, const struct dense_tensor zeta, const long N, const double tol, const long max_vdim, struct mps *phi);

void apply_thc_omp(const struct mps *psi, struct mpo **g, const struct dense_tensor zeta, const long N, const double tol, const long max_vdim, struct mps *phi);