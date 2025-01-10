#pragma once
#include "dense_tensor.h"
#include "mps.h"

struct thc_spin_hamiltonian {
	struct mpo T;         // kinetic MPO
	struct mpo* thc_mpos; // list of length `4 * thc_rank` containing G[μ, σ, ν, τ]
	struct dense_tensor* zeta;

	long thc_rank; // THC-Rank N
};

void construct_thc_spin_hamiltonian(const struct dense_tensor* tkin, struct dense_tensor* zeta, const struct dense_tensor* chi, struct thc_spin_hamiltonian* hamiltonian);

void apply_thc_spin_hamiltonian(const struct thc_spin_hamiltonian* hamiltonian, const struct mps* psi, const double tol, const long max_vdim, struct mps* ret);

void apply_thc_spin_coulomb(const struct thc_spin_hamiltonian* hamiltonian, const struct mps* psi, const double tol, const long max_vdim, struct mps* ret);