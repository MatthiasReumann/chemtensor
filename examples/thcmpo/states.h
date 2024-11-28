#pragma once
#include "mps.h"
#include "aligned_memory.h"

void construct_computational_basis_mps(const int nsites, const unsigned *basis_state, struct mps *mps);

void construct_spin_basis_mps(const int nsites, const unsigned *spin_state, struct mps *mps);

void construct_spin_zero_mps(const int nsites, const unsigned *spin_state, struct mps *mps);