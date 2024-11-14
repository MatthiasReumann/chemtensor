#pragma once

#include "mpo.h"
#include "dense_tensor.h"
#include "block_sparse_tensor.h"

void print_dt(struct dense_tensor dt);

void print_mpo(struct mpo mpo);

void print_bst(struct block_sparse_tensor bst);