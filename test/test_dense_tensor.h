#pragma once

#include <math.h>
#include <stdbool.h>
#include "dense_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Test whether two tensors agree elementwise within tolerance 'tol'.
///
static inline bool dense_tensor_allclose(const struct dense_tensor* s, const struct dense_tensor* t, double tol)
{
	// compare data types
	if (s->dtype != t->dtype) {
		return false;
	}

	// compare degrees
	if (s->ndim != t->ndim) {
		return false;
	}

	// compare dimensions
	for (int i = 0; i < s->ndim; i++)
	{
		if (s->dim[i] != t->dim[i]) {
			return false;
		}
	}

	// compare entries
	const long nelem = dense_tensor_num_elements(s);
	if (uniform_distance(s->dtype, nelem, s->data, t->data) > tol) {
		return false;
	}

	return true;
}
