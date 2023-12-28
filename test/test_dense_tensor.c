#include <complex.h>
#include "dense_tensor.h"
#include "test_dense_tensor.h"
#include "config.h"


char* test_dense_tensor_trace()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_trace.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_trace failed";
	}

	struct dense_tensor t;
	const long tdim[3] = { 5, 5, 5 };
	allocate_dense_tensor(DOUBLE_COMPLEX, 3, tdim, &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	dcomplex tr;
	dense_tensor_trace(&t, &tr);

	// reference value for checking
	dcomplex tr_ref;
	if (read_hdf5_dataset(file, "tr", H5T_NATIVE_DOUBLE, &tr_ref) < 0) {
		return "reading trace value from disk failed";
	}

	// compare
	if (cabs(tr - tr_ref) > 1e-13) {
		return "tensor trace does not match reference";
	}

	// clean up
	delete_dense_tensor(&t);

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_transpose()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_transpose.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_transpose failed";
	}

	// create tensor 't'
	struct dense_tensor t;
	const long dim[4] = { 4, 5, 6, 7 };
	allocate_dense_tensor(SINGLE_REAL, 4, dim,  &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_FLOAT, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// generalized transposition
	const int perm[4] = { 1, 3, 2, 0 };
	struct dense_tensor t_tp;
	transpose_dense_tensor(perm, &t, &t_tp);

	// reference tensor
	const long refdim[4] = { 5, 7, 6, 4 };
	struct dense_tensor t_tp_ref;
	allocate_dense_tensor(SINGLE_REAL, 4, refdim, &t_tp_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "t_tp", H5T_NATIVE_FLOAT, t_tp_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&t_tp, &t_tp_ref, 0.)) {
		return "transposed tensor does not match reference";
	}

	// clean up
	delete_dense_tensor(&t_tp_ref);
	delete_dense_tensor(&t_tp);
	delete_dense_tensor(&t);

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_dot()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_dot.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_dot failed";
	}

	// create tensor 't'
	struct dense_tensor t;
	const long dim[4] = { 2, 3, 4, 5 };
	allocate_dense_tensor(DOUBLE_COMPLEX, 4, dim,  &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// create another tensor 's'
	struct dense_tensor s;
	const long sdim[4] = { 4, 5, 7, 6 };
	allocate_dense_tensor(DOUBLE_COMPLEX, 4, sdim, &s);
	// read values from disk
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_DOUBLE, s.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// multiply tensors and store result in 'r'
	struct dense_tensor t_dot_s;
	dense_tensor_dot(&t, &s, 2, &t_dot_s);

	// reference tensor for checking
	const long refdim[4] = { 2, 3, 7, 6 };
	struct dense_tensor t_dot_s_ref;
	allocate_dense_tensor(DOUBLE_COMPLEX, 4, refdim, &t_dot_s_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "t_dot_s", H5T_NATIVE_DOUBLE, t_dot_s_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&t_dot_s, &t_dot_s_ref, 1e-13)) {
		return "dot product of tensors does not match reference";
	}

	// clean up
	delete_dense_tensor(&t_dot_s_ref);
	delete_dense_tensor(&t_dot_s);
	delete_dense_tensor(&s);
	delete_dense_tensor(&t);

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_dot_update()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_dot_update.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_dot_update failed";
	}

	const scomplex alpha =  1.2f - 0.3f*I;
	const scomplex beta  = -0.7f + 0.8f*I;

	// create tensor 't'
	struct dense_tensor t;
	const long dim[4] = { 2, 3, 4, 5 };
	allocate_dense_tensor(SINGLE_COMPLEX, 4, dim,  &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_FLOAT, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// create another tensor 's'
	struct dense_tensor s;
	const long sdim[4] = { 4, 5, 7, 6 };
	allocate_dense_tensor(SINGLE_COMPLEX, 4, sdim, &s);
	// read values from disk
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_FLOAT, s.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// multiply tensors and update 't_dot_s' with result
	struct dense_tensor t_dot_s;
	const long t_dot_s_dim[4] = { 2, 3, 7, 6 };
	allocate_dense_tensor(SINGLE_COMPLEX, 4, t_dot_s_dim, &t_dot_s);
	// read values from disk
	if (read_hdf5_dataset(file, "t_dot_s_0", H5T_NATIVE_FLOAT, t_dot_s.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	dense_tensor_dot_update(&alpha, &t, &s, 2, &t_dot_s, &beta);

	// reference tensor for checking
	const long refdim[4] = { 2, 3, 7, 6 };
	struct dense_tensor t_dot_s_ref;
	allocate_dense_tensor(SINGLE_COMPLEX, 4, refdim, &t_dot_s_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "t_dot_s_1", H5T_NATIVE_FLOAT, t_dot_s_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&t_dot_s, &t_dot_s_ref, 1e-5)) {
		return "tensor updated by dot product of two other tensors does not match reference";
	}

	// clean up
	delete_dense_tensor(&t_dot_s_ref);
	delete_dense_tensor(&t_dot_s);
	delete_dense_tensor(&s);
	delete_dense_tensor(&t);

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_kronecker_product()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_kronecker_product.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_kronecker_product failed";
	}

	// create tensor 's'
	struct dense_tensor s;
	const long sdim[4] = { 6, 5, 7, 2 };
	allocate_dense_tensor(DOUBLE_COMPLEX, 4, sdim, &s);
	// read values from disk
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_DOUBLE, s.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// create tensor 't'
	struct dense_tensor t;
	const long tdim[4] = { 3, 11, 2, 5 };
	allocate_dense_tensor(DOUBLE_COMPLEX, 4, tdim,  &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	struct dense_tensor r;
	dense_tensor_kronecker_product(&s, &t, &r);

	// load reference values from disk
	struct dense_tensor r_ref;
	const long refdim[4] = { 18, 55, 14, 10 };
	allocate_dense_tensor(DOUBLE_COMPLEX, 4, refdim, &r_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "r", H5T_NATIVE_DOUBLE, r_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&r, &r_ref, 1e-13)) {
		return "Kronecker product of tensors does not match reference";
	}

	// clean up
	delete_dense_tensor(&r_ref);
	delete_dense_tensor(&r);
	delete_dense_tensor(&t);
	delete_dense_tensor(&s);

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_kronecker_product_degree_zero()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_kronecker_product_degree_zero.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_kronecker_product_degree_zero failed";
	}

	// create tensor 's'
	struct dense_tensor s;
	allocate_dense_tensor(SINGLE_COMPLEX, 0, NULL, &s);
	// read values from disk
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_FLOAT, s.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// create tensor 't'
	struct dense_tensor t;
	allocate_dense_tensor(SINGLE_COMPLEX, 0, NULL,  &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_FLOAT, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	struct dense_tensor r;
	dense_tensor_kronecker_product(&s, &t, &r);

	// load reference values from disk
	struct dense_tensor r_ref;
	allocate_dense_tensor(SINGLE_COMPLEX, 0, NULL, &r_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "r", H5T_NATIVE_FLOAT, r_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&r, &r_ref, 1e-5)) {
		return "Kronecker product of tensors does not match reference";
	}

	// clean up
	delete_dense_tensor(&r_ref);
	delete_dense_tensor(&r);
	delete_dense_tensor(&t);
	delete_dense_tensor(&s);

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_qr()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_qr.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_qr failed";
	}

	const enum numeric_type dtypes[4] = { SINGLE_REAL, DOUBLE_REAL, SINGLE_COMPLEX, DOUBLE_COMPLEX };

	// cases m >= n and m < n
	for (int i = 0; i < 2; i++)
	{
		// data types
		for (int j = 0; j < 4; j++)
		{
			const double tol = (j % 2 == 0 ? 1e-6 : 1e-13);

			// matrix 'a'
			struct dense_tensor a;
			const long dim[2] = { i == 0 ? 11 : 5, i == 0 ? 7 : 13 };
			allocate_dense_tensor(dtypes[j], 2, dim, &a);
			// read values from disk
			char varname[1024];
			sprintf(varname, "a_s%i_t%i", i, j);
			if (read_hdf5_dataset(file, varname, j % 2 == 0 ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE, a.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			// perform QR decomposition
			struct dense_tensor q, r;
			dense_tensor_qr(&a, &q, &r);

			// matrix product 'q r' must be equal to 'a'
			struct dense_tensor qr;
			dense_tensor_dot(&q, &r, 1, &qr);
			if (!dense_tensor_allclose(&qr, &a, tol)) {
				return "matrix product Q R is not equal to original A matrix";
			}
			delete_dense_tensor(&qr);

			// 'q' must be an isometry
			struct dense_tensor qh;
			const int perm[2] = { 1, 0 };
			conjugate_transpose_dense_tensor(perm, &q, &qh);
			struct dense_tensor qhq;
			dense_tensor_dot(&qh, &q, 1, &qhq);
			struct dense_tensor id;
			const long dim_id[2] = { q.dim[1], q.dim[1] };
			allocate_dense_tensor(dtypes[j], 2, dim_id, &id);
			dense_tensor_set_identity(&id);
			if (!dense_tensor_allclose(&qhq, &id, tol)) {
				return "Q matrix is not an isometry";
			}
			delete_dense_tensor(&id);
			delete_dense_tensor(&qhq);
			delete_dense_tensor(&qh);

			// 'r' must be upper triangular
			const long k = dim[0] <= dim[1] ? dim[0] : dim[1];
			void* zero_vec = aligned_calloc(MEM_DATA_ALIGN, k, sizeof_numeric_type(r.dtype));
			for (long l = 0; l < k; l++) {
				if (uniform_distance(r.dtype, l, (char*)r.data + (l*dim[1])*sizeof_numeric_type(r.dtype), zero_vec) != 0) {
					return "R matrix is not upper triangular";
				}
			}
			aligned_free(zero_vec);

			delete_dense_tensor(&r);
			delete_dense_tensor(&q);
			delete_dense_tensor(&a);
		}
	}

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_block()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_block.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_transpose failed";
	}

	// create tensor 't'
	struct dense_tensor t;
	const long dim[4] = { 2, 3, 4, 5 };
	allocate_dense_tensor(DOUBLE_COMPLEX, 4, dim,  &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	const long bdim[4] = { 1, 2, 4, 3 };

	// indices along each dimension
	const long idx0[1] = { 1 };
	const long idx1[2] = { 0, 2 };
	const long idx2[4] = { 0, 1, 2, 3 };
	const long idx3[3] = { 1, 4, 4 }; // index 4 appears twice
	const long *idx[4] = { idx0, idx1, idx2, idx3 };

	struct dense_tensor b;
	dense_tensor_block(&t, bdim, idx, &b);

	// reference tensor for checking
	struct dense_tensor b_ref;
	allocate_dense_tensor(DOUBLE_COMPLEX, 4, bdim, &b_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "b", H5T_NATIVE_DOUBLE, b_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&b, &b_ref, 1e-15)) {
		return "extracted sub-block does not match reference";
	}

	// clean up
	delete_dense_tensor(&b_ref);
	delete_dense_tensor(&b);
	delete_dense_tensor(&t);

	H5Fclose(file);

	return 0;
}
