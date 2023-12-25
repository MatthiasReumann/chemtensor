/// \file util.c
/// \brief Utility functions.

#include <math.h>
#include <memory.h>
#include <complex.h>
#include <assert.h>
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Calculate the product of a list of integer numbers.
///
long integer_product(const long* x, const int n)
{
	assert(n >= 0); // n == 0 is still reasonable

	long prod = 1;
	for (int i = 0; i < n; i++)
	{
		prod *= x[i];
	}

	return prod;
}


//________________________________________________________________________________________________________________________
///
/// \brief Uniform distance (infinity norm) between 'x' and 'y'.
///
/// The result is cast to double format (even if the actual entries are of single precision).
///
double uniform_distance(const enum numeric_type dtype, const long n, const void* restrict x, const void* restrict y)
{
	switch (dtype)
	{
		case SINGLE_REAL:
		{
			const float* xv = x;
			const float* yv = y;
			float d = 0;
			for (long i = 0; i < n; i++)
			{
				d = fmaxf(d, fabsf(xv[i] - yv[i]));
			}
			return d;
		}
		case DOUBLE_REAL:
		{
			const double* xv = x;
			const double* yv = y;
			double d = 0;
			for (long i = 0; i < n; i++)
			{
				d = fmax(d, fabs(xv[i] - yv[i]));
			}
			return d;
		}
		case SINGLE_COMPLEX:
		{
			const scomplex* xv = x;
			const scomplex* yv = y;
			float d = 0;
			for (long i = 0; i < n; i++)
			{
				d = fmaxf(d, cabsf(xv[i] - yv[i]));
			}
			return d;
		}
		case DOUBLE_COMPLEX:
		{
			const dcomplex* xv = x;
			const dcomplex* yv = y;
			double d = 0;
			for (long i = 0; i < n; i++)
			{
				d = fmax(d, cabs(xv[i] - yv[i]));
			}
			return d;
		}
		default:
		{
			// unknown data type
			assert(false);
			return 0;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Read an HDF5 dataset from a file.
///
herr_t read_hdf5_dataset(hid_t file, const char* name, hid_t mem_type, void* data)
{
	hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr, "'H5Dopen' for '%s' failed, return value: %" PRId64 "\n", name, dset);
		return -1;
	}

	herr_t status = H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Dread' failed, return value: %d\n", status);
		return status;
	}

	H5Dclose(dset);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Read an HDF5 attribute from a file.
///
herr_t read_hdf5_attribute(hid_t file, const char* name, hid_t mem_type, void* data)
{
	hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
	if (attr < 0)
	{
		fprintf(stderr, "'H5Aopen' for '%s' failed, return value: %" PRId64 "\n", name, attr);
		return -1;
	}

	herr_t status = H5Aread(attr, mem_type, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Aread' failed, return value: %d\n", status);
		return status;
	}

	H5Aclose(attr);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 dataset to a file.
///
herr_t write_hdf5_dataset(hid_t file, const char* name, int degree, const hsize_t dims[], hid_t mem_type_store, hid_t mem_type_input, const void* data)
{
	// create dataspace
	hid_t space = H5Screate_simple(degree, dims, NULL);
	if (space < 0) {
		fprintf(stderr, "'H5Screate_simple' failed, return value: %" PRId64 "\n", space);
		return -1;
	}
	
	// property list to disable time tracking
	hid_t cplist = H5Pcreate(H5P_DATASET_CREATE);
	herr_t status = H5Pset_obj_track_times(cplist, 0);
	if (status < 0) {
		fprintf(stderr, "creating property list failed, return value: %d\n", status);
		return status;
	}

	// create dataset
	hid_t dset = H5Dcreate(file, name, mem_type_store, space, H5P_DEFAULT, cplist, H5P_DEFAULT);
	if (dset < 0) {
		fprintf(stderr, "'H5Dcreate' failed, return value: %" PRId64 "\n", dset);
		return -1;
	}

	// write the data to the dataset
	status = H5Dwrite(dset, mem_type_input, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if (status < 0) {
		fprintf(stderr, "'H5Dwrite' failed, return value: %d\n", status);
		return status;
	}

	H5Dclose(dset);
	H5Pclose(cplist);
	H5Sclose(space);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 scalar attribute to a file.
///
herr_t write_hdf5_scalar_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const void* data)
{
	// create dataspace
	hid_t space = H5Screate(H5S_SCALAR);
	if (space < 0) {
		fprintf(stderr, "'H5Screate' failed, return value: %" PRId64 "\n", space);
		return -1;
	}

	// create attribute
	hid_t attr = H5Acreate(file, name, mem_type_store, space, H5P_DEFAULT, H5P_DEFAULT);
	if (attr < 0) {
		fprintf(stderr, "'H5Acreate' failed, return value: %" PRId64 "\n", attr);
		return -1;
	}

	herr_t status = H5Awrite(attr, mem_type_input, data);
	if (status < 0) {
		fprintf(stderr, "'H5Awrite' failed, return value: %d\n", status);
		return status;
	}

	H5Aclose(attr);
	H5Sclose(space);

	return 0;
}
