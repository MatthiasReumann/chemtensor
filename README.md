[![Unit tests](https://github.com/qc-tum/chemtensor/actions/workflows/main.yml/badge.svg)](https://github.com/qc-tum/chemtensor/actions/workflows/main.yml)


ChemTensor
==========

Tensor network algorithms for chemical systems.


Building
--------
The code requires the BLAS, LAPACKE, HDF5 and Python 3 development libraries with NumPy. These can be installed via 
- `sudo apt install libblas-dev liblapacke-dev libhdf5-dev python3-dev python3-numpy` (on Ubuntu Linux)
- `brew install openblas lapack hdf5 python3 numpy` (on MacOS)

From the project directory, use `cmake` to build the project:
```bash
mkdir build && cd build
cmake ../
cmake --build .
```

Currently, this will compile the unit tests, which you can run via `./chemtensor_test`, as well as the demo examples and Python module library.


Coding style conventions
------------------------
- Generally, follow the current coding style of the project.
- Naming: lower_case_with_underscores in general (variable, function, and struct names); exceptionally CAPITALIZATION for preprocessor and enum constants.
- Tabs for indentation at the beginning of a line, otherwise whitespace. This ensures that vertical alignment (of, e.g., comments for struct members) is independent of tab size. Avoid trailing whitespace.
- Comments: // for normal comments, /// for Doxygen documentation.
- Put curly braces `{ }` after every `if` and `else` (to avoid pitfalls).
- Left-align pointers throughout: `int* p` instead of `int *p`.
- Keep the `struct` and `enum` keywords in variable types: `struct foo f;` instead of `typedef struct foo { ... } foo_t; foo_t f;`.
- Use `const` for function arguments which are not modified by the function.


References
----------
- U. Schollwöck  
  The density-matrix renormalization group in the age of matrix product states  
  [Ann. Phys. 326, 96-192 (2011)](https://doi.org/10.1016/j.aop.2010.09.012) ([arXiv:1008.3477](https://arxiv.org/abs/1008.3477))
- J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete  
  Unifying time evolution and optimization with matrix product states  
  [Phys. Rev. B 94, 165116 (2016)](https://doi.org/10.1103/PhysRevB.94.165116) ([arXiv:1408.5056](https://arxiv.org/abs/1408.5056))
- C. Krumnow, L. Veis, Ö. Legeza, J. Eisert  
  Fermionic orbital optimization in tensor network states  
  [Phys. Rev. Lett. 117, 210402 (2016)](https://doi.org/10.1103/PhysRevLett.117.210402) ([arXiv:1504.00042](https://arxiv.org/abs/1504.00042))
- G. K.-L. Chan, A. Keselman, N. Nakatani, Z. Li, S. R. White  
  Matrix product operators, matrix product states, and ab initio density matrix renormalization group algorithms  
  [J. Chem. Phys. 145, 014102 (2016)](https://doi.org/10.1063/1.4955108) ([arXiv:1605.02611](https://arxiv.org/abs/1605.02611))
- J. Ren, W. Li, T. Jiang, Z. Shuai  
  A general automatic method for optimal construction of matrix product operators using bipartite graph theory  
  [J. Chem. Phys. 153, 084118 (2020)](https://doi.org/10.1063/5.0018149) ([arXiv:2006.02056](https://arxiv.org/abs/2006.02056))
