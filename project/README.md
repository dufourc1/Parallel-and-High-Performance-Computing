PHPC - CONJUGATE GRADIENT PROJECT

HOW TO COMPILE AND RUN
=====================

Requirements : 

- a recent compiler (like gcc or intel)
- a cblas library (like openblas or intel MKL)

compile on SCITAS clusters :

```
$ module load gcc openblas
$ make
```

You should see this output (timing is indicative) :

```
$ srun ./cgsolver lap2D_5pt_n100.mtx 
size of matrix = 10000 x 10000
Call cgsolver() on matrix size (10000 x 10000)
	[STEP 488] residual = 1.103472E-10
Time for CG = 36.269389 [s]
```

The given example is a 5-points stencil for the 2D Laplace problem. The matrix is in sparse format.

The matrix format is [Matrix Market format (.mtx extension)](https://sparse.tamu.edu/). You can use other matrices there or create your own. 

MODIFICATION IN RUN CALL
========================

```
Usage: ./cgsolver [martix-market-filename] | [matrix-number-rows]
```

where matrix-number-rows is the size of the matrix (square matrix) that will be generated as a tridiagonal matrix. Adding this value will run the CGSOLVER on the newly generated matrix and discard the matrix-market-filename. Otherwise, the CGSOLVER will run on the matrix-market-filename.