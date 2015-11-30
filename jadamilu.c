// CCode with documentation

#include <stdio.h>
#include "JadamiluTest.h"  // Required header for JNI

// FORTRAN routines have to be prototyped as extern, and parameters are
// passed by reference.  Note also that for g77 the function name in C by
// default is suffixed by a "_".

extern void dpjd_(int*, double[], int[], int[], double[], double[], double[],
		int*, int*, double*, int*, int*, int*, int*, double*, double*,
		double*, double*, int[], int*, int*, double*);

// When calling C code from Java, main() must be replaced by a declaration
// similar to below, where the function name is given by "Java_" + the name
// of the class in the Java code that calls this C code, in this case
// "JadamiluTest", + "_" + the name of this C function called from Java,
// in this case "pjd".

JNIEXPORT jint JNICALL Java_JadamiluTest_pjd(JNIEnv *env,
                       jclass cls) {
  printf("In Test C program");
  
  //parameters based on EXAMPLE3.f

  int n = 1000, maxeig = 5, maxsp = 20;
  int lx = n*(3*maxsp+maxeig+1) + 4*maxsp*maxsp;
  
  double eigs[maxeig];
  double res[maxeig];
  double x[lx];

  // making the matrix - NOTE THE ARRAY INDEX DIFFERENCE
  int ia[n+1];
  int ja[2*n];
  double a[2*n];

  int i, k=1;
  for(i = 1; i <= n; i++) {
    ia[i-1] = k;
    ja[k-1] = i;
    a[k-1] = (double) (i - 1);
    k++;
    if (i < n) {
      ja[k-1] = i+1;
      a[k-1] = 5.0;
      k++;
    }
  }
  ia[n] = k;

  //more parameters

  int iprint = 6, isearch = 1;
  double sigma = -7.0, shift = -7.0, mem = 20.0, droptol = 0.001;
  int neig = maxeig, ninit = 0, madspace = maxsp, iter = 1000;
  double tol = 1.0e-10;
  int icntl[5];
  for(i = 0; i < 5; i++) icntl[i] = 0;
  
  // the outputs
  int info;
  double gap;

  // plug it all in
  
  dpjd_(&n, a, ja, ia, eigs, res, x, &lx, &neig, &sigma, &isearch, &ninit,
       &madspace, &iter, &tol, &shift, &droptol, &mem, icntl, &iprint,
       &info, &gap);
  
  printf("In C, info: %d, gap: %f\n\n", info, gap);
  return info;
}
