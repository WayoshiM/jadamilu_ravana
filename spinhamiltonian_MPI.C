#include "mpi.h"

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath> 
#include <numeric>
#include <vector>
#include <complex> 
#include <string>
#include <ctime>  
#include <algorithm>
#include <limits>

extern "C" {
	// -------- LAPACK routines 
	// diagonalize symmetric real 
	void dsyev_(const char* choose, const char* uplow, long int* N, double* A, long int* rownum, double* Evals, double* WORK, long int* LWORK, long int* INFO );
	// diagonalize Hermitian 
	void zheev_(const char* choose, const char* uplow, long int* N, double* A, long int* rownum, double* Evals, double* WORK, long int* LWORK, double* RWORK, long int* INFO ); 
	// -------- BLAS routines 
	// matrix operating on a vector 
	void zgemv_(const char*, long int*, long int*, double*, double*, long int*, double*, long int*, double*, double*, long int*); 
	// matrix x matrix
	void zgemm_(const char*, const char*, long int*, long int*, long int*, double*, double*, long int*, double*, long int*, double*, double*, long int*); 
	
	// FORTRAN routines have to be prototyped as extern, and parameters are
	// passed by reference.  Note also that for g77 the function name in C by
	// default is suffixed by a "_".

	void dpjd_(int*, double[], int[], int[], double[], double[], double[],
					int*, int*, double*, int*, int*, int*, int*, double*, double*,
					double*, double*, int[], int*, int*, double*);
}

long int diagonalize(long int Dim, std::complex<double>* Mat, double* Evals);
long int diagonalize_real(long int Dim, double* Mat, double* Evals); 
long int tensor_prod_real(long int DimA, double* A, long int DimB, double* B, long int& DimC, double* C);
long int tprod_addto_real(long int DimA, double* A, long int DimB, double* B, double* C);
long int tensor_prod(long int DimA, std::complex<double>* A, 
					 long int DimB, std::complex<double>* B, long int& DimC, std::complex<double>* C);  
long int A_on_v(    long int N, std::complex<double>* v, std::complex<double>* A); 
long int AT_on_v(   long int N, std::complex<double>* v, std::complex<double>* A); 
long int A_times_B( long int N, std::complex<double>* C, std::complex<double>* A, std::complex<double>* B); 
long int AT_times_B(long int N, std::complex<double>* C, std::complex<double>* A, std::complex<double>* B); 
long int A_times_BT(long int N, std::complex<double>* C, std::complex<double>* A, std::complex<double>* B); 
long int trace_bath(long int Ntot, long int Nsys, std::complex<double>* psi, std::complex<double>* rho); 

long int lint_pow(long int a, long int b); 
long int nchoosek(long int n, long int k); 

double make_gauss(); 

int main(int argc, char* argv[]){

//-------------------------------- initialize MPI

	int my_rank, nproc;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&nproc);
	
	long int spins_min = 7, 
	         spins_max = 7;
	
	double sigma = -2000.0;
	
	if(argc >= 2) {
		spins_min = spins_max = atol(argv[1]);
	}
	if(argc == 3) {
		double min, max;
		
		switch(spins_min) {
			case 15:
				min =  -1.8367546005922E+01;
				max =   1.8316956425920E+01;
				break;
			case 17:
				min =  -2.0973497723021E+01;
				max =   2.1514189481217E+01;
				break;
			case 19:
				min =  -2.3622218647487E+01;
				max =   2.4715579179812E+01;
				break;
			case 21:
				min =  -2.6190122639923E+01;
				max =   2.7918472559467E+01;
				break;
			case 23:
				min =  -2.8853400002761E+01;
				max =   3.1121915407737E+01;
				break;
		}
		
		sigma = (2*min + max)/3;	// 1/3 between min & max e-values
	}
	
	int neig = 10, maxeig = 10, isearch = 2;
	double *eigs = new double[maxeig];
	double *x, *x2;	//the eigenvectors for typicality calculation
	
	const long int sam = 1;    // how many times to repeat each set (for averaging results)
	
	const long int num_ints = 2;   // number of two-body interaction operators
								   // NB: changing this requires other modifications
	
//  ----- cases ----- 
//  fractal		1
//  random			2 
//  chain nn		3
//  ring nn      4
//  all 2 all    5
	
	long int mat_type = 3; 	//should not change in this version of code
	bool display = false; 
	bool rand_int = false; 
	
//---------------------------------------initialize parameters

	const long int seed = 401694;        // 1:1123456, 2:620463, set1: 401694 random seed
	const double pi = 3.1415927;         // fundamental constants
	const std::complex<double> zi(0,1);
	const long int long2 = 2; 
	
	long int Nsys = 2; 
	long int Sn = Nsys*Nsys; 
					
//--------------------------------initialize random numbers 
	std::srand(seed);
	
//--------------------------------initialize timing 
	time_t time0, *time_ptr = 0;       
	time0 = time(time_ptr); 	
	
//------------------ create the Hamiltonian 

	// pointers to matrices and vectors 
	double *UH; 
		
	// loop over different spin numbers 
	for (long int ns=spins_min;ns<=spins_max;ns+=2) {
	// average over many samples 
	for (long int i=0;i<sam;i++) {
		
		double I2[Sn];  
		I2[0] = 1;  I2[2] = 0;
		I2[1] = 0;  I2[3] = 1; 		
		
	
	// 1 qubit interaction operator 
		double Hsys[num_ints*Sn];
		
		if (rand_int) { 
			
			std::complex<double> Hsysc[Sn], U2[Sn]; 

			double xi = static_cast<double>(rand())/static_cast<double>(RAND_MAX);
			double phi = std::asin(std::sqrt(xi)); 
			U2[0] = std::cos(phi);  U2[2] = -std::sin(phi);
			U2[1] = std::sin(phi);  U2[3] = std::cos(phi); 
		
			Hsysc[0] = 1; Hsysc[2] = 0; 
			Hsysc[1] = 0; Hsysc[3] = -1; 
			
			// Hsys = U2' Hsys U2
			std::complex<double> Temp[Sn]; 
			AT_times_B(Nsys,Temp,U2,Hsysc); 
			A_times_B(Nsys,Hsysc,Temp,U2); 
			//real interaction H 
			for (long int i=0;i<Sn;i++) { Hsys[i] = Hsysc[i].real(); };
			
		} else {
			
			Hsys[0] = -0.3827; Hsys[2] = 0.9239; 
			Hsys[1] =  0.9239; Hsys[3] = 0.3827; 
			
			Hsys[4] =  0.8090; Hsys[6] =  0.5878; 
			Hsys[5] =  0.5878; Hsys[7] = -0.8090; 
			
		}
		
		// 2 qubit interaction Hamiltonian 
		double A4[Sn*Sn];  
		tensor_prod_real(Nsys, Hsys, Nsys, Hsys, Sn, A4);  
		
//		for (long int i=0; i<Sn; i++) {
//			for (long int j=0; j<Sn; j++) {
//				std::cout << A4[i + j*Sn] << " ";  
//			} 
//			std::cout << std::endl; 
//		}
//		std::cout << std::endl;
		
		// ----------------- universe dimensions 
		long int Mat_dim = lint_pow(long2,ns); 
		long int Mat_size = Mat_dim*Mat_dim;
		long int up = static_cast<long int>(std::floor(ns/2.0)); 
		long int perm_num = nchoosek(ns,up);  
		long int slice_dim = perm_num; 
		long int slice_size = slice_dim*slice_dim;
        

//		UH = new double[Mat_size];
//		for (long int i=0; i<Mat_size; i++) { UH[i] = 0; };
//		UH = new double[slice_size]; 
//		for (long int i=0; i<slice_size; i++) { UH[i] = 0; };  		
		
		// ----------------- determine degenerate slice 
		long int* perm_vals = new long int[perm_num]; 
		
		long int masks[ns]; 
		for (long int i=0;i<ns;i++) { masks[i] = lint_pow(long2,i); }
				
		// find all int's that have "up" spins up 
		long int ndx = 0; 
		for (long int j=0;j<Mat_dim;j++) {
			long int count = 0; 
			for (long int i=0;i<ns;i++) { if (j & masks[i]) { count++; } }
			if (count==up) { perm_vals[ndx] = j; ndx++; }
		}
		
		// vars for switch case 2 
		long int slice_i, slice_j; 
		
		switch (mat_type) {
																
			case 1:
								
				for (long int n=0; n<spins_max-1; n++) {
					long int repeat_elems = lint_pow(long2,n); 
					long int copies = Mat_dim/(Sn*repeat_elems); 
					
					slice_i = 0; 
					
					for (long int i3=0; i3<copies; i3++) {
						for (long int i2=0; i2<Sn; i2++) {
							for (long int i1=0; i1<repeat_elems; i1++) {
							
								if (i1 + i2*repeat_elems + i3*Sn*repeat_elems == perm_vals[slice_i]) { 
									
									slice_j = 0; 
									
									for (long int j3=0; j3<copies; j3++) {
										for (long int j2=0; j2<Sn; j2++) {
											for (long int j1=0; j1<repeat_elems; j1++) {
												
												if (j1 + j2*repeat_elems + j3*Sn*repeat_elems == perm_vals[slice_j]) { 
																									
													UH[slice_i + slice_j*slice_dim] += A4[i2 + j2*Sn];
										//			UH[(i1 + i2*repeat_elems + i3*Sn*repeat_elems) + (j1 + j2*repeat_elems + j3*Sn*repeat_elems)*Mat_dim] += A4[i2 + j2*Sn];
												
													slice_j++; 
													if (slice_j==slice_dim) {slice_j = 0;}

												}
												
											}
										}
									}
									
									slice_i++;
									if (slice_i==slice_dim) {slice_i = 0;}
									
								}
								
							}
						}
					}
					
				}		
				break;
			
			case 2:

				for (long int is=0;is<slice_dim;is++) { 
					for (long int js=is;js<slice_dim;js++) { 
						double r = make_gauss(); 
						UH[is + js*slice_dim] = r; 
						UH[js + is*slice_dim] = r; 
					}
				} 
			
				break;
				
			case 3:
                // a {} block is created due to scope errors otherwise
                {
                    // nearest neighbor chain
                    int ja_a_counter = 0;
                    
                    // making ia, ja and a
                    // NOTE: THE ARRAY INDEX START OF 0/1 BETWEEN C/C++ AND FORTRAN
                    int my_rows = std::div(slice_dim,static_cast<long int>(nproc-1)).quot;
                    int my_istart = my_rows*my_rank;
                    int my_iend = my_istart + my_rows;
                    if (my_rank == nproc-1) {
                        my_iend = slice_dim;
                    }
                    
                    
                    int* my_ia = new int[my_rows];
                    std::vector<int> my_ja_vector(0);
                    std::vector<double> my_a_vector(0);
                    
                    for (long int is=my_istart;is<my_iend;is++) {
                        
                        // find the permutation for the row index  
                        long int i = perm_vals[is]; 
                        long int perm_i[ns]; 
                        for (long int k=0;k<ns;k++) { 
                          perm_i[k] = i & masks[k] ? 1 : 0;
                        }
                        
    //					std::cout << "i:" << i << "  ";
    //					for (long int k=0;k<ns;k++) { std::cout << perm_i[k] << "  "; }
    //					std::cout << std::endl; 
                        
                        for (long int js=is;js<slice_dim;js++) { 
                            
                            // find the permutation for the col index
                            long int j = perm_vals[js]; 
                            long int perm_j[ns]; 
                            for (long int k=0;k<ns;k++) { 
                              perm_j[k] = j & masks[k] ? 1 : 0;
                            }
                            
    //						std::cout << "j:" << j << "  ";
    //						for (long int k=0;k<ns;k++) { std::cout << perm_j[k] << "  "; }
    //						std::cout << std::endl; 		
                            
                            // initialize the elements  
                            double value = 0.0;
                            
                            // now select, and loop over, the spins that are interacting  
                            for (long int spin1=0;spin1<ns;spin1++) { 
                                for (long int spin2=spin1;spin2<ns;spin2++) { 
                                    
                                    // condition defining the configuration 
                                    if (spin2 == spin1+1 || spin2 == spin1+2) {  // nn chain and next nn chain 
                                        
                                        // calculate the matrix element 
                                        double prod1 = 1, prod2 = 1; 
                                        long int s = 0; 
                                        bool nonzero = true; 
                                        while (nonzero && s<ns) {
                                            
                                            long int id = perm_i[s];
                                            long int jd = perm_j[s];
                                            
                                            if (s!=spin1 && s!=spin2 && id!=jd) { 
                                                nonzero = false; 
                                                prod2 = prod1 = 0; 
                                            }
                                            
                                            if (s==spin1 || s==spin2) { 
                                                prod1 *= Hsys[jd + id*Nsys]; 
                                                prod2 *= Hsys[jd + id*Nsys + 4];
                                            }
                                            else { 
                                                prod1 *= I2[jd + id*Nsys]; 
                                                prod2 *= I2[jd + id*Nsys];
                                            }
                                            
                                            s++; 
                                            
                                        }
                                                                        
    //									std::cout << prod1 << std::endl << std::endl; 
                                        
                                        value += prod1 + prod2;

                                    }

                                }
                            }
                            
                            // here is <-> row, js <-> column
                            if(is == js || value) {
                                if(is == js)
                                    my_ia[is-my_istart] = ja_a_counter+1;
                                my_ja_vector.push_back(js+1);
                                my_a_vector.push_back(value);
                                ja_a_counter++;
                            }
                            
                        }
                    }
                    
                   /* for (int jm=0; jm<nproc; jm++) {
                        if (my_rank==jm) {
                            std::cout << my_rank << ": my_ja_len = " << my_ja_vector.size() << " : a_counter = " << ja_a_counter << std::endl << std::flush;
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    for (int jm=0; jm<nproc-1; jm++) {
                        if (my_rank==jm) {
                            std::cout << my_rank << ": my_rows = " << my_rows << ": " << ": my_iend = " << my_iend << ": " <<my_ia[0] << " " << my_ia[1] << " " << my_ia[2] << std::endl<< std::flush;
                        } else {
                            std::cout << my_rank << ": my_rows = " << my_rows << ": my_iend = " << my_iend << ": " << my_ia[0] << std::endl << std::endl << std::flush;
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
					*/
					
                    // --------------------------------------------------------------------------
                    // we now must combine each set of my_vectors into a single vector for proc 0
                    // --------------------------------------------------------------------------
                    // get all the lengths of the "my_a" vectors to adjust the my_ia values
                    int len_my_a = my_a_vector.size(); 
                    int a_tots[nproc], a_cum[nproc];
                    // share all the lengths of the "my_a" vectors (every proc gets the whole set)
                    MPI_Allgather(&(len_my_a),1,MPI_INT,a_tots,1,MPI_INT,MPI_COMM_WORLD);
                    // cumulative sum the a totals
                    for (int j=0;j<nproc;j++) {a_cum[j] = a_tots[j];}
                    for (int j=1;j<nproc;j++) {a_cum[j] += a_cum[j-1];}
                    // add the a totals to my_ia
                    if (my_rank > 0) { for (int j=0;j<my_rows;j++) {my_ia[j] += a_cum[my_rank-1];} }
                    
                   /* for (int jm=0; jm<nproc; jm++) {
                        if (my_rank==jm) {
                            std::cout << my_rank << ": my_cum = " << a_cum[0] << " " << a_cum[1] << " " << a_cum[2] << " " << a_cum[3] << " " << std::endl<<std::flush;
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    for (int jm=0; jm<nproc-1; jm++) {
                        if (my_rank==jm) {
                            std::cout << my_rank << ": my_rows = " << my_rows << ": " << ": my_iend = " << my_iend << ": " <<my_ia[0] << " " << my_ia[1] << " " << my_ia[2] << std::endl; } else { std::cout << my_rank << ": my_rows = " << my_rows << ": my_iend = " << my_iend << ": " << my_ia[0] << std::endl << std::flush;
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    */
                    // now make the arrays that will store the complete sparse matrix
                    int *ia, *ja;
                    double *a;
                    if (my_rank==0) {
                        ia = new int[my_rows*nproc+1];
                        ja = new int[a_cum[nproc-1]];
                        a =  new double[a_cum[nproc-1]];
                    }
                    
                    // collect together the full ia array (can use Gather as the my_ia are all the same length)
                    MPI_Gather(my_ia,my_rows,MPI_INT,ia,my_rows,MPI_INT,0,MPI_COMM_WORLD);
                    
                    if (my_rank==0) {
						ia[slice_dim] = a_cum[nproc-1];
                       /* std::cout << my_rank << ": full ia = ";
						  for(int i = 0; i < slice_dim+1; i++) printf("%d ", ia[i]);
						std::cout << std::endl << std::flush;*/
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
//                    MPI_Finalize();
//                    std::exit(0);


                    // now collect together all the ja and a vectors
                    // to do this we have to use a bunch of send/recieve calls because they are differen lengths
                    if (my_rank==0) {
                        for (int j=0;j<a_tots[0];j++) {
                            a[j] = my_a_vector[j];
                            ja[j] = my_ja_vector[j];
                        }
                        for (int rank=1;rank<nproc;rank++) {
                            int mesg_label = rank;
                            MPI_Recv(&(ja[a_cum[rank-1]]),a_tots[rank],MPI_INT,   rank,mesg_label,  MPI_COMM_WORLD,&status);
                            MPI_Recv(&(a[a_cum[rank-1]]), a_tots[rank],MPI_DOUBLE,rank,mesg_label+1,MPI_COMM_WORLD,&status);
                        }
                    } else {
                        int mesg_label = my_rank;
                        MPI_Send(&(my_ja_vector[0]),a_tots[my_rank],MPI_INT,0,mesg_label,     MPI_COMM_WORLD);
                        MPI_Send(&(my_a_vector[0]), a_tots[my_rank],MPI_DOUBLE,0,mesg_label+1,MPI_COMM_WORLD);
                    }
                    delete[] my_ia; 
                    
                    // now only proc 0 does the rest of the calculations
                    if (my_rank==0) {

                        //	create other parameters for DPJD
                        printf("Creating remaining parameters for DPJD...\n\n");
                        int n = slice_dim, maxsp = 20;
                        int lx = n*(3*maxsp+maxeig+1) + 3*maxsp*maxsp + std::max(maxsp, neig);

                        double* res = new double[maxeig];
						x = new double[lx];
						x2 = new double[lx];
						
                        //more parameters, see DPJD.f for details rest are at the top of this file since they may be used at the end (particularly x)

                        int iprint = 6;
                        double shift = -7.0, mem = 2000.0, droptol = 0.001;
                        int ninit = 0, madspace = maxsp, iter = std::numeric_limits<int>::max();
                        double tol = 1.0e-10;
                        int* icntl = new int[5];
                        for(int i = 0; i < 5; i++) icntl[i] = 0;

                        time_t time1 = time(time_ptr);

                        // the outputs
                        int info;
                        double gap;

                        // plug it all in
                        std::cout << "Diagonalizing matrix in DPJD (Fortran)..." << std::endl;

                        dpjd_(&n, a, ja, ia, eigs, res, x, &lx, &neig, &sigma, &isearch, &ninit,
                         &madspace, &iter, &tol, &shift, &droptol, &mem, icntl, &iprint,
                         &info, &gap);

                        time_t time2 = time(time_ptr);

                        if(argc != 3) {
							sigma = -sigma;	//now look for largest eigenvalues
							dpjd_(&n, a, ja, ia, eigs, res, x2, &lx, &neig, &sigma, &isearch, &ninit,
							 &madspace, &iter, &tol, &shift, &droptol, &mem, icntl, &iprint,
							 &info, &gap);
						}
						
                        double dur1 = difftime(time2,time1), dur2 = difftime(time1, time0);                                                                                         
                        std::cout << "Setting up " << slice_dim << " took " << dur2 <<" s (" << dur2/60 << " mins) (" << dur2/3600 << " hours)" << std::endl;             
                                      std::cout << "Just diagonalizing " << slice_dim << " took " << dur1 <<" s (" << dur1/60 << " mins)" << std::endl << std::endl; 

                        printf("Back in C++, neig: %d, iter: %d\n\n", neig, iter);
                        
                        delete[] a;
                        delete[] ia;
                        delete[] ja;
                    }
                }
				break;
				
			case 4:
				
				for (long int is=0;is<slice_dim;is++) { 
					
					// find the permutation for the row index  
					long int i = perm_vals[is]; 
					long int perm_i[ns]; 
					for (long int k=0;k<ns;k++) { 
						if (i & masks[k]) { perm_i[k] = 1; } else { perm_i[k] = 0; }
					}
					
					//					std::cout << "i:" << i << "  ";
					//					for (long int k=0;k<ns;k++) { std::cout << perm_i[k] << "  "; }
					//					std::cout << std::endl; 
					
					for (long int js=is;js<slice_dim;js++) { 
						
						// find the permutation for the col index
						long int j = perm_vals[js]; 
						long int perm_j[ns]; 
						for (long int k=0;k<ns;k++) { 
							if (j & masks[k]) { perm_j[k] = 1; } else { perm_j[k] = 0; }
						}
						
						//						std::cout << "j:" << j << "  ";
						//						for (long int k=0;k<ns;k++) { std::cout << perm_j[k] << "  "; }
						//						std::cout << std::endl; 		
						
						// initialize the elements  
						UH[js + is*slice_dim] = 0; 
						UH[is + js*slice_dim] = 0;
						
						// now select, and loop over, the spins that are interacting  
						for (long int spin1=0;spin1<ns;spin1++) { 
							for (long int spin2=spin1;spin2<ns;spin2++) { 
								
								// condition defining the configuration 
								if (spin2 == spin1+1 || (spin1 == 0 && spin2 == ns-1)) {  // nn ring 
									
									// calculate the matrix element 
									double prod1 = 1, prod2 = 1; 
									long int s = 0; 
									bool nonzero = true; 
									while (nonzero && s<ns) {
										
										long int id = perm_i[s];
										long int jd = perm_j[s];
										
										if (s!=spin1 && s!=spin2 && id!=jd) { 
											nonzero = false; 
											prod2 = prod1 = 0; 
										}
										
										if (s==spin1 || s==spin2) { 
											prod1 *= Hsys[jd + id*Nsys]; 
											prod2 *= Hsys[jd + id*Nsys + 4];
										}
										else { 
											prod1 *= I2[jd + id*Nsys]; 
											prod2 *= I2[jd + id*Nsys];
										}
										
										s++; 
										
									}
									
									//									std::cout << prod1 << std::endl << std::endl; 
									
									UH[js + is*slice_dim] += prod1 + prod2; 
									UH[is + js*slice_dim] += prod1 + prod2;  
									
								}
								
							}
						}
					}
				}
				
				//				
				break;				
				
			case 5:
				
				for (long int is=0;is<slice_dim;is++) { 
					
					// find the permutation for the row index  
					long int i = perm_vals[is]; 
					long int perm_i[ns]; 
					for (long int k=0;k<ns;k++) { 
						if (i & masks[k]) { perm_i[k] = 1; } else { perm_i[k] = 0; }
					}
										
					for (long int js=is;js<slice_dim;js++) { 
						
						// find the permutation for the col index
						long int j = perm_vals[js]; 
						long int perm_j[ns]; 
						for (long int k=0;k<ns;k++) { 
							if (j & masks[k]) { perm_j[k] = 1; } else { perm_j[k] = 0; }
						}
						
						// initialize the elements 
						UH[js + is*slice_dim] = 0; 
						UH[is + js*slice_dim] = 0;
						
						// now select, and loop over, the spins that are interacting  
						for (long int spin1=0;spin1<ns;spin1++) { 
							for (long int spin2=spin1;spin2<ns;spin2++) { 
																		
								// calculate the matrix element 
								double prod = 1; 
								long int s = 0; 
								bool nonzero = true; 
								while (nonzero && s<ns) {
									
									long int id = perm_i[s];
									long int jd = perm_j[s];
									
									if (s!=spin1 && s!=spin2 && id!=jd) { 
										nonzero = false; 
										prod = 0; 
									}
									
									if (s==spin1 || s==spin2) { prod *= Hsys[jd + id*Nsys]; }
									else { prod *= I2[jd + id*Nsys]; }
									
									s++; 
									
								}
								
								UH[js + is*slice_dim] += prod; 
								if (js>is) { UH[is + js*slice_dim] += prod; }

							}
						}
					}
				}
				
				break; 
				
			default: 
				std::cout << "Error: no method selected" << std::endl; 
		}
		
		if(my_rank != 0) {
			// tidy up  
			MPI_Finalize();
			return 0;
		}
		
		//---------- data filename and max universe size 
		std::ostringstream dat_filename;
		dat_filename << "eig_typicality_" << spins_min << "_" << (argc == 3 ? "inner" : "edges") << ".log";
		
		std::ofstream output_dat(dat_filename.str().c_str());
			
	//------------------ display matrix 
		
		if (display) { 
			for (long int j=0; j<slice_dim; j++) {
				for (long int i=0; i<slice_dim; i++) {
					std::cout << UH[j + i*slice_dim] << " ";  
				} 
				std::cout << std::endl; 
			}
		}
		
//		std::cout << std::endl; 
//		for (long int j=0; j<Mat_dim; j++) {
//			for (long int i=0; i<Mat_dim; i++) {
//				std::cout << UH[j + i*Mat_dim] << " ";  
//			} 
//			std::cout << std::endl; 
//		}		
			
	//------------------ diagonalize H 
							
		double* Evs = new double[slice_dim]; 
		/*
		std::cout << "Diagonalizing matrix in C++..." << std::endl;
		time_t time1 = time(time_ptr);
		diagonalize_real(slice_dim, UH, Evs); 
		double dur = difftime(time(time_ptr),time1); 
		std::cout << "diagonalizing " << slice_dim << " took " << dur <<" s, or " << dur/60 << " mins" << std::endl << std::endl;  
				
		std::cout << "Evs: ";
		for (long int i=0; i<slice_dim; i++) { std::cout << Evs[i] << " "; } 
		std::cout << std::endl << std::endl; 
		*/
//		// display Matrix 
//		for (long int i=0; i<slice_dim; i++) {
//			for (long int j=0; j<slice_dim; j++) {
//				std::cout << UH[i + j*slice_dim] << " ";  
//			} 
//			std::cout << std::endl; 
//		}
			
		
		//------------------------------------------------------------ 
		//       Calculate the measure of eigenvector typicality  
		//------------------------------------------------------------ 	
		
		double targ = (nchoosek(ns-1,up-1)*1.0)/(nchoosek(ns-1,up)*1.0 + nchoosek(ns-1,up-1)); 
		std::cout << "ns = " << ns << ", targ = " << targ << std::endl;  
		
		/* DPJD variable x contains the eigenvectors, neig the number of eigenvectors/values
			in particular: "On output, approximate eigenvector number i is           
            (or should be) stored in X(1+N*(i-1):N*i), for i=1,...,NEIG"
			where N is of course slice_dim
		*/
		
		for(int l = 0; l < (argc == 3 ? 1 : 2); l++) {
			double* thisX = l == 0 ? x : x2;
			for (long int js=0;js<neig;js++) {   // each eigenvector
				double delta = 0;
				
				for (long int q=0;q<ns;q++) {  // different qbits 
					double sum = 0;  
					for (long int is=0;is<slice_dim;is++) {	//each dimension of this particular eigenvector

						long int ip = perm_vals[is];
						if (ip & masks[q]) { double d = thisX[js*slice_dim + is]; sum += d*d; }
						
					}
					delta += (sum-targ)*(sum-targ); 
				}
				
				delta /= ns;
				std::cout << "delta (for " << (l == 0 ? " highest " : " lowest ") << ") for eigenvector " << js << ": " << delta << std::endl;
				output_dat << slice_dim << " " << delta << std::endl;
			}
		}
		
		//--------------------------------------------------------- 
		//          Calculate the "degeneracy fraction"   
		//---------------------------------------------------------
		/*
		std::vector<double> Evs_vec(Evs,Evs+slice_dim); 
		std::sort(Evs_vec.begin(),Evs_vec.end());
		// normalize E-range 
		double Evs_max; 
		if (std::fabs(Evs_vec[0])<Evs_vec[slice_dim]) { Evs_max = Evs_vec[slice_dim]; }
		else { Evs_max = std::fabs(Evs_vec[0]); }
		for (long int i=0;i<slice_dim;i++) { Evs_vec[i] /= Evs_max; }
		std::vector<double> Egaps(slice_dim-1);
		for (long int i=0;i<slice_dim-1;i++) { Egaps[i] = Evs_vec[i+1] - Evs_vec[i]; } 
		std::sort(Egaps.begin(),Egaps.end());
		// cumulative sum 
		std::vector<double> cumgaps(slice_dim-1);
		std::partial_sum(Egaps.begin(),Egaps.end(),cumgaps.begin()); 
//		std::cout << std::endl;
//		for (long int i=0;i<slice_dim-1;i++) { std::cout << cumgaps[i] << " "; }
//		std::cout << std::endl;
		double avgap = cumgaps[slice_dim-2]/(slice_dim-1);
		double cut_off = avgap/10; 
		// calc degenerate fraction 
		long int cumi = 0; 
		while (cumgaps[cumi]<cut_off) { cumi++; } 
		double deg_frac = (1.0*cumi)/(slice_dim-1); 
		*/
		

		delete[] Evs;
		delete[] perm_vals; 
		//delete[] UH;
		output_dat.close();
    }
    }

    // tidy up  
    MPI_Finalize();
	
    return 0;
}

// ------------------------------------- 
//         additional routines 
// -------------------------------------

long int lint_pow(long int a, long int b) {
	long int x = 1; 
	for (long int i=1; i<=b; i++) { x *= a; }
	return x; 
}

long int nchoosek(long int n, long int k) {
	double nf(n), kf(k), nkf(n-k), nck(1); 
	while (nf>1.0) {
		nck *= nf/(kf*nkf); 
		if (nf>1.0) {nf -= 1.0;} 
		if (kf>1.0)   {kf -= 1.0;} 
		if (nkf>1.0)  {nkf-= 1.0;} 
	}
	double intpart; 
	double fracpart = std::modf(nck, &intpart);
	long int result = static_cast<long int>(intpart); 
    if (fracpart>0.9) { result++; }
    return result; 
}

long int diagonalize(long int Dim, std::complex<double>* Mat, double* Evals) { 
	
	// The unitary U comes back as Mat
	// If the input matrix is Mat = H, then 
	//        H = U diag(Evals) U^T
	//   and  H U = U diag(eig)     so that the columns of U are the eigenvectors
	
	long int info = 0;
	const char* do_vecs = "V", *up = "U";
	double work_dummy[1];
	double* rwork = new double[Dim*3-2]; 
	double* Mat_ptr = reinterpret_cast<double*>(Mat); 
	double* Ev_ptr = reinterpret_cast<double*>(Evals); 
	
	// query to see what the optimal work size is
	long int lwork = -1; 
	zheev_(do_vecs, up, &Dim, Mat_ptr, &Dim, Ev_ptr, work_dummy, &lwork, rwork, &info);
	
	lwork = static_cast<long int>(work_dummy[0]);
	double* work = new double[2*lwork]; 
	
	// calculate the eigenvalues and eigenvectors of A
	zheev_(do_vecs, up, &Dim, Mat_ptr, &Dim, Ev_ptr, work, &lwork, rwork, &info);
	
	delete[] work; 
	delete[] rwork; 
	
	return info;	
}

long int diagonalize_real(long int Dim, double* Mat, double* Evals) { 
	
	// The unitary U comes back as Mat
	// If the input matrix is Mat = H, then 
	//        H = U diag(Evals) U^T
	//   and  H U = U diag(eig)     so that the columns of U are the eigenvectors
	
	long int info = 0;
	const char* do_vecs = "V", *up = "U";
	double work_dummy[1];
	
	// query to see what the optimal work size is
	long int lwork = -1; 
	dsyev_(do_vecs, up, &Dim, Mat, &Dim, Evals, work_dummy, &lwork, &info);
	
	lwork = static_cast<long int>(work_dummy[0]);
	double* work = new double[lwork]; 
	
	// calculate the eigenvalues and eigenvectors of A
	dsyev_(do_vecs, up, &Dim, Mat, &Dim, Evals, work, &lwork, &info);
	
	delete[] work; 
	
	return info;	
}

long int tensor_prod_real(long int DimA, double* A, long int DimB, double* B, long int& DimC, double* C) { 
	// Note: B is the fast index 
	long int Dimtot = DimA*DimB; 
	DimC = Dimtot; 
	double *ptr, *fin; 
	for (long int k=0;k<DimA;k++) { 
		for (long int n=0;n<DimA;n++) {
			double A_nk = A[n + k*DimA];
			for (long int l=0;l<DimB;l++) { 
				ptr = &C[n*DimB + Dimtot*(l + k*DimB)]; fin = ptr + DimB;
				double* B_ptr = &B[l*DimB]; 
				while (ptr!=fin) { *(ptr++) = A_nk*(*(B_ptr++));  }
				//				for (long int m=0;m<DimB;m++) {
				//						C[m + n*DimB + Dimtot*(l + k*DimB)] += A[n + k*DimA]*B[m + l*DimB]; 
				//				} 	
			}
		}
	} 	
	return 0; 
}

long int tensor_prod(long int DimA, std::complex<double>* A, long int DimB, std::complex<double>* B, long int& DimC, std::complex<double>* C) { 
	// Note: B is the fast index 
	long int Dimtot = DimA*DimB; 
	DimC = Dimtot; 
	std::complex<double> *ptr, *fin; 
	for (long int k=0;k<DimA;k++) { 
		for (long int n=0;n<DimA;n++) {
			std::complex<double> A_nk = A[n + k*DimA];
			for (long int l=0;l<DimB;l++) { 
				ptr = &C[n*DimB + Dimtot*(l + k*DimB)]; fin = ptr + DimB;
				std::complex<double>* B_ptr = &B[l*DimB]; 
				while (ptr!=fin) { *(ptr++) = A_nk*(*(B_ptr++));  }
				//				for (long int m=0;m<DimB;m++) {
				//						C[m + n*DimB + Dimtot*(l + k*DimB)] += A[n + k*DimA]*B[m + l*DimB]; 
				//				} 	
			}
		}
	} 	
	return 0; 
}

long int tprod_addto_real(long int DimA, double* A, long int DimB, double* B, double* C) { 
	// Note: B is the fast index 
	double *ptr, *fin; 
	long int Dimtot = DimA*DimB; 
	for (long int k=0;k<DimA;k++) { 
		for (long int n=0;n<DimA;n++) {
			double A_nk = A[n + k*DimA];
			for (long int l=0;l<DimB;l++) { 
				ptr = &C[n*DimB + Dimtot*(l + k*DimB)]; fin = ptr + DimB;
				double* B_ptr = &B[l*DimB]; 
				while (ptr!=fin) { *(ptr++) += A_nk*(*(B_ptr++));  }   // add_to 
			}
		}
	} 	
	return 0; 
}

long int A_on_v(long int N, std::complex<double>* v, std::complex<double>* A) { 
	
	const char* Mat_op = "N"; 
	std::complex<double> alpha[1] = {1.0}, beta[1] = {0.0}; 
	double* alphav = reinterpret_cast<double*> (alpha); 
	double* betav  = reinterpret_cast<double*> (beta); 
	long int incx[1] = {1}, incy[1] = {1}; 
	std::complex<double> x[N]; 
	// copy v to x 
	std::complex<double> *p_x = x, *fin_x = &x[N-1], *p_v = v; fin_x++; 
	while (p_x!=fin_x) { *(p_x++) = *(p_v++); } 
	
	double* A_ptr = reinterpret_cast<double*> (A);
	double* v_ptr = reinterpret_cast<double*> (v); 
	double* x_ptr = reinterpret_cast<double*> (x);
	
	zgemv_(Mat_op, &N, &N, alphav, A_ptr, &N, x_ptr, incx, betav, v_ptr, incy); 
	return 0; 
}

long int AT_on_v(long int N, std::complex<double>* v, std::complex<double>* A) { 
	
	const char* Mat_op = "C"; 
	std::complex<double> alpha[1] = {1.0}, beta[1] = {0.0}; 
	double* alphav = reinterpret_cast<double*> (alpha); 
	double* betav  = reinterpret_cast<double*> (beta); 
	long int incx[1] = {1}, incy[1] = {1}; 
	std::complex<double> x[N]; 
	// copy v to x 
	std::complex<double> *p_x = x, *fin_x = &x[N-1], *p_v = v; fin_x++; 
	while (p_x!=fin_x) { *(p_x++) = *(p_v++); } 
	
	double* A_ptr = reinterpret_cast<double*> (A);
	double* v_ptr = reinterpret_cast<double*> (v); 
	double* x_ptr = reinterpret_cast<double*> (x);
	
	zgemv_(Mat_op, &N, &N, alphav, A_ptr, &N, x_ptr, incx, betav, v_ptr, incy); 
	return 0; 
}


long int A_times_B(long int N, std::complex<double>* C, std::complex<double>* A, std::complex<double>* B) { 
	
	const char *A_Mat_op = "N", *B_Mat_op = "N"; 
	std::complex<double> alpha[1] = {1.0}, beta[1] = {0.0}; 
	double* alphav = reinterpret_cast<double*> (alpha); 
	double* betav  = reinterpret_cast<double*> (beta); 
	
	double* A_ptr = reinterpret_cast<double*> (A);
	double* B_ptr = reinterpret_cast<double*> (B); 
	double* C_ptr = reinterpret_cast<double*> (C);
	
	zgemm_(A_Mat_op, B_Mat_op, &N, &N, &N, alphav, A_ptr, &N, B_ptr, &N, betav, C_ptr, &N); 
	return 0; 
}

long int A_times_BT(long int N, std::complex<double>* C, std::complex<double>* A, std::complex<double>* B) { 
	
	const char *A_Mat_op = "N", *B_Mat_op = "C"; 
	std::complex<double> alpha[1] = {1.0}, beta[1] = {0.0}; 
	double* alphav = reinterpret_cast<double*> (alpha); 
	double* betav  = reinterpret_cast<double*> (beta); 
	
	double* A_ptr = reinterpret_cast<double*> (A);
	double* B_ptr = reinterpret_cast<double*> (B); 
	double* C_ptr = reinterpret_cast<double*> (C);
	
	zgemm_(A_Mat_op, B_Mat_op, &N, &N, &N, alphav, A_ptr, &N, B_ptr, &N, betav, C_ptr, &N); 
	return 0; 
}

long int AT_times_B(long int N, std::complex<double>* C, std::complex<double>* A, std::complex<double>* B) { 
	
	const char *A_Mat_op = "C", *B_Mat_op = "N"; 
	std::complex<double> alpha[1] = {1.0}, beta[1] = {0.0}; 
	double* alphav = reinterpret_cast<double*> (alpha); 
	double* betav  = reinterpret_cast<double*> (beta); 
	
	double* A_ptr = reinterpret_cast<double*> (A);
	double* B_ptr = reinterpret_cast<double*> (B); 
	double* C_ptr = reinterpret_cast<double*> (C);
	
	zgemm_(A_Mat_op, B_Mat_op, &N, &N, &N, alphav, A_ptr, &N, B_ptr, &N, betav, C_ptr, &N); 
	return 0; 
}

long int trace_bath(long int Ntot, long int Nsys, std::complex<double>* psi, std::complex<double>* rho) { 
	
	long int Nbath = Ntot/Nsys; 
	std::complex<double> sum; 
	for (long int n=0;n<Nsys;n++) { 
		for (long int m=0;m<Nsys;m++) { 
			std::complex<double> *n_ztr = &psi[n*Nbath], *n_zin = &psi[(n+1)*Nbath-1]; n_zin++; 
			std::complex<double> *m_ztr = &psi[m*Nbath]; 
			sum = 0.0; 
			while (n_ztr!=n_zin) { sum += (*(m_ztr++))*std::conj(*(n_ztr++)); }
			rho[m + n*Nsys] = sum;
		}
	}
	
	return 0;  
}

double make_gauss() 
{
	double tn1,tn2,rn;
	rn = 2;
	while (rn >= 1 || rn == 0) {
		tn1 = 2*static_cast<double>(rand())/static_cast<double>(RAND_MAX) - 1;
		tn2 = 2*static_cast<double>(rand())/static_cast<double>(RAND_MAX) - 1;
		rn = tn1*tn1 + tn2*tn2;
	}
	rn = std::sqrt((-2*std::log(rn)/rn));
	return tn1*rn;
}

// Haar 2x2 unitary 
//double alp = 2*pi*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
//double psi = 2*pi*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
//double chi = 2*pi*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
//double xi = static_cast<double>(rand())/static_cast<double>(RAND_MAX);
//double phi = std::arcsin(std::sqrt(xi)); 
//U2[0] = std::exp(zi*(alp+psi))*std::cos(phi); 
//U2[1] = -std::exp(zi*(alp-chi))*std::sin(phi); 
//U2[2] = std::exp(zi*(alp+chi))*std::sin(phi); 
//U2[3] = std::exp(zi*(alp-psi))*std::cos(phi); 




//   // --- first U_Dt <- (UH*diag(D))
//for (long int n=0;n<Ntot;n++) { 
//	std::complex<double> Dn = std::exp(-ei*Evs[n]*Dt); 
//	ptr =     &UH[n*Ntot]; fin = &UH[(n+1)*Ntot - 1]; fin++; 
//	ztr = &U_Dt[n*Ntot]; 
//	while (ptr!=fin) { *(ztr++) = (*(ptr++))*Dn; } 
//}
//  // --- then U_Dt <- U_Dt*UH' 
//  // --- virtually in-place multiply (probably much slower than BLAS)
//std::complex<double>* rtemp = new std::complex<double>[Ntot]; 
//for (long int n=0;n<Ntot;n++) { 		
//	for (long int m=0;m<Ntot;m++) { 
//		rtemp[m] = 0.0; 
//		for (long int k=0;k<Ntot;k++) { rtemp[m] += U_Dt[n + k*Ntot]*UH[m + k*Ntot]; }  // HU flipped 
//	}
//	for (long int m=0;m<Ntot;m++) { U_Dt[n + m*Ntot] = rtemp[m]; }
//}
//
//delete[] rtemp; 


//------------------ create the Evolution operator 

//	if (calc_U_Dt) {
//		time0 = time(time_ptr); 	
//		
//		U_Dt = new std::complex<double>[Mat_size]; 
//		UHD  = new std::complex<double>[Mat_size];
//		UHC  = new std::complex<double>[Mat_size];
//		
//		// calculate UH*diag(D)*UH' 
//		//    create complex version of UH 
//		ztr = UHC; 
//		ptr = UH; fin = &UH[Mat_size - 1]; fin++; 
//		while (ptr!=fin) { *(ztr++) = *(ptr++); }
//		//    calc UHD = UH*diag(D) by multiplying each of the columns of UH with D[m]
//		for (long int n=0;n<Ntot;n++) { 
//			std::complex<double> Dn = std::exp(-ei*Evs[n]*Dt); 
//			std::complex<double> *ztr2 = &UHC[n*Ntot], *zin2 = &UHC[(n+1)*Ntot - 1]; zin2++; 
//			ztr = &UHD[n*Ntot]; 
//			while (ztr2!=zin2) { *(ztr++) = (*(ztr2++))*Dn; } 
//		}
//		//    calc U_Dt = UHD*UH' 
//		A_times_BT(Ntot, U_Dt, UHD, UHC); 
//		
//		delete[] UHD; 
//		delete[] UHC; 
//		delete[] UH; 
//		
//		std::cout << "Creating U_Dt took " 
//		<< difftime(time(time_ptr),time0)/60.0 << " mins" << std::endl; 
//		
//		// --- save U_Dt 
//		std::string Udt_fbase = "Udt_", Udt_fname = Udt_fbase + fxbin; 
//		std::ofstream output_Udt(Udt_fname.c_str(), std::ios::out | std::ios::binary);
//		long int nblocks = 10, bsize = Ntot*(Ntot/nblocks), block = bsize*2*sizeof(double);	
//		ztr = U_Dt; 
//		for (long int j=0;j<nblocks;j++) {
//			output_Udt.write(reinterpret_cast<char*>(ztr), block);
//			ztr = ztr + bsize; 
//		}
//		output_Udt.close(); 
//	}

//	if (!calc_U_Dt) { 
//		std::string Udt_fbase = "Udt_", Udt_fname = Udt_fbase + fxbin; 
//		std::ifstream input_Udt(Udt_fname.c_str(), std::ios::out | std::ios::binary);
//		U_Dt = new std::complex<double>[Mat_size];
//		long int nblocks = 10, bsize = Ntot*(Ntot/nblocks), block = bsize*2*sizeof(double);	
//		ztr = U_Dt; 
//		for (long int j=0;j<nblocks;j++) {
//			input_Udt.read(reinterpret_cast<char*>(ztr), block);
//			ztr = ztr + bsize; 
//		}
//		input_Udt.close(); 		
//	}
