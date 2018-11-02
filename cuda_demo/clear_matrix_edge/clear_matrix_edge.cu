/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma diag_suppress 177
// System includes
#include <stdio.h>
#include <cassert>
#include <fstream>


// CUDA runtime
#include <cuda_runtime.h>
//#include <cooperative_groups.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h> 
//#include <helper_cuda_drvapi.h>

#include <iostream>
#include <numeric>
#include <stdlib.h>


// jd add
#define DEL_ARR_MEM(P) if(NULL != (P)){delete [] (P); (P) = NULL;}

#ifdef __linux__
#define __P__  return 0;   //__
#else
#define __P__  system("pause");return 0;   //__
#define popen(fp, fr) _popen((fp),(fr))
#define pclose(fp) _pclose(fp)
#endif
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#ifndef __DEFINE_IDX_I_J_K_i_j_k__
#define _I (blockIdx.y)
#define _J (blockIdx.x)
#define _K (blockIdx.z)

#define _i  (threadIdx.y) 
#define _j (threadIdx.x)
#define _k (threadIdx.z)

#define _COLS (gridDim.x)
#define _ROWS (gridDim.y) 
#define _CHANNELS (gridDim.z)

#define _cols (blockDim.x)
#define _rows (blockDim.y)
#define _channels (blockDim.z)

#define __TID_I_J_K_i_j_k(I,J,K, i, j, k)  ((((I)* _COLS +(J)) * _CHANNELS + (K)) * _cols * _rows * _channels + ((i)* _cols + (j)) * _channels + (k))
#define __TID_J_j(J,j) ((J)* (_cols) + (j))

#define GET_TID  __TID(_I,_J,_K, _i, _j, _k)

#define DEFINE_I_J_K_i_j_k_idx  auto idx = __TID_I_J_K_i_j_k(_I,_J,_K, _i, _j, _k);
#define DEFINE_J_j_idx auto idx = __TID_J_j(_J,_j);

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	system("pause");
	exit(1);
}
#endif 







using namespace std; 
// jd end 

static const int rows = 4; 
static const int cols = 5;



__global__ void testk(int *arr_gpu_a, int sz_bytes)
{
	DEFINE_J_j_idx;


	if(_J == _COLS - 1 || _J == 0 || _j == _cols - 1 || _j == 0)
	{
		arr_gpu_a[idx] = 0; 
	}
	else
	{
		arr_gpu_a[idx] += 1; 
	}

}

// main__ 
int main(int argc, char **argv)
{

	// jd add 
	const int sz_bytes = rows * cols * sizeof(int); 
	int *arr_cpu_a = (int*)malloc(sz_bytes);


	cout << "- before gpu process..." << endl; 
	for(int i=0;i<rows;i++)
	{
		int first_v = 2; 
		for(int j=0;j<cols;j++)
		{
			auto &v = 		arr_cpu_a[i*cols+j];
			v = first_v; 
			printf("%3d ", v);; 
			first_v++; 
		}
		cout << endl; 
	}

	cout << endl; 


	int* arr_gpu_a = NULL;
	CUDA_CHECK_RETURN( cudaMalloc((void **)&arr_gpu_a, sz_bytes) );

	CUDA_CHECK_RETURN( cudaMemcpy(arr_gpu_a, arr_cpu_a, sz_bytes, cudaMemcpyHostToDevice) );

	dim3 grid_(rows, 1, 1);
	dim3 block_(cols, 1, 1);

	testk << <grid_, block_ >> >(arr_gpu_a, sz_bytes);

	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	CUDA_CHECK_RETURN( cudaMemcpy(arr_cpu_a, arr_gpu_a, sz_bytes, cudaMemcpyDeviceToHost) );

	cout << "- after gpu process..." << endl; 
	for(int i=0;i<rows;i++)
	{
		//int first_v = 2; 
		for(int j=0;j<cols;j++)
		{
			auto &v = 		arr_cpu_a[i*cols+j];
			//v = first_v; 
			printf("%3d ", v);; 
			//first_v++; 
		}
		cout << endl; 
	}


	CUDA_CHECK_RETURN( cudaFree(arr_gpu_a) ); 
	free(arr_cpu_a); 


	// jd end 

	__P__;
}
