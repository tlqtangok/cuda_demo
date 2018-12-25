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
#include <thread>


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

static const int rows = 5;
static const int cols = 2;



__global__ void testk_no_extern(int *arr_gpu_a, int sz)
{
	DEFINE_I_J_K_i_j_k_idx;
	//printf("- %d %d %d,  %d %d %d\n",  _J, _I,_K,  _j, _i, _k);

	__shared__ int s_arr[1];
	
	if(idx % cols == 0)
	{
		memset(s_arr, 0, sizeof(int) * 1); 
	}

	__syncthreads();

	if(idx < rows * cols)
	{
		atomicAdd(&s_arr[0],  arr_gpu_a[idx]);
	}
	__syncthreads(); 

	if(idx % cols == 0)
	{
		printf("%d\n", s_arr[0]); 
	}
	__syncthreads();
}

__global__ void testk_extern(int *arr_gpu_a, int sz)
{
	DEFINE_I_J_K_i_j_k_idx;
	//printf("- %d %d %d,  %d %d %d\n",  _J, _I,_K,  _j, _i, _k);


	extern __shared__ int s_arr[];
	if (idx % cols == 0)
	{
		memset(s_arr, 0, sizeof(int) * 1); 
	}

	__syncthreads();

	if(idx < rows * cols)
	{
		atomicAdd(&s_arr[0],  arr_gpu_a[idx]);
	}
	__syncthreads(); 

	if(idx % cols == 0)
	{
		printf("%d\n", s_arr[0]); 
	}
	__syncthreads(); 

}

#if 1 
struct link_node_t
{
		int x; 
			link_node_t *next; 
};
#endif 


__global__ void test_inner_gpu(const int label, int *x)
{
	DEFINE_I_J_K_i_j_k_idx; 

	int t = 0; 
	if (label == 0)
	{

		for(int i=0;i<1e7;i++)
		{
			t+= 22; 	
		}
	
	}

	x[idx] = idx * idx + label; 

	printf("- t: %d , label: %d, idx: %d\n", t , label, idx); 
}

// main__ 
int main(int argc, char **argv)
{

    // jd add 


    //float* arr_gpu_a = NULL;
    //cudaMalloc((void **)&arr_gpu_a, sizeof(float));


    dim3 grid_(rows, 1, 1);
    dim3 block_(cols, 1, 1);
	int *p_t_0 = NULL; 
	int *p_t_1 = NULL; 
	int *p_t_2 = NULL; 

	CUDA_CHECK_RETURN( cudaMallocManaged((void**)&p_t_0, rows * cols * sizeof(int)) ); 
	CUDA_CHECK_RETURN( cudaMallocManaged((void**)&p_t_1, rows * cols * sizeof(int)) ); 
	CUDA_CHECK_RETURN( cudaMallocManaged((void**)&p_t_2, rows * cols * sizeof(int)) ); 


	auto t_0 = std::thread([](int* &x){
			printf("- 0\n"); 	
			const int label = 0; 
			test_inner_gpu<<<rows,cols>>>(label, x);
			},

			std::ref(p_t_0)
			);

	auto t_1 = std::thread([](int* &x){
			printf("- 1\n"); 	
			const int label = 1; 
			test_inner_gpu<<<rows,cols>>>(label, x);
			},

			std::ref(p_t_1)
			);

	auto t_2 = std::thread([](int* &x){
			printf("- 2\n"); 	
			const int label = 2; 
			test_inner_gpu<<<rows,cols>>>(label, x);
			},

			std::ref(p_t_2)
			);

	t_0.join(); 
	t_1.join(); 
	t_2.join(); 


	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

#if 1

	auto max_idx = rows * cols - 1; 
	cout << p_t_0[max_idx] << endl; 
	cout << p_t_1[max_idx] << endl; 
	cout << p_t_2[max_idx] << endl; 

	assert(p_t_0[max_idx] == max_idx*max_idx + 0); 
	assert(p_t_1[max_idx] == max_idx*max_idx + 1); 
	assert(p_t_2[max_idx] == max_idx*max_idx + 2); 


#endif 	


    CUDA_CHECK_RETURN(cudaFree(p_t_0)); 
    CUDA_CHECK_RETURN(cudaFree(p_t_1)); 
    CUDA_CHECK_RETURN(cudaFree(p_t_2)); 

	printf("- END\n"); 


    // jd end 

    __P__;
}
