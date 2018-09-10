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
#include <unordered_map>
#include <cfloat>
#include <unordered_map>
#include <map>



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


// jd add 


#if 1 
static const int rows = 6;
static const int cols = 5;
#endif 





__global__ void test(cudaTextureObject_t tex)
{

    DEFINE_J_j_idx; 

    printf("%d : %f\n", idx, tex1Dfetch<float>(tex, idx));

}




texture<float, 1, cudaReadModeElementType> tex1;
__global__ void test1()
{

    DEFINE_J_j_idx;

    printf("%d : %f\n", idx, tex1Dfetch(tex1, idx));

}

// main__ 
int main(int argc, char **argv)
{

#if 1 //simple bind texture

    dim3 g_(rows, 1, 1);
    dim3 b_(cols, 1, 1);

    float *buffer;
    cudaMallocManaged(&buffer, rows * cols * sizeof(float));


    for (int i = 0; i < rows * cols; i++)
    {
        buffer[i] = i*0.1f;

    }
   
    cudaBindTexture(0, tex1, buffer, rows * cols * sizeof(float));

    buffer[2] = 222.222;

    test1 << <g_, b_ >> >();


    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaUnbindTexture(tex1);


#endif 



#if 0 
    dim3 g_(rows, 1, 1);
    dim3 b_(cols, 1, 1);

    float *buffer;
    cudaMallocManaged(&buffer, rows * cols * sizeof(float));


    for (int i = 0; i < rows * cols; i++)
    {
        buffer[i] = i*0.1f;
    }

    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = buffer;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = rows * cols * sizeof(float);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    // create texture object: we only have to do this once!
    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);


	buffer[2] = 333.333;  // still can work !!! 

    test << <g_, b_ >> >(tex);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaDestroyTextureObject(tex);
#endif 






	__P__;
}





