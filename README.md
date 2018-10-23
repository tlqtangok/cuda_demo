# cuda_demo
Author : tlqtangok@126.com (Jidor Tang) at 2018-07-25

cuda guide for new comers. 

# to use 
`cd cuda_demo/cuda_demo/0_Simple/`
you will see many project names. 

use :

```
cd cuda_demo/cuda_demo/PROJECT_NAME/
make 
make run 
make clean 
```
to try. the **PROJECT_NAME** is one of 

- verify_nvcc_env
- try_idx  
- split_tid  
- DEFINE_I_J_K_i_j_k_idx  
- clear_matrix_edge  
- bind_texture
- shared_memory
- shared_memory_merge


enjoy yourself :)


# project's explaination.

- `verify_nvcc_env`

  verify if nvcc is ready. make sure you "nvcc -V" can work properly.



- `try_idx`

  try if you can see the idx number from multi-thread.



- `DEFINE_I_J_K_i_j_k_idx`

  about how to figure out the idx.



- `split_tid`

  how to use the idx to do some computing job.


  
- `clear_matrix_edge`

  a real example of using cuda to do matrix transformation.

- `shared_memory`

  usage of shared memory. very simple usage. no recommend

- `shared_memory_merge`

  to sum an array on shared memory


# extra article
please see [NVIDIA CUDA 入门与并行优化建议](https://gitbook.cn/gitchat/activity/5b49ecf81f72d149b2ded0b0)
