Readme
======

Overview
------
	to build and run gpu version `BQSR-PR`, 
	the `rows` and the `cols` is used to control the threads number.

	this GPU version is >= 15x fater than C++ single version , 
	and >= 4x faster than C++ multiple thread (40 cores).

	author: Jidor Tang <tanglinqi@genomics.cn> 
	date: 2018-08-21


file list
------
	readme.txt 
	main_pr_gpu.cu
	Makefile
	t0/sample_small_17B029145-1-79.realn.sam	:	an sample sam to try
	t0/all_fcout.recal_data.grp.txt				:	an sample grp file to try
	t0/c_realn.recal.sam						:	an should-be recal-sam file 


make & run 
------
	cd to the folder where `readme.txt` is, then run:
	> make 
	> ./main_pr_gpu -grpPath /home/bgi902/t/cuda_demo/0_Simple/main_pr_gpu/t0/all_fcout.recal_data.grp.txt -I /home/bgi902/t/cuda_demo/0_Simple/main_pr_gpu/t0/sample_small_17B029145-1-79.realn.sam  -o /home/bgi902/t/cuda_demo/0_Simple/main_pr_gpu/t0/sample_small_17B029145-1-79.realn.sam_out


synopsis 
------
	-grpPath:	the grp table pass in
	-I:			the input sam file path
	-o:			the output sam file path


