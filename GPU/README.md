Skyloop
=======
First, execute 
$ source cWB/trunk/local_watenv.sh

Then compile the main.cu to found the main.so library: 
$ cd cWB/TEST/S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2/macro 
and use this command: 
$ nvcc -arch=sm_20 --compiler-options '-fPIC' -o main.so --shared main.cu -I /home/hpc/cWB/root-v5-32/include/ -I /home/hpc/cWB/trunk/wat/

Then back to the "cWB/TEST/S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2" 
$ cd cWB/TEST/S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2 
And run the program 
$ cwb_inet2G data/coherence_932063240_1000_S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2_job200.root SUPERCLUSTER config/user_parameters.C
(the last line in README in S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2)
