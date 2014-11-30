#!/bin/bash
nvcc -arch=sm_20 --compiler-options '-fPIC' -o main.so --shared main.cu -I /home/hpc/cWB/root-v5-32/include/ -I /home/hpc/cWB/trunk/wat/
