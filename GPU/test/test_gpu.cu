#include<iostream>
#include<stdio.h>

bool InitCUDA();

int main()
{
	InitCUDA();
	return 0;	
}

bool InitCUDA(void)
{
  int count = 0;
  int i = 0;
  cudaGetDeviceCount(&count); 
  //cudaDeviceQuery();
  cudaDeviceProp prop;
  for(i = 0; i < count; i++)  
  {
    if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
    {
       if(prop.major >= 1)
       {
          break;
       }
     }
   }
   cudaDeviceProp sDevProp = prop;
   printf( "%d \n", i);
   printf( "Device name: %s\n", sDevProp.name );
   printf( "Device memory: %d\n", sDevProp.totalGlobalMem );
   printf( "Shared Memory per-block: %d\n", sDevProp.sharedMemPerBlock );
   printf( "Register per-block: %d\n", sDevProp.regsPerBlock );
   printf( "Warp size: %d\n", sDevProp.warpSize );
   printf( "Memory pitch: %d\n", sDevProp.memPitch );
   printf( "Constant Memory: %d\n", sDevProp.totalConstMem );
   printf( "Max thread per-block: %d\n", sDevProp.maxThreadsPerBlock );
   printf( "Max thread dim: ( %d, %d, %d )\n", sDevProp.maxThreadsDim[0],
             sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2] );
   printf( "Max grid size: ( %d, %d, %d )\n", sDevProp.maxGridSize[0],  
            sDevProp.maxGridSize[1], sDevProp.maxGridSize[2] );
   printf( "Ver: %d.%d\n", sDevProp.major, sDevProp.minor );
   printf( "Clock: %d\n", sDevProp.clockRate );
   printf( "textureAlignment: %d\n", sDevProp.textureAlignment );
   cudaSetDevice(i);
   printf("\n CUDA initialized.\n");
   return true;
}
