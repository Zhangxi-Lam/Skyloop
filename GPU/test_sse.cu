#include <xmmintrin.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
using namespace std;
void call_my_test_sse();
int my_test_sse();
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void* gpu_b);
__global__ void kernel(int *gpu_a, int *gpu_b);

#define CUDA_CHECK(value) {                                             \
    cudaError_t _m_cudaStat = value;                                    \
    if (_m_cudaStat != cudaSuccess) {                                   \
        fprintf(stderr, "Error %s at line %d in file %s\n",             \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
        exit(1); }}
int main(void)
{
	int *cpu_a[2];
	int *cpu_b[2];
	int *gpu_a[2];
	int *gpu_b[2];
	my_test_sse();
	cout<<"test finish"<<endl;
	cudaHostAlloc(&(cpu_a[0]), 10 *  sizeof(float), cudaHostAllocMapped);
	cudaHostAlloc(&(cpu_b[0]), 10 *  sizeof(float), cudaHostAllocMapped);
	cudaMalloc(&(gpu_a[0]), 10 * sizeof(float) );
	cudaMalloc(&(gpu_b[0]), 10 * sizeof(float) );
	
	for(int i=0; i<10; i++)
		cpu_a[0][i] = 0;
	cudaStream_t stream[2];
	cudaStreamCreate(&stream[0]);
	cudaMemcpyAsync(gpu_a[0], cpu_a[0], 1 * sizeof(float), cudaMemcpyHostToDevice, stream[0] );
	kernel<<<1, 16, 0, stream[0]>>>(gpu_a[0], gpu_b[0]);
	cudaMemcpyAsync(cpu_b[0], gpu_b[0], 1 * sizeof(float), cudaMemcpyDeviceToHost, stream[0] );
	CUDA_CHECK(cudaStreamAddCallback(stream[0], MyCallback, (void*)&gpu_b[0], 0));
	return 0;
}
__global__ void kernel(int *gpu_a, int *gpu_b)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid<10)
		gpu_b[tid] = gpu_a[tid];
	return;
}
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void* gpu_b)
{
	call_my_test_sse();
	return;
}
void call_my_test_sse()
{
	int out;
	out=my_test_sse();
	cout<<out<<endl;
	return;
}
int my_test_sse()
{
	clock_t Clock[10];
	int Lsky = 196608;	
	float vvv[4] = {0, 0, 0, 0};
	__m128 _E_o;
	__m128 _E_n;

	cout<<"nvcc"<<endl;
	cout<<"1"<<endl;
	Clock[0] = clock();
	cout<<"2"<<endl;
	_E_o = _mm_setzero_ps();
	cout<<"3"<<endl;
	
	for(int c=190000; c<Lsky; c++)
		for(int d=0; d<Lsky; d++)
		{
			_E_o = _mm_setzero_ps();
			_E_n = _mm_set1_ps(1.);
			_E_o = _mm_add_ps(_E_o, _E_n);				
		}
	_mm_storeu_ps(vvv,_E_o);	
	cout<<"finish"<<endl;
	cout<<"vvv = "<<vvv[0]<<endl;
	Clock[1] = clock();	
	printf("time = %f\n", (double)(Clock[1]-Clock[0])/CLOCKS_PER_SEC);
	return vvv[0]+vvv[1]+vvv[2]+vvv[3];
}
