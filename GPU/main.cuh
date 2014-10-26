#ifndef _MAIN_
#define _MAIN_
 
void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int eTDDim, int V4max, int Lsky, size_t K);
void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int eTDDim, int V4max , int Lsky, size_t K);
void cleanup_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, cudaStream_t *stream);
void cleanup_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, cudaStream_t *stream);
void QuickSort(size_t *V_array, int *k_array, int p, int r);
int Partition(size_t *V_array, int *k_array, int p, int r);

__host__ void push_work_into_gpu(struct pre_data *input_data, struct post_data *post_gpu_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, size_t *alloced_V4_array, size_t *etddim_array, int Lsky, int *pixel_array, int work_size, cudaStream_t *stream);

__global__ void kernel_skyloop(float *eTD, short *ml_mm, size_t *V_tsize, float *gpu_output, int pixelcount); 
__inline__ __device__ void kernel_skyloop_calculate(float *PE_0, float *PE_1, float *PE_2, float *PE_3, size_t V, size_t V4, size_t rEDim, float *gpu_output, int l, size_t output_ptr);
__inline__ __device__ float kernel_minSNE_ps (float pE, float* pe);
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void* post_gpu_data);	
void after_skyloop(void* post_gpu_data, int pixelcount, size_t output_ptr);

#endif
