#ifndef _MAIN_
#define _MAIN_
 
void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int eTDDim, int V4max, int Lsky);
void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int eTDDim, int V4max , int Lsky);
void cleanup_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, cudaStream_t *stream);
void cleanup_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, cudaStream_t *stream);
void test_function(void);

__host__ void push_work_into_gpu(struct pre_data *input_data, struct post_data *post_gpu_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, int eTDDim, int V4max, int Lsky, int work_size, cudaStream_t *stream);

__global__ void kernel_skyloop(float *eTD_0, float *eTD_1, float *eTD_2, float *eTD_3, short *ml_0, short *ml_1, short *ml_2, short *ml_3, short *gpu_mm, size_t *gpu_V, size_t *gpu_V4, size_t *gpu_tsize, float *gpu_T_En, float *gpu_T_Es, float *gpu_rE, float *gpu_aa, int Lsky);
__inline__ __device__ void kernel_skyloop_calculate(float *PE_0, float *PE_1, float *PE_2, float *PE_3, size_t V, size_t V4, float T_En, float T_Es, float *gpu_rE, float *gpu_aa, int l);
__inline__ __device__ float kernel_minSNE_ps (float pE, float* pe);
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void* post_gpu_data);

#endif
