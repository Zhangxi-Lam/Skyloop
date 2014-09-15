#ifndef _MAIN_
#define _MAIN_
 
void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int eTDDim, int mlDim, int Lsky);
void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int eTDDim, int mlDim, int Lsky);
void cleanup_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data);
void cleanup_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other);

__host__ void push_work_into_gpu(struct pre_data *input_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, struct post_data *post_gpu_data, int work_size, int eTDDim, int mlDim, int Lsky, cudaStream_t *stream);

__global__ void kernel_skyloop (struct other* skyloop_other, struct skyloop_output *skyloop_output, int eTDDim, int mlDim);
__inline__ __device__ void kernel_skyloop_calculate(struct other *skyloop_other, struct skyloop_output *skyloop_output, int l, float **PE);
__inline__ __device__ float kernel_minSNE_ps (float pE, float* pe, float msk);
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void* post_gpu_data);

void _skyloop(float **pe, size_t V4, float T_En, float T_Es, float *rE, float *pE, int l, short *mm, float &Ln, float &Eo, float &Ls, size_t &m);
#endif
