#ifndef _MAIN_
#define _MAIN_
 
void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int eTDDim, int V4max, int Lsky, size_t K);
void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int eTDDim, int V4max , int Lsky, size_t K);
void cleanup_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, cudaStream_t *stream);
void cleanup_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, cudaStream_t *stream);
void QuickSort(size_t *V_array, int *k_array, int p, int r);
int Partition(size_t *V_array, int *k_array, int p, int r);

__host__ void push_work_into_gpu(struct pre_data *input_data, struct post_data *post_gpu_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, size_t *vtddim_array, size_t *etddim_array, size_t *alloced_V4_array, int Lsky, int *pixel_array, int work_size, cudaStream_t *stream);


__global__ void kernel_skyloop(float *eTD, float *vtd_vTD_nr, double *FP_FX, short *ml_mm, size_t *V_tsize, float *gpu_BB, float *gpu_bb, float *gpu_fp, float *gpu_fx, float *gpu_Fp, float *gpu_Fx, float *gpu_tmp, float *gpu_output, int pixelcount);

__global__ void kernel_reduction(float *tmp, float *gpu_output, float *eTD, float *vtd_vTD_nr, short *ml_mm, size_t *V_tsize);

__global__ void kernel_clear(float *tmp);

__inline__ __device__ void kernel_skyloop_calculate(short **ml, float *nr, double **FP, double **FX, float *gpu_BB, float *gpu_bb, float *gpu_fp, float *gpu_fx, float *gpu_Fp, float *gpu_Fx, float **pa, float **pA, float *PE_0, float *PE_1, float *PE_2, size_t V, float *gpu_output,  int l, struct STAT *_s, int tid, int k, int output_ptr);

__inline__ __device__ void kernel_store_result_to_tmp(float *tmp, struct STAT *_s);

__device__ void kernel_store_stat(float *tmp, int tid);

__device__ float kernel_store_final_stat(float *tmp, float *gpu_output, size_t output_ptr);

 __device__ float kernel_minSNE_ps (float pE, float* pe);

__inline__ __device__ void kernel_cpf_(float *a, double **p, size_t i);

__inline__ __device__ void kernel_sse_cpf_ps(float *a, float *p);

__inline__ __device__ void kernel_sse_mul_ps(float *a, float *b);

__inline__ __device__ float kernel_sse_abs_ps(float *bb, float *BB);

__inline__ __device__ float kernel_sse_maxE_ps(float *a, float *A);

__inline__ __device__ void kernel_sse_dpf4_ps(float *Fp, float *Fx, float *fp, float *fx);

__inline__ __device__ void kernel_sse_dpf4_ps(float *Fp, float *Fx, float *fp, float *fx, int k, int l, float *gpu_output);	// debug

__inline__ __device__ void kernel_sse_ort4_ps(float *u, float *v, float *_s, float *_c, int k, int l, float *gpu_output);	// debug

__inline__ __device__ void kernel_sse_dot4_ps(float *u, float *v, float *out);

__inline__ __device__ void kernel_sse_rot4p_m_ps(float *Fp, float *_c, float *Fx, float *_s, float *fp, float *fx);

__inline__ __device__ void kernel_sse_rot4m_ps(float *Fx, float *_c, float *Fp, float *_s, float *fx);

__inline__ __device__ void kernel_sse_like4_ps(float *fp, float *fx, float *bb, float *BB, float *_Es);


void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* post_gpu_data);	
void MyCallback(struct post_data *post_gpu_data, float &Lo);
#endif
