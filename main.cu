#include "/home/hpc/cWB/TEST/S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2/macro/gpu_struct.h"
#include "main.cuh"
#include <xmmintrin.h>
#include "wavearray.hh"
#include "gpu_network.hh"
//#include "function.h"
/*#include "cwb.hh"
#include "cwb2G.hh"
#include "config.hh"
#include "network.hh"
#include "TString.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TRandom.h"
#include "TComplex.h"*/

#define num_blocks 16											// 16 blocks
#define num_threads 256											// 256 threads per block
#define shared_memory_usage 0										// no share memory

#define eTDTotal 3
#define TSize 1000
#define StreamNum 16
#define BufferNum 16  

#define CUDA_CHECK(value) {                                             \
    cudaError_t _m_cudaStat = value;                                    \
    if (_m_cudaStat != cudaSuccess) {                                   \
        fprintf(stderr, "Error %s at line %d in file %s\n",             \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
        exit(1); }}

void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int eTDDim, int mlDim, int Lsky)// allocate locked memory on CPU 
{
	for(int i = 0; i<BufferNum; i++)
	{
		CUDA_CHECK(cudaMallocHost(&(pre_gpu_data[i].other_data->T_En), eTDDim * sizeof(float) ) );
		cout<<"alloc eTD"<<endl;
	}
	for( int i = 0; i<StreamNum; i++)
	{	
		CUDA_CHECK(cudaMallocHost(&(post_gpu_data[i].output->rE), eTDDim * sizeof(float) ) );
		cout<<"alloc rE"<<endl;
	}
		return;
}

void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int eTDDim, int mlDim, int Lsky)// allocate memory on GPU
{
	for (int i = 0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaMalloc( (void**)&skyloop_output[i].rE, eTDDim * sizeof(float) ) );
	}
	
	return;
}

void cleanup_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data)
{
 	for(int i=0; i<BufferNum; i++)
	{
		CUDA_CHECK(cudaFreeHost(&pre_gpu_data[i].other_data->T_En));
		cout<<"cleanup eTD"<<endl;
	}		
	for( int i = 0; i<StreamNum; i++)
	{		
		CUDA_CHECK(cudaFreeHost(&post_gpu_data[i].output->rE));
		cout<<"cleanup rE"<<endl;
	}
}
		
//void cleanup_cpu_mem(struct skyloop_output *skyloop_output)

__host__ void push_work_into_gpu(struct pre_data *input_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, struct post_data *post_gpu_data, int work_size, int eTDDim, int mlDim, int Lsky, cudaStream_t *stream)
{
        for(int i = 0; i<work_size ; i++)
        {
                for(int j = 0; j<NIFO ; j++)
                {
                        cudaMemcpyAsync(skyloop_other[i].eTD[j], input_data[i].other_data->eTD[j], eTDDim * sizeof(float), cudaMemcpyHostToDevice, stream[i] );
                        cudaMemcpyAsync(skyloop_other[i].ml[j], input_data[i].other_data->ml[j], mlDim * sizeof(short), cudaMemcpyHostToDevice, stream[i] );
                        cudaMemcpyAsync(skyloop_other[i].pa[j], input_data[i].other_data->pa[j], eTDDim * sizeof(float), cudaMemcpyHostToDevice, stream[i] );
                        cudaMemcpyAsync(skyloop_other[i].pA[j], input_data[i].other_data->pA[j], eTDDim * sizeof(float), cudaMemcpyHostToDevice, stream[i] );
                        cudaMemcpyAsync(skyloop_other[i].ml[j], input_data[i].other_data->ml[j], mlDim * sizeof(short), cudaMemcpyHostToDevice, stream[i] );
                        cudaMemcpyAsync(skyloop_other[i].FP[j], input_data[i].other_data->FP[j], mlDim * sizeof(double), cudaMemcpyHostToDevice, stream[i] );
                        cudaMemcpyAsync(skyloop_other[i].FX[j], input_data[i].other_data->FX[j], mlDim * sizeof(double), cudaMemcpyHostToDevice, stream[i] );
                }
				
                cudaMemcpyAsync(skyloop_other[i].mm, input_data[i].other_data->mm, Lsky * sizeof(short), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].T_En, input_data[i].other_data->T_En, sizeof(float), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].T_Es, input_data[i].other_data->T_Es, sizeof(float), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].TH, input_data[i].other_data->TH, sizeof(float), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].netRHO, input_data[i].other_data->netRHO, sizeof(float), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].a_00, input_data[i].other_data->a_00, sizeof(float), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].a_90, input_data[i].other_data->a_90, sizeof(float), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].le, input_data[i].other_data->le, sizeof(int), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].vint_size, input_data[i].other_data->vint_size, sizeof(int), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].rNRG_size, input_data[i].other_data->rNRG_size, sizeof(int), cudaMemcpyHostToDevice, stream[i] );
		cudaMemcpyAsync(skyloop_other[i].lag, input_data[i].other_data->lag, sizeof(int), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].id, input_data[i].other_data->id, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].nIFO, input_data[i].other_data->nIFO, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].V, input_data[i].other_data->V, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].V4, input_data[i].other_data->V4, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].hist, input_data[i].other_data->hist, sizeof(class TH2F*), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].pwc, input_data[i].other_data->pwc, sizeof(class netcluster *), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].nLikelihood, input_data[i].other_data->nLikelihood, sizeof(class skymap*), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].wdmMRA, input_data[i].other_data->wdmMRA, sizeof(class monster*), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].pNRG, input_data[i].other_data->pNRG, sizeof(wavearray<float> *), cudaMemcpyHostToDevice, stream[i] );
				cudaMemcpyAsync(skyloop_other[i].count, input_data[i].other_data->count, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
				cudaMemcpyAsync(skyloop_other[i].finish, input_data[i].other_data->finish, sizeof(bool), cudaMemcpyHostToDevice, stream[i] );
        }
        for(int i = 0; i<work_size ; i++)
        {
                kernel_skyloop<<<num_blocks, num_threads, shared_memory_usage, stream[i]>>>(&skyloop_other[i], &skyloop_output[i], eTDDim, mlDim);
        }
        for(int i = 0; i<work_size ; i++)
        {
                for(int j = 0; j<NIFO ; j++)
                {
                        cudaMemcpyAsync(post_gpu_data[i].other_data->eTD[j], skyloop_other[i].eTD[j], eTDDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                        cudaMemcpyAsync(post_gpu_data[i].other_data->pa[j], skyloop_other[i].pa[j], eTDDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                        cudaMemcpyAsync(post_gpu_data[i].other_data->pA[j], skyloop_other[i].pA[j], eTDDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                        cudaMemcpyAsync(post_gpu_data[i].other_data->ml[j], skyloop_other[i].ml[j], mlDim * sizeof(short), cudaMemcpyDeviceToHost, stream[i] );
                        cudaMemcpyAsync(post_gpu_data[i].other_data->FP[j], skyloop_other[i].FP[j], mlDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                        cudaMemcpyAsync(post_gpu_data[i].other_data->FX[j], skyloop_other[i].FX[j], mlDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                }
                cudaMemcpyAsync(post_gpu_data[i].output->rE, skyloop_output[i].rE, eTDDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output->pE, skyloop_output[i].pE, eTDDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output->Eo, skyloop_output[i].Eo, mlDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output->En, skyloop_output[i].En, mlDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output->Es, skyloop_output[i].Es, mlDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output->Mm, skyloop_output[i].Mm, mlDim * sizeof(int), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->mm, skyloop_other[i].mm, Lsky * sizeof(short), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->T_En, skyloop_other[i].T_En, sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->T_Es, skyloop_other[i].T_Es, sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->TH, skyloop_other[i].TH, sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->netRHO, skyloop_other[i].netRHO, sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->a_00, skyloop_other[i].a_00, sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->a_90, skyloop_other[i].a_90, sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->le, skyloop_other[i].le, sizeof(int), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->vint_size, skyloop_other[i].vint_size, sizeof(int), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->rNRG_size, skyloop_other[i].rNRG_size, sizeof(int), cudaMemcpyDeviceToHost, stream[i] );
		cudaMemcpyAsync(post_gpu_data[i].other_data->lag, skyloop_other[i].lag, sizeof(int), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->id, skyloop_other[i].id, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->nIFO, skyloop_other[i].nIFO, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->V, skyloop_other[i].V, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->V4, skyloop_other[i].V4, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->hist, skyloop_other[i].hist, sizeof(class TH2F*), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->pwc, skyloop_other[i].pwc, sizeof(class netcluster*), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->nLikelihood, skyloop_other[i].nLikelihood, sizeof(class skymap*), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->wdmMRA, skyloop_other[i].wdmMRA, sizeof(class monster*), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].other_data->pNRG, skyloop_other[i].pNRG, sizeof(wavearray<float> *), cudaMemcpyDeviceToHost, stream[i] );
		cudaMemcpyAsync(post_gpu_data[i].other_data->count, skyloop_other[i].count, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
		cudaMemcpyAsync(post_gpu_data[i].other_data->finish, skyloop_other[i].finish, sizeof(bool), cudaMemcpyDeviceToHost, stream[i] );
                cudaStreamAddCallback(stream[i], MyCallback, (void*)&post_gpu_data[i], 0);
        }
}

__global__ void kernel_skyloop (struct other *skyloop_other, struct skyloop_output *skyloop_output,  int eTDDim, int mlDim)
{
        int tn = blockDim.x * gridDim.x;
        int l = threadIdx.x + blockIdx.x * blockDim.x;
        float *pe[NIFO];
        short *ml[NIFO];
        size_t V4;
        while( l < mlDim )
        {
                // init_point_ps
                pe[0] = skyloop_other->eTD[0];
                pe[1] = skyloop_other->eTD[1];
                pe[2] = skyloop_other->eTD[2];
                ml[0] = skyloop_other->ml[0];
                ml[1] = skyloop_other->ml[1];
                ml[2] = skyloop_other->ml[2];
                V4 = *skyloop_other->V4;

                pe[0] = pe[0] + TSize * V4 / 2 + ml[0][l] * V4;
                pe[1] = pe[1] + TSize * V4 / 2 + ml[1][l] * V4;
                pe[2] = pe[2] + TSize * V4 / 2 + ml[2][l] * V4;


                kernel_skyloop_calculate( skyloop_other, skyloop_output, l, pe);
                l += tn;
        }
}

__inline__ __device__ void kernel_skyloop_calculate(struct other *skyloop_other, struct skyloop_output *skyloop_output, int l, float **PE)
{
        float msk;
        float pe[NIFO];
        float rE, pE;
        float Eo,En,Es;
		float T_En, T_Es;				// two threshold
        int Mm,v;
		size_t V;

        Eo = 0;
        En = 0;
        Es = 0;
        Mm = 0;
        v = 0;
		V = *skyloop_other->V;
		T_En = *skyloop_other->T_En;
		T_Es = *skyloop_other->T_Es;
        while( v < V )                                  // Skyloop, for loop
        {
                pe[0] = PE[0][v];
                pe[1] = PE[1][v];
                pe[2] = PE[2][v];
                rE = pe[0] + pe[1] + pe [2];
                msk = ( rE >= T_En );
                Mm += msk;
                pE = rE * msk;
                Eo += pE;
                pE = kernel_minSNE_ps(pE, pe, msk);
                Es += pE;
                msk = ( pE >= T_Es );
                rE *= msk;
                En += rE;
                skyloop_output->rE[v] = rE;
                skyloop_output->pE[v] = pE;
                v++;
        }
        Eo += 0.01;
        Mm = Mm * 2 + 0.01;
        *skyloop_output->Eo = Eo;                        // Write back to output
        *skyloop_output->En = En;
        *skyloop_output->Es = Es;
        *skyloop_output->Mm = Mm;
}

__inline__ __device__ float kernel_minSNE_ps (float pE, float* pe, float msk)
{
        float a,b,c,ab,bc,ac;
        float temp;
        float flag;


        a = pe[0];
        b = pe[1];
        c = pe[2];
        ab = ( a>=b );
        ac = ( a>=c );
        bc = ( b>=c );
        temp = a+b+c - ab*ac*a - (1-ab)*bc*b - (1-ac)*(1-bc)*c;
        flag = ( temp >= pE );
        temp = temp + pE - flag*temp - (1-flag)*pE;
        return temp;
}

void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *post_gpu_data)
{
    float *rE;
    float *pE;
	// other variable
	float *eTD[NIFO];
	float *pe[NIFO];	
	float *pa[NIFO];
	float *pA[NIFO];
	short *ml[NIFO];
	short *mm;
	double *FP[NIFO];
	double *FX[NIFO];
	float T_En, T_Es, TH, netRHO;
	float *a_00, *a_90;
	int le,	vint_size, rNRG_size, lag;
	size_t id, nIFO, V, V4;
	class TH2F *hist;
	class netcluster *pwc;
	class skymap *nLikelihood;
	class monster *wdmMRA;
	wavearray<float> *pNRG;
	size_t *count;
	bool *finish;
	
	// variable used in Callback
	float vvv[NIFO];
	float *v00[NIFO];
	float * v90[NIFO];
	int f_ = NIFO/4;
	int lm, Vm;
	int lb = 0 ;
	float ee, em, Ls, Eo, AA, aa, Lo, To, Ln, rHo;
	float stat, Lm, Em, Am, EE;
	double submra = 0; 
	double suball = 0;
	size_t m4, m;
	bool mra = false;
	__m128 _E_n;
	__m128 _E_s;
        class netpixel *pix;
	
	int l;							// indicate the location
	bool is_goto = false;			// goto flag
	// initialize other variable
	lm = Vm = -1;
	stat = Lm = Em = Am = EE = 0.;
	le = *((post_data*)post_gpu_data)->other_data->le;
	V4 = *((post_data*)post_gpu_data)->other_data->V4;
	T_En = *((post_data*)post_gpu_data)->other_data->T_En;
	T_Es = *((post_data*)post_gpu_data)->other_data->T_Es;
	TH = *((post_data*)post_gpu_data)->other_data->TH;
	netRHO = *((post_data*)post_gpu_data)->other_data->netRHO;
	a_00 = ((post_data*)post_gpu_data)->other_data->a_00;
	a_90 = ((post_data*)post_gpu_data)->other_data->a_90;
	vint_size = *((post_data*)post_gpu_data)->other_data->vint_size;
	rNRG_size = *((post_data*)post_gpu_data)->other_data->rNRG_size;
	lag = *((post_data*)post_gpu_data)->other_data->lag;
	id = *((post_data*)post_gpu_data)->other_data->id;
	nIFO = *((post_data*)post_gpu_data)->other_data->nIFO;
	V = *((post_data*)post_gpu_data)->other_data->V;
	hist = ((post_data*)post_gpu_data)->other_data->hist;
	pwc = ((post_data*)post_gpu_data)->other_data->pwc;
	nLikelihood = ((post_data*)post_gpu_data)->other_data->nLikelihood;
	wdmMRA = ((post_data*)post_gpu_data)->other_data->wdmMRA;
	pNRG = ((post_data*)post_gpu_data)->other_data->pNRG;
	count = ((post_data*)post_gpu_data)->other_data->count;
	finish = ((post_data*)post_gpu_data)->other_data->finish;
	
	rE = (float*)malloc(TSize * V4 * sizeof(float) );
	pE = (float*)malloc(TSize * V4 * sizeof(float) );
	rE = ((post_data*)post_gpu_data)->output->rE;
	pE = ((post_data*)post_gpu_data)->output->pE;
        for(int i = 0; i<NIFO; i++)
        {		
                eTD[i] = (float*)malloc(TSize * V4 * sizeof(float) );
                pa[i] = (float*)malloc(TSize * V4 * sizeof(float) );
                pA[i] = (float*)malloc(TSize * V4 * sizeof(float) );
                ml[i] = (short*)malloc((le + 1) *sizeof(short) );
                FP[i] = (double*)malloc((le + 1) * sizeof(double) );
                FX[i] = (double*)malloc((le + 1) * sizeof(double) );

                eTD[i] = ((post_data*)post_gpu_data)->other_data->eTD[i];
                pa[i] = ((post_data*)post_gpu_data)->other_data->pa[i];
                pA[i] = ((post_data*)post_gpu_data)->other_data->pA[i];
                ml[i] = ((post_data*)post_gpu_data)->other_data->ml[i];
                FP[i] = ((post_data*)post_gpu_data)->other_data->FP[i];
                FX[i] = ((post_data*)post_gpu_data)->other_data->FX[i];
        }
		
        mm = (short*)malloc((le + 1) *sizeof(short) );
        mm = ((post_data*)post_gpu_data)->other_data->mm;
        wavearray<float>  fp(NIFO*V4);  fp=0;            // aligned array for + antenna pattern 
        wavearray<float>  fx(NIFO*V4);  fx=0;            // aligned array for x antenna pattern 
        wavearray<float>  nr(NIFO*V4);  nr=0;            // aligned array for inverse rms 
        wavearray<float>  Fp(NIFO*V4);  Fp=0;            // aligned array for pattern 
        wavearray<float>  Fx(NIFO*V4);  Fx=0;            // aligned array for patterns 
        wavearray<float>  am(NIFO*V4);  am=0;            // aligned array for TD amplitudes 
        wavearray<float>  AM(NIFO*V4);  AM=0;            // aligned array for TD amplitudes 
        wavearray<float>  bb(NIFO*V4);  bb=0;            // temporary array for MRA amplitudes 
        wavearray<float>  BB(NIFO*V4);  BB=0;            // temporary array for MRA amplitudes 
        wavearray<float>  xi(NIFO*V4);  xi=0;            // 00 array for reconctructed responses 
        wavearray<float>  XI(NIFO*V4);  XI=0;            // 90 array for reconstructed responses

        __m128* _Fp = (__m128*) Fp.data;
        __m128* _Fx = (__m128*) Fx.data;
        __m128* _am = (__m128*) am.data;
        __m128* _AM = (__m128*) AM.data;
        __m128* _xi = (__m128*) xi.data;
        __m128* _XI = (__m128*) XI.data;
        __m128* _fp = (__m128*) fp.data;
        __m128* _fx = (__m128*) fx.data;
        __m128* _nr = (__m128*) nr.data;
        __m128* _bb = (__m128*) bb.data;
        __m128* _BB = (__m128*) BB.data;

        __m128* _aa = (__m128*) a_00;
        __m128* _AA = (__m128*) a_90;
	
	// callback caculation begin
	skyloop:
        for ( l = 0; l <= le ; l++)
        {
		if(!mm[l] || l<0) continue;                  // skip delay configurations	
		if(!is_goto)
		{	
               	 	Ln = ((post_data*)post_gpu_data)->output->En[l];
	                Eo = ((post_data*)post_gpu_data)->output->Eo[l];
        	        Ls = ((post_data*)post_gpu_data)->output->Es[l];
                	m = ((post_data*)post_gpu_data)->output->Mm[l];
		}
		else
		{
			pe[0] = eTD[0] + TSize * V4 / 2 + ml[0][l] * V4;
			pe[1] = eTD[1] + TSize * V4 / 2 + ml[1][l] * V4;
			pe[2] = eTD[2] + TSize * V4 / 2 + ml[2][l] * V4;
			_skyloop(pe, V4, T_En, T_Es, rE, pE, l, mm, Ln, Eo, Ls, m);
		}
                aa = Ls*Ln/(Eo-Ls);
                if((aa-m)/(aa+m)<0.33) continue;

                pnt_(v00, pa, ml, (int)l, (int)V4);
                pnt_(v90, pA, ml, (int)l, (int)V4);
                float *pfp = fp.data;
                float *pfx = fx.data;
                float *p00 = a_00;
                float *p90 = a_90;

                m = 0;
                for(int j=0; j<V; j++)
                {
                        int jf = j*f_;
                        cpp_(p00,v00);
                        cpp_(p90,v90);
                        cpf_(pfp, FP, l);
                        cpf_(pfx, FX, l);


                        _sse_zero_ps (_xi + jf);
                        _sse_zero_ps (_XI + jf);
                        _sse_cpf_ps (_am+jf, _aa+jf);
                        _sse_cpf_ps (_AM+jf, _AA+jf);
                        if(rE[j]>T_En)
                                m++;
                }

                __m128* _pp = (__m128*) am.data;
                __m128* _PP = (__m128*) AM.data;

                if(mra)
                {
                        _sse_MRA_ps( xi.data, XI.data, T_En, m, wdmMRA, a_00, a_90, rE, pE, rNRG_size, pNRG);
                        _pp = (__m128*) xi.data;
                        _PP = (__m128*) XI.data;
                }

                m = 0; Ls=Ln=Eo=0;
                for(int j = 0; j<V; j++)
                {
                        int jf = j*f_;
                        int mf = m*f_;
                        _sse_zero_ps(_bb+jf);
                        _sse_zero_ps(_BB+jf);
                        ee = _sse_abs_ps(_pp+jf,_PP+jf);           // total pixel energy
                        if(ee<T_En) continue;
                        _sse_cpf_ps(_bb+mf,_pp+jf);                // copy 00 amplitude/PC
                        _sse_cpf_ps(_BB+mf,_PP+jf);                // copy 90 amplitude/PC
                        _sse_cpf_ps(_Fp+mf,_fp+jf);                // copy F+
                        _sse_cpf_ps(_Fx+mf,_fx+jf);                // copy Fx
                        _sse_mul_ps(_Fp+mf,_nr+jf);                // normalize f+ by rms
                        _sse_mul_ps(_Fx+mf,_nr+jf);                // normalize fx by rms
                        m++;
                        em = _sse_maxE_ps(_pp+jf,_PP+jf);          // dominant pixel energy 
                        Ls += ee-em; Eo += ee;                     // subnetwork energy, network energy
                        if(ee-em>T_Es)
                                Ln += ee;                     // network energy above subnet threshold
                }

                size_t m4 = m + (m%4 ? 4 - m%4 : 0);
                _E_n = _mm_setzero_ps();                     // + likelihood

                for(int j=0; j<m4; j+=4)
                {
                        int jf = j*f_;
                        _sse_dpf4_ps(_Fp+jf,_Fx+jf,_fp+jf,_fx+jf);                // go to DPF
                        _E_s = _sse_like4_ps(_fp+jf,_fx+jf,_bb+jf,_BB+jf);        // std likelihood
                        _E_n = _mm_add_ps(_E_n,_E_s);                             // total likelihood
                }
                _mm_storeu_ps(vvv,_E_n);

                Lo = vvv[0]+vvv[1]+vvv[2]+vvv[3];
                AA = aa/(fabs(aa)+fabs(Eo-Lo)+2*m*(Eo-Ln)/Eo);        //  subnet stat with threshold
                ee = Ls*Eo/(Eo-Ls);
                em = fabs(Eo-Lo)+2*m;                                 //  suball NULL
                ee = ee/(ee+em);                                      //  subnet stat without threshold
                aa = (aa-m)/(aa+m);

                if(AA>stat && !mra)
                {
                        stat=AA; Lm=Lo; Em=Eo; Am=aa; lm=l; Vm=m; suball=ee; EE=em;
                }
        }

        if(!mra && lm>=0) {mra=true; le=lb=lm; goto skyloop;}    // get MRA principle components

        pwc->sCuts[id-1] = -1;
        pwc->cData[id-1].likenet = Lm;
        pwc->cData[id-1].energy = Em;
        pwc->cData[id-1].theta = nLikelihood->getTheta(lm);
        pwc->cData[id-1].phi = nLikelihood->getPhi(lm);
        pwc->cData[id-1].skyIndex = lm;

        rHo = 0.;
        if(mra)
        {
                submra = Ls*Eo/(Eo-Ls);                                     // MRA subnet statistic
                submra/= fabs(submra)+fabs(Eo-Lo)+2*(m+6);                  // MRA subnet coefficient 
                To = 0;
                pwc->p_Ind[id-1].push_back(lm);
                for(int j=0; j<vint_size; j++)
                {
                        pix = pwc->getPixel(id,j);
                        pix->theta = nLikelihood->getTheta(lm);
                        pix->phi   = nLikelihood->getPhi(lm);
                        To += pix->time/pix->rate/pix->layers;
                        if(j==0&&mra) pix->ellipticity = submra;                 // subnet MRA propagated to L-stage
                        if(j==0&&mra) pix->ellipticity = submra;                 // subnet MRA propagated to L-stage
                        if(j==0&&mra) pix->polarisation = fabs(Eo-Lo)+2*(m+6);   // submra NULL propagated to L-stage
                        if(j==1&&mra) pix->ellipticity = suball;                 // subnet all-sky propagated to L-stage
                        if(j==1&&mra) pix->polarisation = EE;                    // suball NULL propagated to L-stage
                }
                To /= vint_size;
                rHo = sqrt(Lo*Lo/(Eo+2*m)/nIFO);                                // estimator of coherent amplitude
        }                                                                       // end of skyloop

        if(hist && rHo>netRHO)
                for(int j=0;j<vint_size;j++) hist->Fill(suball,submra);

        if(fmin(suball,submra)>TH && rHo>netRHO)
        {
               	*count += vint_size;
                if(hist)
                {
                        printf("lag|id %3d|%3d rho=%5.2f To=%5.1f stat: %5.3f|%5.3f|%5.3f ",
                                int(lag),int(id),rHo,To,suball,submra,stat);
                        printf("E: %6.1f|%6.1f L: %6.1f|%6.1f|%6.1f pix: %4d|%4d|%3d|%2d \n",
                                Em,Eo,Lm,Lo,Ls,int(vint_size),int(V),Vm,int(m));
                }
        }
        else pwc->sCuts[id-1]=1;


        V = vint_size;
        for(int j=0; j<V; j++)
        {                           // loop over pixels           
                pix = pwc->getPixel(id,j);
                pix->core = true;
                if(pix->tdAmp.size()) pix->clean();
        }

                                                    // end of loop over clustersa
        *finish = true;
}
                                                    
void _skyloop(float **pe, size_t V4, float T_En, float T_Es, float *rE, float *pE, int l, short *mm, float &Ln, float &Eo, float &Ls, size_t &m)
{

         if(!mm[l] || l<0) return;                  // skip delay configurations

		 float vvv[NIFO];
	
         __m128 _msk;
         __m128 _E_o = _mm_setzero_ps();              // total network energy
         __m128 _E_n = _mm_setzero_ps();              // network energy above the threshold
         __m128 _E_s = _mm_setzero_ps();              // subnet energy above the threshold
         __m128 _M_m = _mm_setzero_ps();              // # of pixels above threshold
         __m128* _rE = (__m128*) rE;           // m128 pointer to energy array     
         __m128* _pE = (__m128*) pE;           // m128 pointer to energy array     
   	 __m128* _pe[NIFO];

	 _pe[0] = (__m128*) pe[0];
	 _pe[1] = (__m128*) pe[1];
	 _pe[2] = (__m128*) pe[2];
		
  	 __m128 _En = _mm_set1_ps(T_En);
         __m128 _Es = _mm_set1_ps(T_Es);
         __m128 _1  = _mm_set1_ps(1.);
         for(int j=0; j<V4; j+=4) {                                // loop over selected pixels 
            *_rE = _sse_sum_ps(_pe);                           // get pixel energy     
            _msk = _mm_and_ps(_1,_mm_cmpge_ps(*_rE,_En));      // E>En  0/1 mask
	        _M_m = _mm_add_ps(_M_m,_msk);                      // count pixels above threshold
    	    *_pE = _mm_mul_ps(*_rE,_msk);                      // zero sub-threshold pixels 
            _E_o = _mm_add_ps(_E_o,*_pE);                      // network energy
            _sse_minSNE_ps(_rE,_pe,_pE);                       // subnetwork energy with _pe increment
            _E_s = _mm_add_ps(_E_s,*_pE);                      // subnetwork energy
            _msk = _mm_and_ps(_1,_mm_cmpge_ps(*_pE++,_Es));    // subnet energy > Es 0/1 mask 
            _E_n = _mm_add_ps(_E_n,_mm_mul_ps(*_rE++,_msk));   // network energy
         }

         _mm_storeu_ps(vvv,_E_n);
         Ln = vvv[0]+vvv[1]+vvv[2]+vvv[3];             // network energy above subnet threshold
         _mm_storeu_ps(vvv,_E_o);
         Eo = vvv[0]+vvv[1]+vvv[2]+vvv[3]+0.01;        // total network energy
         _mm_storeu_ps(vvv,_E_s);
         Ls = vvv[0]+vvv[1]+vvv[2]+vvv[3];             // subnetwork energy
         _mm_storeu_ps(vvv,_M_m);
	     m = 2*(vvv[0]+vvv[1]+vvv[2]+vvv[3])+0.01;     // pixels above threshold
}
