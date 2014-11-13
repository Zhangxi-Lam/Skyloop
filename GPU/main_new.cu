#include "main.cuh"
#include "gpu_network.hh"
#include "wavearray.hh"
#include <xmmintrin.h>
#include "/home/hpc/cWB/trunk/wat/GPU/gpu_struct.h"

#define num_blocks 16 
#define num_threads 256 
#define shared_memory_usage 0

#define StreamNum 1 
#define BufferNum 1 
#define CONSTANT_SIZE 1500
#define MaxPixel 10 
#define CLOCK_SIZE 10
#define outputSize 20

network *gpu_net;
TH2F *gpu_hist;
netcluster *pwc;
int gpu_Lsky;
double *FP[NIFO];
double *FX[NIFO];
float *pa[StreamNum][MaxPixel][NIFO];
float *pA[StreamNum][MaxPixel][NIFO];
double gpu_time[CLOCK_SIZE];
size_t streamCount[StreamNum];	// the result of each stream

// GPU constant memory
__constant__ float constEn, constEs;	// two threshold
__constant__ int constLsky;
__constant__ size_t constK;
#define CUDA_CHECK(value) {                                             \
    cudaError_t _m_cudaStat = value;                                    \
    if (_m_cudaStat != cudaSuccess) {                                   \
        fprintf(stderr, "Error %s at line %d in file %s\n",             \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
        exit(1); }}
extern void after_skyloop(void *post_gpu_data, network *net, TH2F *hist, netcluster *pwc, double **FP, double **FX, float **pa, float **pA, int pixelcount, size_t output_ptr, int Lsky, double *gpu_time, size_t *streamCount, int &cc);
extern void my_test_sse(void);
long gpu_subNetCut(network *net, int lag, float snc, TH2F *hist, double *time)
{
	// define variables
	size_t nIFO = net->ifoList.size();
	
	float En = 2*net->acor*net->acor*nIFO;  // network energy threshold in the sky loop
        float Es = 2*net->e2or;                 // subnet energy threshold in the sky loop
        float TH = fabs(snc);                   // sub network threshold

        int l;
        float aa, AA;
        size_t i, j, k, V, V4, id, K;
        int Lsky = int(net->index.size());
	gpu_Lsky = Lsky;
        short *mm = net->skyMask.data;

        short *ml[NIFO];

	std::vector<int> pI;                    // buffer for pixel IDs 
        wavearray<double> cid;                  // buffer for cluster ID
        netpixel *pix;
        std::vector<int> *vint;
        pwc = &net->wc_List[lag];
        size_t count = 0;
        size_t tsize = 0;
        size_t V4max = 0;                       // store the maximum of V4
        size_t Tmax = 0;                        // store the maximum of tsize
        size_t *V_array, *tsize_array, *V4_array;
        int *k_sortArray;
        int kcount = 0;                         // store the k that is not rejected/processed
        int CombineSize;
	bool CombineFinish = false;		
        int etd_ptr, v_ptr;                          // indicate the eTD's, vtd's and vTD's location
        size_t etddim_array[StreamNum];
        size_t alloced_V4_array[StreamNum];
        int pixel_array[StreamNum];
        int pixelCount;                         // indicate the pixel number of each stream
        size_t alloced_V4;                      // indicate the overall V4 of each stream
        int etddim;              

	for( i=0; i<NIFO; i++)
        {
                if(i<nIFO)
                {
                        ml[i] = net->getifo(i)->index.data;
                        FP[i] = net->getifo(i)->fp.data;
                        FX[i] = net->getifo(i)->fx.data;
                }
                else
                {
                        ml[i] = net->getifo(0)->index.data;
                        FP[i] = net->getifo(0)->fp.data;
                        FX[i] = net->getifo(0)->fx.data;
                }
        }

//++++++++++++++++++++++++++++++++
// find out the maximum of V and tsize 
//++++++++++++++++++++++++++++++++
	cid = pwc->get((char*)"ID",  0,'S',0);                  // get cluster ID
        K = cid.size();

        V_array = (size_t*)malloc(sizeof(size_t) * K);
        V4_array = (size_t*)malloc(sizeof(size_t) * K);
        tsize_array = (size_t*)malloc(sizeof(size_t) * K);
        k_sortArray = (int*)malloc(sizeof(int) * K);

	for(k=0; k<K; k++)
	{
		V_array[k] = 0;
		tsize_array[k] = 0;
		id = size_t(cid.data[k]+0.1);
		if(pwc->sCuts[id-1] != -2) continue;    // skip rejected/processed culster
                vint = &(pwc->cList[id-1]);             // pixel list
                V = vint->size();                       // pixel list size
                if(!V) continue;
                pI = net->wdmMRA.getXTalk(pwc, id);
                V = pI.size();                          // number of loaded pixels
                if(!V) continue;
                pix = pwc->getPixel(id, pI[0]);
                tsize = pix->tdAmp[0].size();
                if(!tsize || tsize&1)                   // tsize%1 = 1/0 = power/amplitude 
                {
                        cout<<"network::subNetCut() error: wrong pixel TD data\n";
                        exit(1);
                }

                tsize /= 2;
                V4 = V + (V%4 ? 4 - V%4 : 0);
                V_array[k] = V;
                V4_array[k] = V4;
                tsize_array[k] = tsize;
                k_sortArray[kcount] = k;
                kcount++;
                if( tsize > Tmax )
                        Tmax = tsize;
                if( V4 > V4max )
                        V4max = V4;
        }
	CombineSize = V4max;
//++++++++++++++++++++++++++++++++
// declare the variables used for gpu calculation 
//++++++++++++++++++++++++++++++++
	struct pre_data pre_gpu_data[BufferNum];		
	struct post_data post_gpu_data[StreamNum];      // store the data transfer from gpu
        struct skyloop_output skyloop_output[StreamNum];// store the skyloop_output data
        struct other skyloop_other[StreamNum];          // store the data which is not output

        int vTDDim = 0;                                 // the size of each eTD
        int alloced_gpu = 0;                            // the number of gpu which has been allocated data

        vTDDim = Tmax * V4max;
        for(int i=0; i<StreamNum; i++)
                streamCount[i] = 0;

	// allocate the memory on cpu and gpu
	allocate_cpu_mem(pre_gpu_data, post_gpu_data, vTDDim, V4max, Lsky, K);
	allocate_gpu_mem(skyloop_output, skyloop_other, vTDDim, V4max, Lsky, K);
	
	gpu_net = net;
	gpu_hist = hist;
	cudaStream_t stream[StreamNum];			// define the stream
	for(int i=0; i<StreamNum; i++)
		CUDA_CHECK(cudaStreamCreate(&stream[i]));       // create the new stream
        cudaMemcpyToSymbol(constEn, &En, sizeof(float));
        cudaMemcpyToSymbol(constEs, &Es, sizeof(float));
	cudaMemcpyToSymbol(constLsky, &Lsky, sizeof(int));
	cudaMemcpyToSymbol(constK, &K, sizeof(size_t));
	


	return;

}
void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int vTDDim, int V4max, int Lsky, size_t K)// allocate locked memory on CPU 
{
        for(int i = 0; i<BufferNum; i++)
        {
                CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.vtd_vTD_nr), 2 * NIFO * vTDDim * sizeof(float) + NIFO * V4max * sizeof(float) + MaxPixel * sizeof(float), cudaHostAllocMapped ) );
        }
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.FP_FX),  2 * NIFO * Lsky * sizeof(double), cudaHostAllocMapped ) );
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.ml_mm), (1 + NIFO) * Lsky * sizeof(short), cudaHostAllocMapped ) );
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.V_tsize), K * 2 * sizeof(size_t), cudaHostAllocMapped ) );
        for(int i = 0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.output), outputSize * sizeof(float), cudaHostAllocMapped ) );
                post_gpu_data[i].other_data.ml_mm = (short*)malloc(sizeof(size_t) * (1 + NIFO) * Lsky);
        }
        return;
}
void cleanup_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, cudaStream_t *stream)
{
        for(int i = 0; i<BufferNum; i++)
        {
                CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.vtd_vTD_nr));
        }
        CUDA_CHECK(cudaFreeHost(pre_gpu_data[0].other_data.FP_FX));
        CUDA_CHECK(cudaFreeHost(pre_gpu_data[0].other_data.ml_mm));
        CUDA_CHECK(cudaFreeHost(pre_gpu_data[0].other_data.V_tsize));
        for(int i=0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaFreeHost(post_gpu_data[i].output.output));
                free(post_gpu_data[i].other_data.ml_mm);
        }
        return;
}
void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int vTDDim, int V4max, int Lsky, size_t K)// allocate the memory on GPU
{
	for(int i = 0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].vtd_vTD_nr), 2 * NIFO * vTDDim * sizeof(float) + NIFO * V4max * sizeof(float) + MaxPixel * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_output[i].output), outputSize * sizeof(float) ) );
        }
        CUDA_CHECK(cudaMalloc(&(skyloop_other[0].FP_FX), 2 * NIFO * Lsky * sizeof(double) ) );
        CUDA_CHECK(cudaMalloc(&(skyloop_other[0].ml_mm), (1 + NIFO) * Lsky * sizeof(short) ) );
        CUDA_CHECK(cudaMalloc(&(skyloop_other[0].V_tsize), K * 2 * sizeof(size_t) ) );

}
void cleanup_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, cudaStream_t *stream)// cleanup the memory on GPU
{
        for(int i = 0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaFree(skyloop_other[i].vtd_vTD_nr) );
                CUDA_CHECK(cudaFree(skyloop_output[i].output) );
        }
        CUDA_CHECK(cudaFree(skyloop_other[0].FP_FX) );
        CUDA_CHECK(cudaFree(skyloop_other[0].ml_mm) );
        CUDA_CHECK(cudaFree(skyloop_other[0].V_tsize) );
        return;
}

