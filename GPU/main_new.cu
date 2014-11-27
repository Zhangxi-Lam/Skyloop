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
#define OutputSize 8 
#define CLOCK_SIZE 10
#define LOUD 300

//inline int _sse_MRA_ps(network *net, float *amp, float *AMP, float Eo, int K);

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
extern void after_skyloop(void *post_gpu_data, network *net, TH2F *hist, netcluster *pwc, double **FP, double **FX, float **pa, float **pA, int pixelcount, size_t output_ptr, int Lsky, double *gpu_time, size_t *streamCount);
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
	double xx[NIFO];

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
	bool CombineFinish = false;		
        int v_ptr;                          // indicate the eTD's, vtd's and vTD's location
        size_t vtddim_array[StreamNum];
        int pixel_array[StreamNum];
        int pixelCount;                         // indicate the pixel number of each stream
        int vtddim;              

	for(i=0; i<CLOCK_SIZE; i++)
		gpu_time[i] = 0;
	
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
//++++++++++++++++++++++++++++++++
// declare the variables used for gpu calculation 
//++++++++++++++++++++++++++++++++
	struct pre_data pre_gpu_data[BufferNum];		
	struct post_data post_gpu_data[StreamNum];      // store the data transfer from gpu
        struct skyloop_output skyloop_output[StreamNum];// store the skyloop_output data
        struct other skyloop_other[StreamNum];          // store the data which is not output

	int vDim = 0;					// the size of each vtd and vTD
	int v_size;					// the overall size of vtd_vTD_nr	
	int nr_size;					// the size of nr.data
        int alloced_gpu = 0;                            // the number of gpu which has been allocated data

        vDim = Tmax * V4max;
	nr_size = NIFO * V4max;
	v_size = vDim * NIFO * 2 + MaxPixel + nr_size;
        for(int i=0; i<StreamNum; i++)
                streamCount[i] = 0;
	// allocate the memory on cpu and gpu
	allocate_cpu_mem(pre_gpu_data, post_gpu_data, vDim, V4max, Lsky, K);
	allocate_gpu_mem(skyloop_output, skyloop_other, vDim, V4max, Lsky, K);
	
	gpu_net = net;
	gpu_hist = hist;
	cudaStream_t stream[StreamNum];			// define the stream
	for(int i=0; i<StreamNum; i++)
		CUDA_CHECK(cudaStreamCreate(&stream[i]));       // create the new stream
        cudaMemcpyToSymbol(constEn, &En, sizeof(float));
        cudaMemcpyToSymbol(constEs, &Es, sizeof(float));
	cudaMemcpyToSymbol(constLsky, &Lsky, sizeof(int));
	cudaMemcpyToSymbol(constK, &K, sizeof(size_t));
	
	for(int i=0; i<BufferNum; i++)          // initialize the data
        {
                for(int l=0; l<Lsky; l++)
                {
                        for(int j=0; j<NIFO; j++)
                        {
                                int ptr;
                                ptr = j*Lsky;
                                post_gpu_data[i].other_data.ml_mm[ptr + l] = ml[j][l];
				if(i==0)
				{
					pre_gpu_data[0].other_data.FP_FX[ptr + l] = FP[j][l];
					pre_gpu_data[0].other_data.FP_FX[NIFO*Lsky + ptr + l] = FX[j][l];
					pre_gpu_data[0].other_data.ml_mm[ptr + l] = ml[j][l];
				}
                        }
                        post_gpu_data[i].other_data.ml_mm[NIFO*Lsky + l] = mm[l];
			if(i==0)
				pre_gpu_data[0].other_data.ml_mm[NIFO*Lsky + l] = mm[l];
                }
                post_gpu_data[i].other_data.T_En = En;
                post_gpu_data[i].other_data.T_Es = Es;
                post_gpu_data[i].other_data.TH = TH;
                post_gpu_data[i].other_data.le = Lsky - 1;
                post_gpu_data[i].other_data.lag = lag;
                post_gpu_data[i].other_data.nIFO = nIFO;
        }
	for(int j=0; j<StreamNum; j++)
		for(int i=0; i<MaxPixel; i++)
		{
                        post_gpu_data[j].other_data.id[i] = 0;	
			post_gpu_data[j].other_data.k[i] = 0;
                        post_gpu_data[j].other_data.V[i] = 0;
                        post_gpu_data[j].other_data.V4[i] = 0;
                        post_gpu_data[j].other_data.tsize[i] = 0;	
		}
	for(int k=0; k<K; k++)
	{
		pre_gpu_data[0].other_data.V_tsize[k] = V_array[k];
		pre_gpu_data[0].other_data.V_tsize[k + K] = tsize_array[k];	
	}
	cudaMemcpyAsync(skyloop_other[0].FP_FX, pre_gpu_data[0].other_data.FP_FX, 2 * NIFO * Lsky * sizeof(double), cudaMemcpyHostToDevice, stream[0] );
	cudaMemcpyAsync(skyloop_other[0].ml_mm, pre_gpu_data[0].other_data.ml_mm, (1 + NIFO) * Lsky * sizeof(short), cudaMemcpyHostToDevice, stream[0] );
	cudaMemcpyAsync(skyloop_other[0].V_tsize, pre_gpu_data[0].other_data.V_tsize, K * 2 * sizeof(size_t), cudaMemcpyHostToDevice, stream[0] );
//++++++++++++++++++++++++++++++++
// loop over cluster
//++++++++++++++++++++++++++++++++

	QuickSort(V_array, k_sortArray, 0, kcount-1);
	cid = pwc->get((char*)"ID", 0,'S',0);		// get cluster ID
	K = cid.size();
	v_ptr = MaxPixel;
	pixelCount = 0;
	for(int z=0; z<kcount;)			// loop over unskiped clusters
	{
		while(!CombineFinish && z<kcount)
		{
		k = k_sortArray[z];
		V = V_array[k];
		V4 = V4_array[k];
		tsize = tsize_array[k];
		vtddim = V4 * tsize;
		nr_size = NIFO * V4;
		if( (v_ptr + 2*NIFO*vtddim + nr_size) <= v_size )
		{
			id = size_t(cid.data[k]+0.1);
                        pI = net->wdmMRA.getXTalk(pwc, id);
			for(j=0; j<V; j++)
			{
				pix = pwc->getPixel(id,pI[j]);
				double rms = 0.;
				for(i=0; i<nIFO; i++)
				{
					xx[i] = 1./pix->data[i].noiserms;
					rms += xx[i]*xx[i];
				}
			
				for(i=0; i<nIFO; i++)
                                {
					pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[v_ptr + vtddim*NIFO*2 + j*NIFO+i] = (float)xx[i]/sqrt(rms);	// normalized 1/rms
                                        for( l=0; l<tsize; l++)
                                        {
                                                aa = pix->tdAmp[i].data[l];             // copy TD 00 data 
                                                AA = pix->tdAmp[i].data[l+tsize];       // copy TD 90 data 
						pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[i*vtddim + l*V4+j + v_ptr] = aa;
						pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[NIFO*vtddim + i*vtddim + l*V4+j + v_ptr] = AA;
                                                // assign the data 
                                                if(i == nIFO - 1 && NIFO > nIFO)
                                                	for(int I = nIFO; I<NIFO; I++)
								{
									pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[I*vtddim + l*V4+j + v_ptr] = 0;
									pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[NIFO*vtddim + I*vtddim + l*V4+j + v_ptr] = 0;
									if(j==(V-1))
										for(int J=V; J<V4; J++)
										{
                                                        				pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[I*vtddim + l*V4+J + v_ptr] = 0;
                                                        				pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[NIFO*vtddim + I*vtddim + l*V4+J + v_ptr] = 0;
										}
								}
                                        }
                                }
				
			}

			i = alloced_gpu;
			v_ptr += 2*NIFO*vtddim + nr_size;	
			pre_gpu_data[i].other_data.vtd_vTD_nr[pixelCount] = k+1;
                        post_gpu_data[i].other_data.k[pixelCount] = k+1;
                        post_gpu_data[i].other_data.V[pixelCount] = V;
                        post_gpu_data[i].other_data.V4[pixelCount] = V4;
                        post_gpu_data[i].other_data.tsize[pixelCount] = tsize;
                        post_gpu_data[i].other_data.id[pixelCount] = id;
                        pixelCount++;
			z++;
			//cout<<"z = "<<z<<" kcount = "<<kcount<<endl;
			if(pixelCount >= MaxPixel)
				CombineFinish = true;
		}
		else
			CombineFinish = true;
		}
		post_gpu_data[i].other_data.stream = i;
		vtddim_array[i] = v_ptr;
		pixel_array[i] = pixelCount;
		alloced_gpu++;
//++++++++++++++++++++++++++++++++
// assign the data 
//++++++++++++++++++++++++++++++++
		if(alloced_gpu == StreamNum)
		{
			push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, vtddim_array, Lsky, pixel_array, StreamNum, stream);
			for(int i=0; i<StreamNum; i++)
				CUDA_CHECK(cudaStreamSynchronize(stream[i]));
			//MyCallback(post_gpu_data);
			//clear
			alloced_gpu = 0;
			for(int j=0; j<StreamNum; j++)
				for(int i=0; i<pixel_array[j]; i++)
				{
					post_gpu_data[j].other_data.k[i] = 0;
                                        post_gpu_data[j].other_data.V4[i] = 0;
                                        post_gpu_data[j].other_data.tsize[i] = 0;	
				}
		}
		// clear
		v_ptr = MaxPixel;
		pixelCount = 0;
		CombineFinish = false;
		
	}
	if(alloced_gpu != 0)
	{
		
		push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, vtddim_array, Lsky, pixel_array, StreamNum, stream);
		for(int i=0; i<StreamNum; i++)
			CUDA_CHECK(cudaStreamSynchronize(stream[i]));
		//MyCallback(post_gpu_data);
		alloced_gpu = 0;
	}		
	free(V_array);
	free(V4_array);
	free(tsize_array);
	free(k_sortArray);
	cleanup_cpu_mem(pre_gpu_data, post_gpu_data, stream);
        cleanup_gpu_mem(skyloop_output, skyloop_other, stream);
	for(i=0; i<StreamNum; i++)
		cudaStreamDestroy(stream[i]);
	for(int i=0; i<StreamNum; i++)				// add count
                count += streamCount[i];
        cout<<"count = "<<count<<endl;
	cout<<"after_loop time = "<<gpu_time[0]<<endl;
	cout<<"after_loop preparation = "<<gpu_time[1]<<endl;
	cout<<"after_loop overall loop = "<<gpu_time[3]<<endl;
	cout<<"after_loop loop = "<<gpu_time[4]<<endl;
	cout<<"my after_loop loop = "<<gpu_time[9]<<endl;
	return count;
}

__host__ void push_work_into_gpu(struct pre_data *input_data, struct post_data *post_gpu_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, size_t *vtddim_array, int Lsky, int *pixel_array, int work_size, cudaStream_t *stream)
{
        for(int i=0; i<work_size; i++)// transfer the data from CPU to GPU
                cudaMemcpyAsync(skyloop_other[i].vtd_vTD_nr, input_data[i].other_data.vtd_vTD_nr, vtddim_array[i] * sizeof(float), cudaMemcpyHostToDevice, stream[i] );

        for(int i=0; i<work_size; i++)// call for gpu caculation
                kernel_skyloop<<<num_blocks, num_threads, shared_memory_usage, stream[i]>>>(skyloop_other[i].eTD, skyloop_other[i].vtd_vTD_nr, skyloop_other[i].rNRG, skyloop_other[0].FP_FX, skyloop_other[0].ml_mm, skyloop_other[0].V_tsize, skyloop_output[i].output, pixel_array[i]);
        for(int i=0; i<work_size; i++)// transfer the data back from GPU to CPU
                cudaMemcpyAsync(post_gpu_data[i].output.output, skyloop_output[i].output, OutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
}

__global__ void kernel_skyloop(float *eTD, float *vtd_vTD_nr, float *rNRG, double *FP_FX, short *ml_mm, size_t *V_tsize, float *gpu_output, int pixelcount)
{
	const int grid_size = blockDim.x * gridDim.x;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	struct STAT _stat;
	float *pe[NIFO];
	float *pa[NIFO];
	float *pA[NIFO];
	float *vtd[NIFO];
	float *vTD[NIFO];
	short *ml[NIFO];
	float *k_array;
	float *nr;
	short *mm;
	float stat;
	size_t V, V4, tsize;
	int Lsky = constLsky;
	int l;
	int k;
	int msk;
	int count = 0;
	int vDim;
	size_t v_ptr = MaxPixel;
	size_t rNRG_ptr = 0;
	
	_stat.stat = _stat.Lm = _stat.Em = _stat.Am = _stat.suball = _stat.EE = 0.;
	_stat.lm = _stat.Vm = -1;
	k_array = vtd_vTD_nr;	
	ml[0] = ml_mm;
	ml[1] = ml_mm + Lsky;
	ml[2] = ml_mm + 2*Lsky;
	ml[3] = ml_mm + 3*Lsky;
	mm = ml_mm + 4*Lsky;
	
	while(count<pixelcount)
	{
		k = k_array[count] - 1;
		V = V_tsize[k];
		tsize = V_tsize[k+constK];
		msk = V%4;	
		msk = (msk>0);
		V4 = V + msk*(4-V%4);
		vDim = V4*tsize;	

		vtd[0] = vtd_vTD_nr + v_ptr;
		vtd[1] = vtd_vTD_nr + vDim + v_ptr;
		vtd[2] = vtd_vTD_nr + 2*vDim + v_ptr;
		vtd[3] = vtd_vTD_nr + 3*vDim + v_ptr;
		vTD[0] = vtd_vTD_nr + NIFO*vDim + v_ptr;
		vTD[0] = vtd_vTD_nr + NIFO*vDim + vDim + v_ptr;
		vTD[0] = vtd_vTD_nr + NIFO*vDim + 2*vDim + v_ptr;
		vTD[0] = vtd_vTD_nr + NIFO*vDim + 3*vDim + v_ptr;
		
		// Get eTD
		for(l=tid; l<vDim; l+=grid_size)
		{
			eTD[l] = vtd[0][l]*vtd[0][l] + vTD[0][l]*vTD[0][l];
			eTD[l + vDim] = vtd[1][l]*vtd[1][l] + vTD[1][l]*vTD[1][l];
			eTD[l + 2*vDim] = vtd[2][l]*vtd[2][l] + vTD[2][l]*vTD[2][l];
			eTD[l + 3*vDim] = vtd[3][l]*vtd[3][l] + vTD[3][l]*vTD[3][l];
		}
		// Wait for all threads to update eTD
		__syncthreads();

		pe[0] = eTD;
		pe[1] = eTD + vDim;
		pe[2] = eTD + 2*vDim;
		pe[3] = eTD + 3*vDim;
			
		pe[0] = pe[0] + (tsize/2)*V4;
		pe[1] = pe[1] + (tsize/2)*V4;
		pe[2] = pe[2] + (tsize/2)*V4;
                pe[3] = pe[3] + (tsize/2)*V4;
		pa[0] = vtd[0] + (tsize/2)*V4;
		pa[1] = vtd[1] + (tsize/2)*V4;
		pa[2] = vtd[2] + (tsize/2)*V4;
		pa[3] = vtd[3] + (tsize/2)*V4;
		pA[0] = vTD[0] + (tsize/2)*V4;
		pA[1] = vTD[1] + (tsize/2)*V4;
		pa[2] = vTD[2] + (tsize/2)*V4;
		pA[3] = vTD[3] + (tsize/2)*V4;

		for(l=tid; l<Lsky; l+=grid_size)
		{
			if(!mm[l])	continue;
                        pe[0] = pe[0] + ml[0][l] * (int)V4;
                        pe[1] = pe[1] + ml[1][l] * (int)V4;
                        pe[2] = pe[2] + ml[2][l] * (int)V4;
                        pe[3] = pe[3] + ml[3][l] * (int)V4;
	
			kernel_skyloop_calculate(ml, rNRG, pa, pA, pe[0], pe[1], pe[2], pe[3], V, V4, V4*Lsky, gpu_output, l, rNRG_ptr, &_stat);
		}

		kernel_store_result_to_tmp(tmp, tid, &_stat);
		//Wait for all threads to finish calculation
		__syncthreads();
		if(tid < 32)
		{
			kernel_store_stat(tmp, tid);	
			__syncthreads();
		}
		if(tid == 0)
		{
			stat = kernel_store_final_stat(tmp, gpu_output, output_ptr);
			if(stat > 0)
			{
				pe[0] = eTD + (tsize/2)*V4;
				pe[1] = eTD + vDim + (tsize/2)*V4;
				pe[2] = eTD + 2*vDim + (tsize/2)*V4;
				pe[3] = eTD + 3*vDim + (tsize/2)*V4;
                        	pe[0] = pe[0] + ml[0][l] * (int)V4;
	                        pe[1] = pe[1] + ml[1][l] * (int)V4;
        	                pe[2] = pe[2] + ml[2][l] * (int)V4;
                	        pe[3] = pe[3] + ml[3][l] * (int)V4;
				size_t v;
				for(v=0; v<V; v++)
					gpu_output[output_ptr+OutputSize+v] = PE_0[v] + PE_1[v] + PE_2[v] + PE_3[v];
			}
		}
		output_ptr += (OutputSize + V4); 
		count++;
	}
	
	return;
}
__inline__ __device__ void kernel_skyloop_calculate(short **ml, float *rNRG, float *nr, float *gpu_BB, float *gpu_bb, float *gpu_fp, float *gpu_fx, float *gpu_Fp, float *gpu_Fx, float **pa, float **pA, float *PE_0, float *PE_1, float *PE_2, float *PE_3, size_t V, size_t V4, size_t rEDim, float *gpu_output,  int l, size_t rNRG_ptr, struct STAT *_s)
{
        int msk;                                              // mask
        size_t v;                                  // indicate the pixel
	size_t j;
        size_t ptr;                                                // indicate the location 
        float pe[NIFO];
	float *v00[NIFO];
	float *v90[NIFO];
        float _Eo[4], _Es[4], _En[4];
        float En, Es, Eo, aa;
        int Mm;
        float rE;                                               // energy array rNRG.data 
        float pE;                                               // energy array pNRG.data
        int count;
	int m; 

        Mm = 0;                                                 // # of pixels above the threshold
        for(count=0; count<4; count++)
        {
                _Eo[count] = 0;
                _Es[count] = 0;
                _En[count] = 0;
        }

        count = 0;
        ptr = l*V4 + rNRG_ptr;
        for(v=0; v<V; v++)                                      // loop over selected pixels    
        {
                // *_rE = _sse_sum_ps(_pe);
                pe[0] = PE_0[v];
                pe[1] = PE_1[v];
                pe[2] = PE_2[v];
                pe[3] = PE_3[v];
                rE = pe[0] + pe[1] + pe[2] + pe[3];                                                             // get pixel energy
	        // E>En  0/1 mask
                msk = ( rE>=constEn );                                                                          // E>En  0/1 mask
                Mm += msk;                                                                                              // count pixels above threshold
                pE = rE * msk;                                                                                  // zero sub-threshold pixels
                _Eo[count] += pE;
                //Eo += pE;                                                                                             // network energy
                pE = kernel_minSNE_ps(pE, pe);                                          // subnetwork energy
                _Es[count] += pE;
                //Es += pE;                                                                                             // subnetwork energy
                msk = ( pE>=constEs );                                                                          // subnet energy > Es 0/1 mask
                rE *= msk;
                _En[count] += rE;
                // assign the value to the local memory
                count++;
                count = count%4;
        }

        En = _En[0] + _En[1] + _En[2] + _En[3];                 // Write back to output
        Eo = _Eo[0] + _Eo[1] + _Eo[2] + _Eo[3] + 0.01;
        Es = _Es[0] + _Es[1] + _Es[2] + _Es[3];
        Mm = Mm * 2 + 0.01;
        aa = Es*En/(Eo-Es);

        msk = ((aa-Mm)/(aa+Mm)<0.33);

	if(msk)	return;
	float *bb, *BB, *fp, *Fp, *fx, *Fx;
	// after skyloop
	v00[0] = pa[0] + ml[0][l] * (int)V4;
	v00[1] = pa[1] + ml[1][l] * (int)V4;
	v00[2] = pa[2] + ml[2][l] * (int)V4;
	v00[3] = pa[3] + ml[3][l] * (int)V4;
	v90[0] = pA[0] + ml[0][l] * (int)V4;
	v90[1] = pA[1] + ml[1][l] * (int)V4;
	v90[2] = pA[2] + ml[2][l] * (int)V4;
	v90[3] = pA[3] + ml[3][l] * (int)V4;
	// point to the memory
	bb = gpu_bb + num_blocks*num_threads*NIFO*V4max;	
	BB = gpu_BB + num_blocks*num_threads*NIFO*V4max;	
	fp = gpu_fp + num_blocks*num_threads*NIFO*V4max;
	Fp = gpu_Fp + num_blocks*num_threads*NIFO*V4max;
	fx = gpu_fx + num_blocks*num_threads*NIFO*V4max;
	Fx = gpu_Fx + num_blocks*num_threads*NIFO*V4max;
	
	for(j=0; j<V; j++)
	{
		bb[j*NIFO+0] = v00[0][j];
		bb[j*NIFO+1] = v00[1][j];
		bb[j*NIFO+2] = v00[2][j];
		bb[j*NIFO+3] = v00[3][j];
		BB[j*NIFO+0] = v90[0][j];
		BB[j*NIFO+1] = v90[1][j];
		BB[j*NIFO+2] = v90[2][j];
		BB[j*NIFO+3] = v90[3][j];
		
		kernel_cpf_(fp+j*NIFO, FP, l);
		kernel_cpf_(fx+j*NIFO, FX, l);
	}	
	m = 0; Ls=Ln=Eo=0;
	for(j=0; j<V; j++)
	{
		ee = kernel_sse_abs_ps(bb+NIFO*j, BB+NIFO*j);
		if(ee<constEn)	continue;
		kernel_sse_cpf_ps(bb+NIFO*m, bb+NIFO*j);
		kernel_sse_cpf_ps(BB+NIFO*m, BB+NIFO*j);
		kernel_sse_cpf_ps(Fp+NIFO*m, fp+NIFO*j);
		kernel_sse_cpf_ps(Fx+NIFO*m, fx+NIFO*j);
		kernel_sse_mul_ps(Fp+NIFO*m, nr+NIFO*j);
		kernel_sse_mul_ps(Fx+NIFO*m, nr+NIFO*j);
		m++;
		em = kernel_sse_maxE_ps(bb+j*NIFO, BB+j*NIFO);
		Ls+= ee-em;	Eo += ee;
		msk = (ee-em>consEs);
		Ln += msk*ee;
	}

	
	msk = m%4;
	msk = (msk>0);
	size_t m4 = m + msk*(4-m%4);
	_En[0] = _En[1] = _En[2] = _En[3] = 0;
	
	for(j=0; j<m4; j+=4)
	{
		kernel_sse_dpf4_ps(Fp+j*NIFO, Fx+j*NIFO, fp+j*NIFO, fx+j*NIFO);
		kernel_sse_like4_ps(fp+j*NIFO, fx+j*NIFO, bb+j*NIFO, BB+j*NIFO, _Es);
		_En[0] = _En[0] + _Es[0];
		_En[1] = _En[1] + _Es[1];
		_En[2] = _En[2] + _Es[2];
		_En[3] = _En[3] + _Es[3];
	}
	
	Lo = _En[0] + _En[1] + _En[2] + _En[3];
	AA = aa/(fabs(aa)+fabs(Eo-Lo)+2*m*(Eo-Ln)/Eo);
	ee = Ls*Eo/(Eo-Ls);
	em = fabs(Eo-Lo)+2*m;
	ee = ee/(ee+em);
	aa = (aa-m)/(aa+m);
	
	msk = (AA > _s->stat);
	_s->stat = _s->stat+AA - _s->stat*msk - AA*(1-msk);
	_s->Lm = _s->Lm+Lo - _s->Lm*msk - Lo*(1-msk);
	_s->Em = _s->Em+Eo - _s->Em*msk - Eo*(1-msk);
	_s->Am = _s->Am+aa - _s->Am*msk - aa*(1-msk);
	_s->lm = _s->lm+l - _s->lm*msk - l*(1-msk);
	_s->Vm = _s->Vm+m - _s->Vm*msk - m*(1-msk);
	_s->suball = _s->suball+e - _s->suball*msk - e*(1-msk);
	_s->EE = _s->EE+em - _s->EE*msk - em*(1-msk);
	
}
__inline__ __device__ void kernel_store_result_to_tmp(float *tmp, int tid, struct STAT *_s)
{
	tmp[tid*OutputSize] = _s->stat;
	tmp[tid*OutputSize+1] = _s->Lm;
	tmp[tid*OutputSize+2] = _s->Em;
	tmp[tid*OutputSize+3] = _s->Am;
	tmp[tid*OutputSize+4] = _s->suball;
	tmp[tid*OutputSize+5] = _s->EE;
	tmp[tid*OutputSize+6] = _s->lm;
	tmp[tid*OutputSize+7] = _s->Vm;
	return;
}
__device__ void kernel_store_stat(float *tmp, int tid)
{
	float max = 0;
	float temp;
	size_t l;
	bool msk;
	for(int i=0; i<num_blocks*num_threads, i+=32)
	{
		temp = tmp[i*OutputSize];
		msk = (temp>max);
		max = max+temp - max*msk - (1-msk)*temp; 
		l = l+i - l*msk - (1-msk)*i; 
	}
	tmp[tid*OutputSize] = tmp[l*OutputSize];
	tmp[tid*OutputSize+1] = tmp[l*OutputSize+1];
	tmp[tid*OutputSize+2] = tmp[l*OutputSize+2];
	tmp[tid*OutputSize+3] = tmp[l*OutputSize+3];
	tmp[tid*OutputSize+4] = tmp[l*OutputSize+4];
	tmp[tid*OutputSize+5] = tmp[l*OutputSize+5];
	tmp[tid*OutputSize+6] = tmp[l*OutputSize+6];
	tmp[tid*OutputSize+7] = tmp[l*OutputSize+7];
	return;
}
__device__ float kernel_store_final_stat(float *tmp, float *gpu_output, size_t output_ptr)
{
	float max = 0;
	float temp;
	size_t l;
	bool msk;
	for(int i=0; i<32; i++)
	{
		temp = tmp[i*OutputSize];
		msk = (temp>max);
		max = max+temp - max*msk - (1-msk)*temp; 
		l = l+i - l*msk - (1-msk)*i; 
	}
	gpu_output[output_ptr] = tmp[l*OutputSize];
	gpu_output[output_ptr+1] = tmp[l*OutputSize+1];
	gpu_output[output_ptr+2] = tmp[l*OutputSize+2];
	gpu_output[output_ptr+3] = tmp[l*OutputSize+3];
	gpu_output[output_ptr+4] = tmp[l*OutputSize+4];
	gpu_output[output_ptr+5] = tmp[l*OutputSize+5];
	gpu_output[output_ptr+6] = tmp[l*OutputSize+6];
	gpu_output[output_ptr+7] = tmp[l*OutputSize+7];
	return max;
}
	/*	for(j=0; j<V4*NIFO; j+=4)
	{
		ee = kernel_sse_abs_ps(v00, v90, V4, j);	
		msk = !(ee<constEn);
		m_list[j/4] = msk;
		m += msk;
		em = kernel_sse_maxE_ps(v00, v90, V4, j);	
		Ls += msk*(ee-em);	Eo += msk*ee;
		msk = (ee-em>Ls);
		Ln += msk*ee;
	}
	int I = -1;
	for(j=0; j<m4*NIFO; j+=16)
	{
		int count = 0;
		int i[NIFO] = {-1, -1, -1, -1};			// indicate the m_list
		int r[NIFO];					// indicate the row of v00 and v90
		do
		{
			I++;
			i[count] = I;
			count += m_list[I];
		}
		while(count!=NIFO)
		for(count = 0; count<NIFO; count++)
		{
			r[count] = i[count]*4/V4;
			i[count] = (i[count]*4 - r[count]*V4);		// indicate the location of v00[r] and v90[r]
		}
		kernel_sse_dpf4_ps(v00, v90, i, j, r);
	}*/
	
}
__inline__ __device__ float kernel_minSNE_ps(float pE, float *pe)
{
        float a, b, c, d;
        int ab, ac, ad, bc, bd, cd;
        float temp;
        int flag;

        a = pe[0];
        b = pe[1];
        c = pe[2];
        d = pe[3];
        ab = ( a>=b );                                                                                  // if a>=b, ab 1/0
        ac = ( a>=c );                                                                                  // if a>=c, ac 1/0
        ad = ( a>=d );                                                                                  // if a>=d, ad 1/0
        bc = ( b>=c );                                                                                  // if b>=c, bc 1/0
        bd = ( b>=d );                                                                                  // if b>=d, bd 1/0
        cd = ( c>=d );                                                                                  // if c>=d, cd 1/0

        temp = a+b+c+d - ab*ac*ad*a - (1-ab)*bc*bd*b - (1-ac)*(1-bc)*cd*c - (1-ad)*(1-bd)*(1-cd)*d;
        flag = ( temp>=pE );                                                                            // if temp>=pE, flag 1/0
        temp = temp + pE - flag*temp - (1-flag)*pE;
        return temp;
}
__inline__ __device__ void kernel_cpf_(float *a, double **p, size_t i)
{
	a[0] = p[0][i];
	a[1] = p[1][i];
	a[2] = p[2][i];
	a[3] = p[3][i];
	return;
}
__inline__ __device__ void kernel_sse_cpf_ps(float *a, float *p)
{
	a[0] = p[0];
	a[1] = p[1];
	a[2] = p[2];
	a[3] = p[3];
	return;
}
__inline__ __device__ void kernel_sse_mul_ps(float *a, float *b)
{
	a[0] = a[0]*b[0];
	a[1] = a[1]*b[1];
	a[2] = a[2]*b[2];
	a[3] = a[3]*b[3];
	return;
}
__inline__ __device__ float kernel_sse_abs_ps(float *bb, float *BB)
{
	float out;
	out = bb[0]*bb[0] + BB[0]*BB[0];
	out += bb[1]*bb[1] + BB[1]*BB[1];
	out += bb[2]*bb[2] + BB[2]*BB[2];
	out += bb[3]*bb[3] + BB[3]*BB[3];
	return out;
}
__inline__ __device__ float kernel_sse_maxE_ps(float *a, float *A)
{
	float out;
	float temp;
	bool flag;
	out = a[0]*a[0] + A[0]*A[0];
	temp = a[1]*a[1] + A[1]*A[1];
	flag = (temp>out);
	out = temp+out - (1-flag)*temp - flag*out;
	temp = a[2]*a[2] + A[2]*A[2];
	flag = (temp>out);
	out = temp+out - (1-flag)*temp - flag*out;
	temp = a[3]*a[3] + A[3]*A[3];
	flag = (temp>out);
	out = temp+out - (1-flag)*temp - flag*out;
	return out;
}
__inline__ __device__ void kernel_sse_dpf4_ps(float *Fp, float Fx, float fp, float fx)
{
	float _c[NIFO];					// cos
	float _s[NIFO];					// sin
	kernel_sse_ort4_ps(Fp, Fx, _s, _c);
	kernel_sse_rot4p_ps(Fp, _c, Fx, _s, fp);		// get fp = Fp*c+Fx*s
	kernel_sse_rot4m_ps(Fx, _c, Fp, _s, fx);		// get fx = Fx*c-Fp*s
} 
__inline__ __device__ void kernel_sse_ort4_ps(float *u, float *v, float *_s, float *_c)
{
	static const float sm[NIFO] = {0.f, 0.f, 0.f, 0.f};	
	static const float _o[NIFO] = {1.e-24, 1.e-24, 1.e-24, 1.e-24};
	float _n[NIFO], _m[NIFO], gI[NIFO], gR[NIFO], _p[NIFO], _q[NIFO];	
	float _out[NIFO];
	
	kernel_sse_dot4_ps(u, v, _out);
	gI[0] = _out[0]*2; gI[1] = _out[1]*2; gI[2] = _out[2]*2; gI[3] = _out[3]*2; 		// ~sin(2*psi) or 2*u*v
	kernel_sse_dot4_ps(u, u, _out);
	gR[0] = _out[0]; gR[1] = _out[1]; gR[2] = _out[2]; gR[3] = _out[3]; 
	kernel_sse_dot4_ps(v, v, _out);
	gR[0] -= _out[0]; gR[1] -= _out[1]; gR[2] -= _out[2]; gR[3] -= _out[3]; 		// u^2-v^2
	
	_p[0] = (gR[0]>0); _p[1] = (gR[1]>0); _p[2] = (gR[2]>0); _p[3] = (gR[3]>0);		// 1 if gR>0. or 0 if gR<0.
	
	_q[0] = 1-_p[0]; _q[1] = 1-_p[1]; _q[2] = 1-_p[2]; _q[3] = 1-_p[3]; 			// 0 if gR>0. or 1 if gR<0.
	
	_n[0] = sqrt(gI[0]*gI[0] + gR[0]*gR[0]); _n[1] = sqrt(gI[1]*gI[1] + gR[1]*gR[1]); _n[2] = sqrt(gI[2]*gI[2] + gR[2]*gR[2]); _n[3] = sqrt(gI[3]*gI[3] + gR[3]*gR[3]); // go

	gR[0] = !(sm[0]&&gR[0]) + (_n[0]+_o[0]); gR[1] = !(sm[1]&&gR[1]) + (_n[1]+_o[1]); gR[2] = !(sm[2]&&gR[2]) + (_n[2]+_o[2]); gR[3] = !(sm[3]&&gR[3]) + (_n[3]+_o[3]); 
	
	_n[0] = _n[0]*2 + _o[0]; _n[1] = _n[1]*2 + _o[1]; _n[2] = _n[2]*2 + _o[2]; _n[3] = _n[3]*2 + _o[3]; 	// 2*go + eps
	
	gI[0] = gI[0]/_n[0]; gI[1] = gI[1]/_n[1]; gI[2] = gI[2]/_n[2]; gI[3] = gI[3]/_n[3]; 	// sin(2*psi)

	_n[0] = sqrt(gR[0]-_n[0]); _n[1] = sqrt(gR[1]-_n[1]); _n[2] = sqrt(gR[2]-_n[2]); _n[3] = sqrt(gR[3]-_n[3]);	// sqrt((gc+|gR|)/(2gc+eps)

	_m[0] = (gI[0]>0); _m[1] = (gI[1]>0); _m[2] = (gI[2]>0); _m[3] = (gI[3]>0); 		// if gI>0. or 0 if gI<0.
	
	_m[0] = (_m[0]*2-1) * _n[0]; _m[1] = (_m[1]*2-1) * _n[1]; _m[2] = (_m[2]*2-1) * _n[2]; _m[3] = (_m[3]*2-1) * _n[3];	// _n if gI>0 or -_n if gI<
	// sin(psi)
	_s[0] = _q[0]*_m[0] + _p[0]*(_gI[0]/_n[0]); 
	_s[1] = _q[1]*_m[1] + _p[1]*(_gI[1]/_n[1]); 	
	_s[2] = _q[2]*_m[2] + _p[2]*(_gI[2]/_n[2]); 
	_s[3] = _q[3]*_m[3] + _p[3]*(_gI[3]/_n[3]); 
	
	gI[0] = !(sm[0]&&gI[0]);  gI[1] = !(sm[1]&&gI[1]); gI[2] = !(sm[2]&&gI[2]); gI[3] = !(sm[3]&&gI[3]); 	// |gI|
	// cos(psi)
	_c[0] = _p[0]*_n[0] + _q[0]*(gI[0]/_n[0]); 
	_c[1] = _p[1]*_n[1] + _q[1]*(gI[1]/_n[1]); 
	_c[2] = _p[2]*_n[2] + _q[2]*(gI[2]/_n[2]); 
	_c[3] = _p[3]*_n[3] + _q[3]*(gI[3]/_n[3]); 
	return;
}
__inline__ __device__ void kernel_sse_dot4_ps(float *u, float *v, float *out)
{
	out[0] = u[0]*v[0];
	out[0] += u[1]*v[1];
	out[0] += u[2]*v[2];
	out[0] += u[3]*v[3];

	out[1] = u[4]*v[4];
	out[1] += u[5]*v[5];
	out[1] += u[6]*v[6];
	out[1] += u[7]*v[7];

	out[2] = u[8]*v[8];
	out[2] = u[9]*v[9];
	out[2] += u[10]*v[10];
	out[2] += u[11]*v[11];

	out[3] += u[12]*v[12];
	out[3] += u[13]*v[13];
	out[3] += u[14]*v[14];
	out[3] += u[15]*v[15];
	return;
}
__inline__ __device__ void kernel_sse_rot4p_ps(float *Fp, float *_c float *Fx, float *_s, float *fp)
{
	fp[0] = Fp[0]*_c[0] + Fx[0]*_s[0];	
	fp[1] = Fp[1]*_c[0] + Fx[1]*_s[0];	
	fp[2] = Fp[2]*_c[0] + Fx[2]*_s[0];	
	fp[3] = Fp[3]*_c[0] + Fx[3]*_s[0];	
	
	fp[4] = Fp[4]*_c[1] + Fx[4]*_s[1];	
	fp[5] = Fp[5]*_c[1] + Fx[5]*_s[1];	
	fp[6] = Fp[6]*_c[1] + Fx[6]*_s[1];	
	fp[7] = Fp[7]*_c[1] + Fx[7]*_s[1];	

	fp[8] = Fp[8]*_c[2] + Fx[8]*_s[2];	
	fp[9] = Fp[9]*_c[2] + Fx[9]*_s[2];	
	fp[10] = Fp[10]*_c[2] + Fx[10]*_s[2];	
	fp[11] = Fp[11]*_c[2] + Fx[11]*_s[2];	

	fp[12] = Fp[12]*_c[3] + Fx[12]*_s[3];	
	fp[13] = Fp[13]*_c[3] + Fx[13]*_s[3];	
	fp[14] = Fp[14]*_c[3] + Fx[14]*_s[3];	
	fp[15] = Fp[15]*_c[3] + Fx[15]*_s[3];	
	return;
}
__inline__ __device__ void kernel_sse_rot4m_ps(float *Fx, float *_c, float *Fp, float *_s, float *fx)
{
	fx[0] = Fx[0]*_c[0] - Fp[0]*_s[0];	
	fx[1] = Fx[1]*_c[0] - Fp[1]*_s[0];	
	fx[2] = Fx[2]*_c[0] - Fp[2]*_s[0];	
	fx[3] = Fx[3]*_c[0] - Fp[3]*_s[0];	
	
	fx[4] = Fx[4]*_c[1] - Fp[4]*_s[1];	
	fx[5] = Fx[5]*_c[1] - Fp[5]*_s[1];	
	fx[6] = Fx[6]*_c[1] - Fp[6]*_s[1];	
	fx[7] = Fx[7]*_c[1] - Fp[7]*_s[1];	

	fx[8] = Fx[8]*_c[2] - Fp[8]*_s[2];	
	fx[9] = Fx[9]*_c[2] - Fp[9]*_s[2];	
	fx[10] = Fx[10]*_c[2] - Fp[10]*_s[2];	
	fx[11] = Fx[11]*_c[2] - Fp[11]*_s[2];	

	fx[12] = Fx[12]*_c[3] - Fp[12]*_s[3];	
	fx[13] = Fx[13]*_c[3] - Fp[13]*_s[3];	
	fx[14] = Fx[14]*_c[3] - Fp[14]*_s[3];	
	fx[15] = Fx[15]*_c[3] - Fp[15]*_s[3];	
	return;
}
__inline__ __device__ void kernel_sse_like4_ps(float *fp, float *fx, float *bb, float *BB, float *_Es)
{
	float xp[NIFO];
	float XP[NIFO];
	float xx[NIFO];
	float XX[NIFO];
	float gp[NIFO];
	float gx[NIFO];
	float tmp[NIFO];

	kernel_sse_dot4_ps(fp, bb, xp);
	kernel_sse_dot4_ps(fp, BB, XP);
	kernel_sse_dot4_ps(fx, bb, xx);
	kernel_sse_dot4_ps(fx, BB, XX);

	kernel_sse_dot4_ps(fp, fp, tmp);
	gp[0] = tmp[0] + 1.e-12;
	gp[1] = tmp[1] + 1.e-12;	
	gp[2] = tmp[2] + 1.e-12;	
	gp[3] = tmp[3] + 1.e-12;	

	kernel_sse_dot4_ps(fx, fx, tmp);
	gx[0] = tmp[0] + 1.e-12;
	gx[1] = tmp[1] + 1.e-12;	
	gx[2] = tmp[2] + 1.e-12;	
	gx[3] = tmp[3] + 1.e-12;	
	
	xp[0] = xp[0]*xp[0] + XP[0]*XP[0];
	xp[1] = xp[1]*xp[1] + XP[1]*XP[1];
	xp[2] = xp[2]*xp[2] + XP[2]*XP[2];
	xp[3] = xp[3]*xp[3] + XP[3]*XP[3];
	
	xx[0] = xx[0]*xx[0] + XX[0]*XX[0];
	xx[1] = xx[1]*xx[1] + XX[1]*XX[1];
	xx[2] = xx[2]*xx[2] + XX[2]*XX[2];
	xx[3] = xx[3]*xx[3] + XX[3]*XX[3];
	
	_Es[0] = xp[0]/gp[0] + xx[0]/gx[0];
	_Es[1] = xp[1]/gp[1] + xx[1]/gx[1];
	_Es[2] = xp[2]/gp[2] + xx[2]/gx[2];
	_Es[3] = xp[3]/gp[3] + xx[3]/gx[3];
	return;
}
/*__inline__ __device__ float kernel_sse_abs_ps(float **v00, float **v90, size_t V4, size_t j)
{
	float out;
	int i;
	i = (j*4)/V4;
	out = v00[i][4*j]*v00[i][4*j] + v90[i][4*j]*v90[i][4*j];
	out += v00[i][4*j+1]*v00[i][4*j+1] + v90[i][4*j+1]*v90[i][4*j+1];
	out += v00[i][4*j+2]*v00[i][4*j+2] + v90[i][4*j+2]*v90[i][4*j+2];
	out += v00[i][4*j+3]*v00[i][4*j+3] + v90[i][4*j+3]*v90[i][4*j+3];
	return out;
}*/
/*__inline__ __device__ float kernel_sse_maxE_ps(float **v00, float **v90, size_t V4, size_t j)
{
	float out;
	float temp;
	int flag;
	int i;
	i = (j*4)/V4;
	out = v00[i][4*j]*v00[i][4*j] + v90[i][4*j]*v90[i][4*j];
	temp = v00[i][4*j+1]*v00[i][4*j+1] + v90[i][4*j+1]*v90[i][4*j+1];
	flag = (temp>out);
	out = temp+out - (1-flag)*temp - flag*out;
	temp = v00[i][4*j+2]*v00[i][4*j+2] + v90[i][4*j+2]*v90[i][4*j+2];
	flag = (temp>out);
	out = temp+out - (1-flag)*temp - flag*out;
	temp = v00[i][4*j+3]*v00[i][4*j+3] + v90[i][4*j+3]*v90[i][4*j+3];
	flag = (temp>out);
	out = temp+out - (1-flag)*temp - flag*out;
	return out;
}*/
/*__inline__ __device__ void kernel_sse_dpf4_ps(float **v00, float **v90, int *i, int j, int *r)
{
//	transformation to DPF for 4 consecutive pixels
//	rotate vectors Fp and Fx into DPF: fp and fx
	float _s[NIFO];		// sin
	float _c[NIFO];		// cos 
	kernel_sse_ort4_ps(v00, v90, _s, _c, i, r);
	kernel_sse_rot4p_ps(v00, _c, v90, _s, i, j, r);
	
}*/
/*__inline__ __device__ void kernel_sse_ort4_ps(float **v00, float **v90, float *_s, float *_c, int *i, int *r)
{
	static const float sm[NIFO] = {0.f, 0.f, 0.f, 0.f};	
	static const float _o[NIFO] = {1.e-24, 1.e-24, 1.e-24, 1.e-24};
	float _n[NIFO], _m[NIFO], gI[NIFO], gR[NIFO], _p[NIFO], _q[NIFO];	
	float _out[NIFO];
	int msk;
	kernel_sse_dot4_ps(v00, v90, i, r, _out);
	gI[0] = _out[0]*2; gI[1] = _out[1]*2; gI[2] = _out[2]*2; gI[3] = _out[3]*2; 		// ~sin(2*psi) or 2*u*v

	kernel_sse_dot4_ps(v00, v00, i, r, _out);
	gR[0] = _out[0]; gR[1] = _out[1]; gR[2] = _out[2]; gR[3] = _out[3]; 
	kernel_sse_dot4_ps(v90, v90, i, r, _out);
	gR[0] -= _out[0]; gR[1] -= _out[1]; gR[2] -= _out[2]; gR[3] -= _out[3]; 		// u^2-v^2
	
	_p[0] = (gR[0]>0); _p[1] = (gR[1]>0); _p[2] = (gR[2]>0); _p[3] = (gR[3]>0);		// 1 if gR>0. or 0 if gR<0.
	
	_q[0] = 1-_p[0]; _q[1] = 1-_p[1]; _q[2] = 1-_p[2]; _q[3] = 1-_p[3]; 			// 0 if gR>0. or 1 if gR<0.
	
	_n[0] = sqrt(gI[0]*gI[0] + gR[0]*gR[0]); _n[1] = sqrt(gI[1]*gI[1] + gR[1]*gR[1]); _n[2] = sqrt(gI[2]*gI[2] + gR[2]*gR[2]); _n[3] = sqrt(gI[3]*gI[3] + gR[3]*gR[3]); // go

	gR[0] = !(sm[0]&&gR[0]) + (_n[0]+_o[0]); gR[1] = !(sm[1]&&gR[1]) + (_n[1]+_o[1]); gR[2] = !(sm[2]&&gR[2]) + (_n[2]+_o[2]); gR[3] = !(sm[3]&&gR[3]) + (_n[3]+_o[3]); 
	
	_n[0] = _n[0]*2 + _o[0]; _n[1] = _n[1]*2 + _o[1]; _n[2] = _n[2]*2 + _o[2]; _n[3] = _n[3]*2 + _o[3]; 	// 2*go + eps
	
	gI[0] = gI[0]/_n[0]; gI[1] = gI[1]/_n[1]; gI[2] = gI[2]/_n[2]; gI[3] = gI[3]/_n[3]; 	// sin(2*psi)

	_n[0] = sqrt(gR[0]-_n[0]); _n[1] = sqrt(gR[1]-_n[1]); _n[2] = sqrt(gR[2]-_n[2]); _n[3] = sqrt(gR[3]-_n[3]);	// sqrt((gc+|gR|)/(2gc+eps)

	_m[0] = (gI[0]>0); _m[1] = (gI[1]>0); _m[2] = (gI[2]>0); _m[3] = (gI[3]>0); 		// if gI>0. or 0 if gI<0.
	
	_m[0] = (_m[0]*2-1) * _n[0]; _m[1] = (_m[1]*2-1) * _n[1]; _m[2] = (_m[2]*2-1) * _n[2]; _m[3] = (_m[3]*2-1) * _n[3];	// _n if gI>0 or -_n if gI<
	// sin(psi)
	_s[0] = _q[0]*_m[0] + _p[0]*(_gI[0]/_n[0]); 
	_s[1] = _q[1]*_m[1] + _p[1]*(_gI[1]/_n[1]); 	
	_s[2] = _q[2]*_m[2] + _p[2]*(_gI[2]/_n[2]); 
	_s[3] = _q[3]*_m[3] + _p[3]*(_gI[3]/_n[3]); 
	
	gI[0] = !(sm[0]&&gI[0]);  gI[1] = !(sm[1]&&gI[1]); gI[2] = !(sm[2]&&gI[2]); gI[3] = !(sm[3]&&gI[3]); 	// |gI|
	// cos(psi)
	_c[0] = _p[0]*_n[0] + _q[0]*(gI[0]/_n[0]); 
	_c[1] = _p[1]*_n[1] + _q[1]*(gI[1]/_n[1]); 
	_c[2] = _p[2]*_n[2] + _q[2]*(gI[2]/_n[2]); 
	_c[3] = _p[3]*_n[3] + _q[3]*(gI[3]/_n[3]); 

	return;
}*/
/*__inline__ __device__ void kernel_sse_dot4_ps(float **v00, float **v90, int *i, int *r, float *_out)
{
	int row, col;
	row = r[0];
	col = i[0];
	_out[0] = v00[row][col]*v00[row][col] + v90[row][col]*v90[row][col];
	_out[0] += v00[row][col+1]*v00[row][col+1] + v90[row][col+1]*v90[row][col+1];
	_out[0] += v00[row][col+2]*v00[row][col+2] + v90[row][col+2]*v90[row][col+2];
	_out[0] += v00[row][col+3]*v00[row][col+3] + v90[row][col+3]*v90[row][col+3];
	row = r[1];
	col = i[1];
	_out[1] = v00[row][col]*v00[row][col] + v90[row][col]*v90[row][col];
	_out[1] += v00[row][col+1]*v00[row][col+1] + v90[row][col+1]*v90[row][col+1];
	_out[1] += v00[row][col+2]*v00[row][col+2] + v90[row][col+2]*v90[row][col+2];
	_out[1] += v00[row][col+3]*v00[row][col+3] + v90[row][col+3]*v90[row][col+3];
	row = r[2];
	col = i[2];
	_out[2] = v00[row][col]*v00[row][col] + v90[row][col]*v90[row][col];
	_out[2] += v00[row][col+1]*v00[row][col+1] + v90[row][col+1]*v90[row][col+1];
	_out[2] += v00[row][col+2]*v00[row][col+2] + v90[row][col+2]*v90[row][col+2];
	_out[2] += v00[row][col+3]*v00[row][col+3] + v90[row][col+3]*v90[row][col+3];
	row = r[3];
	col = i[3];
	_out[3] = v00[row][col]*v00[row][col] + v90[row][col]*v90[row][col];
	_out[3] += v00[row][col+1]*v00[row][col+1] + v90[row][col+1]*v90[row][col+1];
	_out[3] += v00[row][col+2]*v00[row][col+2] + v90[row][col+2]*v90[row][col+2];
	_out[3] += v00[row][col+3]*v00[row][col+3] + v90[row][col+3]*v90[row][col+3];
 
}*/
void MyCallback(struct post_data *post_gpu_data)
{
	double Clock[CLOCK_SIZE];
//
	int Lsky = gpu_Lsky;
	int k;
	size_t V4;
	int pixelcount=0;
	int streamNum;
	size_t output_ptr = 0;
	for(int i=0; i<StreamNum; i++)
	{
		k = post_gpu_data[i].other_data.k[pixelcount] - 1;
		
		while(k != -1)
		{
			V4 = post_gpu_data[i].other_data.V4[pixelcount];
	        	streamNum = post_gpu_data[i].other_data.stream;

			Clock[0] = clock();
			after_skyloop((void*)&post_gpu_data[i], gpu_net, gpu_hist, pwc, FP, FX, pa[streamNum][pixelcount], pA[streamNum][pixelcount], pixelcount, output_ptr, Lsky, gpu_time, streamCount);
			Clock[1] = clock();
			gpu_time[9] += (double)(Clock[1]-Clock[0])/CLOCKS_PER_SEC;
		
			output_ptr = output_ptr + V4*Lsky + Lsky;
			pixelcount++;
			if(pixelcount<MaxPixel)
				k = post_gpu_data[i].other_data.k[pixelcount] - 1;
			else 
				break;
		}
	}
//	fclose(fpt);
}
	

void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void *post_gpu_data)
{
//	FILE *fpt = fopen("./debug_files/skyloop_output", "a");
//debug
	double Clock[CLOCK_SIZE];
//
	int Lsky = gpu_Lsky;
	int k;
	size_t V4;
	int pixelcount=0;
	int streamNum;
	size_t output_ptr = 0;
	//cout<<"Callback"<<endl;
	k = ((post_data*)post_gpu_data)->other_data.k[pixelcount] - 1;
	
	while(k != -1)
	{
	//	cout<<"k = "<<k<<endl;
		V4 = ((post_data*)post_gpu_data)->other_data.V4[pixelcount];
        	streamNum = ((post_data*)post_gpu_data)->other_data.stream;

		Clock[0] = clock();
		after_skyloop(post_gpu_data, gpu_net, gpu_hist, pwc, FP, FX, pa[streamNum][pixelcount], pA[streamNum][pixelcount], pixelcount, output_ptr, Lsky, gpu_time, streamCount);
		Clock[1] = clock();
		gpu_time[0] += (double)(Clock[1]-Clock[0])/CLOCKS_PER_SEC;
		
		output_ptr = output_ptr + V4*Lsky + Lsky;
		pixelcount++;
		if(pixelcount<MaxPixel)
			k = ((post_data*)post_gpu_data)->other_data.k[pixelcount] - 1;
		else 
			break;
	}
//	fclose(fpt);
}
/*	after_skyloop:
	vint = &(pwc->cList[id-1]);
	pwc->sCuts[id-1] = -1;
	pwc->cData[id-1].likenet = Lm;
	pwc->cData[id-1].energy = Em;
	pwc->cData[id-1].theta = gpu_net->nLikelihood.getTheta(lm);
	pwc->cData[id-1].phi = gpu_net->nLikelihood.getPhi(lm);
	pwc->cData[id-1].skyIndex = lm;
	rHo = 0.;
	if(mra)
	{
		submra = Ls*Eo/(Eo-Ls);				// MRA subnet statistic
		submra /= fabs(submra) + fabs(Eo-Lo) + 2*(m+6);		// MRA subnet coefficient	
		To = 0;
		pwc->p_Ind[id-1].push_back(lm);
		for(int j=0; j<vint->size(); j++)
		{
			pix = pwc->getPixel(id,j);
                        pix->theta = gpu_net->nLikelihood.getTheta(lm);
			pix->phi   = gpu_net->nLikelihood.getPhi(lm);
			To += pix->time/pix->rate/pix->layers;
			if(j==0&&mra) pix->ellipticity = submra;    // subnet MRA propagated to L-stage
			if(j==0&&mra) pix->polarisation = fabs(Eo-Lo)+2*(m+6);	// submra NULL propagated to L-stage
			if(j==1&&mra) pix->ellipticity = suball;   // subnet all-sky propagated to L-stage
			if(j==1&&mra) pix->polarisation = EE;      // suball NULL propagated to L-stage
		}
		To /= vint->size();
		rHo = sqrt(Lo*Lo/(Eo+2*m)/nIFO);	// estimator of coherent amplitude
	}
	
	if(gpu_hist && rHo>gpu_net->netRHO)
		for(int j=0; j<vint->size(); j++)
			gpu_hist->Fill(suball, submra);
	
	if(fmin(suball, submra)>TH && rHo>gpu_net->netRHO)
        {
                count += vint->size();
                if(gpu_hist)
                {
                	printf("lag|id %3d|%3d rho=%5.2f To=%5.1f stat: %5.3f|%5.3f|%5.3f ",
                	int(lag),int(id),rHo,To,suball,submra,stat);
            		printf("E: %6.1f|%6.1f L: %6.1f|%6.1f|%6.1f pix: %4d|%4d|%3d|%2d \n",
                   	Em,Eo,Lm,Lo,Ls,int(vint->size()),int(V),Vm,int(m));
                }
        }
	else
		pwc->sCuts[id-1] = 1;
// clean time delay data
	V = vint->size();
	for(int j=0; j<V; j++)
	{
		pix = pwc->getPixel(id,j);
		pix->core = true;
		if(pix->tdAmp.size())
			pix->clean();
	}
	streamCount[stream] += count;
	//cout<<"4"<<endl;
	return;
	
}*/

void QuickSort(size_t *V_array, int *k_array, int p, int r)
{
        int q;
        if(p<r)
        {
                q = Partition(V_array, k_array, p, r);
                QuickSort(V_array, k_array, p, q-1);
                QuickSort(V_array, k_array, q+1, r);
        }
}
int Partition(size_t *V_array, int *k_array, int p, int r)
{
        int x, i, j;
        int temp;
        x = V_array[k_array[r]];
        i = p-1;
        for(j = p; j<r; j++)
        {
                if(V_array[k_array[j]]<=x)
                {
                        i = i + 1;
                        temp = k_array[i];
                        k_array[i] = k_array[j];
                        k_array[j] = temp;
                }
        }
        temp = k_array[i+1];
        k_array[i+1] = k_array[r];
        k_array[r] = temp;
        i++;
        return i;
}
void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int vDim, int V4max, int Lsky, size_t K)// allocate locked memory on CPU 
{
        for(int i = 0; i<BufferNum; i++)
        {
                CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.vtd_vTD_nr), 2*NIFO*vDim*sizeof(float) + NIFO*V4max*sizeof(float) + MaxPixel*sizeof(float), cudaHostAllocMapped ) );
        }
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.FP_FX), 2 * NIFO * Lsky * sizeof(double), cudaHostAllocMapped ) );
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.ml_mm), (1 + NIFO) * Lsky * sizeof(short), cudaHostAllocMapped ) );
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.V_tsize), K * 2 * sizeof(size_t), cudaHostAllocMapped ) );
        for(int i = 0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.output), OutputSize*sizeof(float), cudaHostAllocMapped ) );
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
void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int vDim, int V4max, int Lsky, size_t K)// allocate the memory on GPU
{
        for(int i = 0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].vtd_vTD_nr), 2*NIFO*vDim*sizeof(float) + NIFO*V4max*sizeof(float) + MaxPixel*sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].eTD), NIFO * vDim * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].BB), num_blocks * num_threads * V4max * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].bb), num_blocks * num_threads * V4max * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].fp), num_blocks * num_threads * V4max * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].fx), num_blocks * num_threads * V4max * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].Fp), num_blocks * num_threads * V4max * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].Fx), num_blocks * num_threads * V4max * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_output[i].output), OutputSize*sizeof(float) + V4max*sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].tmp), num_blocks*num_threads*OutputSize*sizeof(float) ) );
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
                CUDA_CHECK(cudaFree(skyloop_other[i].eTD) );
                CUDA_CHECK(cudaFree(skyloop_other[i].rNRG) );
                CUDA_CHECK(cudaFree(skyloop_output[i].output) );
        }
        CUDA_CHECK(cudaFree(skyloop_other[0].FP_FX) );
        CUDA_CHECK(cudaFree(skyloop_other[0].ml_mm) );
        CUDA_CHECK(cudaFree(skyloop_other[0].V_tsize) );
        return;
}


