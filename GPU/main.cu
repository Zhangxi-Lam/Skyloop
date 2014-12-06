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
#define VMAX 300
#define gpu_nIFO 3 
#define HEAD_SIZE 32
#define GRID_SIZE 4096

//inline int _sse_MRA_ps(network *net, float *amp, float *AMP, float Eo, int K);
size_t cc = 0;
network *gpu_net;
TH2F *gpu_hist;
netcluster *pwc;
int gpu_Lsky;
double *FP[gpu_nIFO];
double *FX[gpu_nIFO];
float *pa[StreamNum][MaxPixel][gpu_nIFO];
float *pA[StreamNum][MaxPixel][gpu_nIFO];
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
	
	cout<<"nIFO = "<<nIFO<<endl;
	cout<<"NIFO = "<<NIFO<<endl;
	cout<<"New"<<endl;

	float En = 2*net->acor*net->acor*nIFO;  // network energy threshold in the sky loop
        float Es = 2*net->e2or;                 // subnet energy threshold in the sky loop
        float TH = fabs(snc);                   // sub network threshold

        int l;
        float aa, AA;
        size_t i, j, k, V, id, K;
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
        size_t Vmax = 0;                       // store the maximum of V
        size_t Tmax = 0;                        // store the maximum of tsize
        size_t *V_array, *tsize_array;
        int *k_sortArray;
        int kcount = 0;                         // store the k that is not rejected/processed
	bool CombineFinish = false;		
        int v_ptr;                          // indicate vtd's and vTD's location
        int etd_ptr;                          // indicate the eTD's location
        size_t vtddim_array[StreamNum];
	size_t etddim_array[StreamNum];
        size_t alloced_V_array[StreamNum];
        int pixel_array[StreamNum];
        int pixelCount;                         // indicate the pixel number of each stream
	int alloced_V;
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
                V_array[k] = V;
                tsize_array[k] = tsize;
                k_sortArray[kcount] = k;
                kcount++;
                if( tsize > Tmax )
                        Tmax = tsize;
                if( V > Vmax )
                        Vmax = V;
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

        vDim = Tmax * Vmax;
	nr_size = gpu_nIFO* Vmax;
	v_size = vDim * gpu_nIFO * 2 + MaxPixel + nr_size;
        for(int i=0; i<StreamNum; i++)
                streamCount[i] = 0;
	// allocate the memory on cpu and gpu
	cudaFuncSetCacheConfig(kernel_skyloop, cudaFuncCachePreferL1);
	allocate_cpu_mem(pre_gpu_data, post_gpu_data, vDim, Vmax, Lsky, K);
	allocate_gpu_mem(skyloop_output, skyloop_other, vDim, Vmax, Lsky, K);
	
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
                        for(int j=0; j<gpu_nIFO; j++)
                        {
                                int ptr;
                                ptr = j*Lsky;
                                post_gpu_data[i].other_data.ml_mm[ptr + l] = ml[j][l];
				if(i==0)
				{
					pre_gpu_data[0].other_data.FP_FX[ptr + l] = FP[j][l];
					pre_gpu_data[0].other_data.FP_FX[gpu_nIFO*Lsky + ptr + l] = FX[j][l];
					pre_gpu_data[0].other_data.ml_mm[ptr + l] = ml[j][l];
				}
                        }
                        post_gpu_data[i].other_data.ml_mm[gpu_nIFO*Lsky + l] = mm[l];
			if(i==0)
				pre_gpu_data[0].other_data.ml_mm[gpu_nIFO*Lsky + l] = mm[l];
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
                        post_gpu_data[j].other_data.tsize[i] = 0;	
		}
	for(int k=0; k<K; k++)
	{
		pre_gpu_data[0].other_data.V_tsize[k] = V_array[k];
		pre_gpu_data[0].other_data.V_tsize[k + K] = tsize_array[k];	
	}
	cudaMemcpyAsync(skyloop_other[0].FP_FX, pre_gpu_data[0].other_data.FP_FX, 2 * gpu_nIFO* Lsky * sizeof(double), cudaMemcpyHostToDevice, stream[0] );
	cudaMemcpyAsync(skyloop_other[0].ml_mm, pre_gpu_data[0].other_data.ml_mm, (1 + gpu_nIFO) * Lsky * sizeof(short), cudaMemcpyHostToDevice, stream[0] );
	cudaMemcpyAsync(skyloop_other[0].V_tsize, pre_gpu_data[0].other_data.V_tsize, K * 2 * sizeof(size_t), cudaMemcpyHostToDevice, stream[0] );
//++++++++++++++++++++++++++++++++
// loop over cluster
//++++++++++++++++++++++++++++++++

	QuickSort(V_array, k_sortArray, 0, kcount-1);
	cid = pwc->get((char*)"ID", 0,'S',0);		// get cluster ID
	K = cid.size();
	v_ptr = HEAD_SIZE;
	etd_ptr = 0;
	pixelCount = 0;
	alloced_V = 0;
	for(int z=0; z<kcount;)			// loop over unskiped clusters
	{
		while(!CombineFinish && z<kcount)
		{
		k = k_sortArray[z];
		V = V_array[k];
		tsize = tsize_array[k];
		vtddim = V * tsize;
		nr_size = gpu_nIFO * V;
		if( (v_ptr+2*gpu_nIFO*vtddim+nr_size)<=v_size && ((pixelCount+1)*OutputSize+alloced_V+V)<=(OutputSize+Vmax) )
		{
			id = size_t(cid.data[k]+0.1);
                        pI = net->wdmMRA.getXTalk(pwc, id);
			alloced_V +=V;
			for(j=0; j<V; j++)
			{
				pix = pwc->getPixel(id,pI[j]);
				double rms = 0.;
				for(i=0; i<nIFO; i++)
				{
					xx[i] = 1./pix->data[i].noiserms;
					rms += xx[i]*xx[i];
				}
				
				for(i=0; i<gpu_nIFO; i++)
				{
				//for(i=0; i<nIFO; i++)
					if(i<nIFO)
                                	{
						pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[v_ptr + vtddim*gpu_nIFO*2 + j*gpu_nIFO+i] = (float)xx[i]/sqrt(rms);	// normalized 1/rms
	                                	for( l=0; l<tsize; l++)
	                                       	{
        		               			aa = pix->tdAmp[i].data[l];             // copy TD 00 data 
                		                        AA = pix->tdAmp[i].data[l+tsize];       // copy TD 90 data 
							pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[i*vtddim + l*V+j + v_ptr] = aa;
							pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[gpu_nIFO*vtddim + i*vtddim + l*V+j + v_ptr] = AA;
							pre_gpu_data[alloced_gpu].other_data.eTD[i*vtddim + l*V+j + etd_ptr] = aa*aa + AA*AA;
						}
                                	}
					else
					{
						pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[v_ptr + vtddim*gpu_nIFO*2 + j*gpu_nIFO+i] = (float)xx[i]/sqrt(rms);	// normalized 1/rms
	                                	for( l=0; l<tsize; l++)
	                                       	{
							pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[i*vtddim + l*V+j + v_ptr] = 0;
							pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[gpu_nIFO*vtddim + i*vtddim + l*V+j + v_ptr] = 0;
							pre_gpu_data[alloced_gpu].other_data.eTD[i*vtddim + l*V+j + etd_ptr] = 0;
						}
					}
				}	
			}
/*			FILE *fpt = fopen("./new_debug/my_k151vtd", "a");
			FILE *fpt1 = fopen("./new_debug/my_k151vTD", "a");
			if(k == 151)
			{
				
				for(int l=0; l<vtddim; l++)
				{
					fprintf(fpt, "l = %d vtd[0] = %f vtd[1] = %f vtd[2] = %f\n", l, pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[l + v_ptr], pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[vtddim + l + v_ptr], pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[2*vtddim + l + v_ptr]);
					fprintf(fpt1, "l = %d vTD[0] = %f vTD[1] = %f vTD[2] = %f\n", l, pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[3*vtddim+l + v_ptr], pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[3*vtddim + vtddim + l + v_ptr], pre_gpu_data[alloced_gpu].other_data.vtd_vTD_nr[3*vtddim + 2*vtddim + l + v_ptr]);
				}
				cout<<"finish"<<endl;
				
			}	
			fclose(fpt);
			fclose(fpt1);*/
			i = alloced_gpu;
			v_ptr += 2*gpu_nIFO*vtddim + nr_size;	
			etd_ptr += gpu_nIFO*vtddim;
			pre_gpu_data[i].other_data.vtd_vTD_nr[pixelCount] = k+1;
                        post_gpu_data[i].other_data.k[pixelCount] = k+1;
                        post_gpu_data[i].other_data.V[pixelCount] = V;
                        post_gpu_data[i].other_data.tsize[pixelCount] = tsize;
                        post_gpu_data[i].other_data.id[pixelCount] = id;
                        pixelCount++;
			z++;
			//cout<<"input k = "<<k<<" V = "<<V<<endl;
			//cout<<"z = "<<z<<" kcount = "<<kcount<<endl;
			if(pixelCount >= MaxPixel)
				CombineFinish = true;
		}
		else
			CombineFinish = true;
		}
		i = alloced_gpu;
		post_gpu_data[i].other_data.stream = alloced_gpu;
		vtddim_array[i] = v_ptr;
		etddim_array[i] = etd_ptr;
		alloced_V_array[i] = alloced_V;
		pixel_array[i] = pixelCount;
		alloced_gpu++;
//++++++++++++++++++++++++++++++++
// assign the data 
//++++++++++++++++++++++++++++++++
		if(alloced_gpu == StreamNum)
		{
			push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, vtddim_array, etddim_array, alloced_V_array, Lsky, pixel_array, StreamNum, stream);
			for(int i=0; i<StreamNum; i++)
				CUDA_CHECK(cudaStreamSynchronize(stream[i]));
			//MyCallback(post_gpu_data);
			//clear
			alloced_gpu = 0;
			for(int j=0; j<StreamNum; j++)
				for(int i=0; i<pixel_array[j]; i++)
				{
					post_gpu_data[j].other_data.k[i] = 0;
                                        post_gpu_data[j].other_data.V[i] = 0;
                                        post_gpu_data[j].other_data.tsize[i] = 0;	
				}
		}
		// clear
		v_ptr = MaxPixel;
		etd_ptr = 0;
		pixelCount = 0;
		CombineFinish = false;
		alloced_V = 0;
	}
	if(alloced_gpu != 0)
	{
		
		push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, vtddim_array, etddim_array, alloced_V_array, Lsky, pixel_array, StreamNum, stream);
		for(int i=0; i<StreamNum; i++)
			CUDA_CHECK(cudaStreamSynchronize(stream[i]));
		//MyCallback(post_gpu_data);
		alloced_gpu = 0;
	}		
	free(V_array);
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
	cout<<"cc = "<<cc<<endl;
	cc = 0;
	return count;
}

__host__ void push_work_into_gpu(struct pre_data *input_data, struct post_data *post_gpu_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, size_t *vtddim_array, size_t *etddim_array, size_t *alloced_V_array, int Lsky, int *pixel_array, int work_size, cudaStream_t *stream)
{
        for(int i=0; i<work_size; i++)// transfer the data from CPU to GPU
	{
                cudaMemcpyAsync(skyloop_other[i].vtd_vTD_nr, input_data[i].other_data.vtd_vTD_nr, vtddim_array[i] * sizeof(float), cudaMemcpyHostToDevice, stream[i] );
                cudaMemcpyAsync(skyloop_other[i].eTD, input_data[i].other_data.eTD, etddim_array[i] * sizeof(float), cudaMemcpyHostToDevice, stream[i] );
	}

        for(int i=0; i<work_size; i++)// call for gpu caculation
                kernel_skyloop<<<num_blocks, num_threads, shared_memory_usage, stream[i]>>>(skyloop_other[i].eTD, skyloop_other[i].vtd_vTD_nr,  skyloop_other[0].FP_FX, skyloop_other[0].ml_mm, skyloop_other[0].V_tsize, skyloop_other[i].BB, skyloop_other[i].bb, skyloop_other[i].fp, skyloop_other[i].fx, skyloop_other[i].Fp, skyloop_other[i].Fx, skyloop_other[i].tmp, skyloop_output[i].output, pixel_array[i]);
        for(int i=0; i<work_size; i++)// transfer the data back from GPU to CPU
        {       
		//cudaMemcpyAsync(post_gpu_data[i].output.output, skyloop_output[i].output, OutputSize*pixel_array[i]*sizeof(float) + alloced_V_array[i]*sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output.output, skyloop_output[i].output, MaxPixel*Lsky*sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
	}
	for(int i=0; i<work_size; i++)
		cudaStreamAddCallback(stream[i], MyCallback, (void*)&post_gpu_data[i], 0);
}

__global__ void kernel_skyloop(float *eTD, float *vtd_vTD_nr, double *FP_FX, short *ml_mm, size_t *V_tsize, float *gpu_BB, float *gpu_bb, float *gpu_fp, float *gpu_fx, float *gpu_Fp, float *gpu_Fx, float *gpu_tmp, float *gpu_output, int pixelcount)
{
	const int grid_size = blockDim.x * gridDim.x;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	struct STAT _stat;
	float *pe[gpu_nIFO];
	float *pa[gpu_nIFO];
	float *pA[gpu_nIFO];
	float *vtd[gpu_nIFO];
	float *vTD[gpu_nIFO];
	short *ml[gpu_nIFO];
	float *k_array;
	float *nr;
	short *mm;
	double *FP[gpu_nIFO];
	double *FX[gpu_nIFO];
	float stat;
	size_t V, tsize;
	int Lsky = constLsky;
	int l;
	int k;
	int msk;
	int count = 0;
	int vDim;
	size_t v_ptr = HEAD_SIZE;
	size_t etd_ptr = 0;
	size_t output_ptr = 0;
	
	_stat.stat = _stat.Lm = _stat.Em = _stat.Am = _stat.suball = _stat.EE = 0.;
	_stat.lm = _stat.Vm = -1;
	k_array = vtd_vTD_nr;	
	ml[0] = ml_mm;
	ml[1] = ml_mm + Lsky;
	ml[2] = ml_mm + 2*Lsky;
	mm = ml_mm + gpu_nIFO*Lsky;
	FP[0] = FP_FX;
	FP[1] = FP_FX + Lsky;
	FP[2] = FP_FX + 2*Lsky;
	FX[0] = FP_FX + gpu_nIFO*Lsky;
	FX[1] = FP_FX + (1+gpu_nIFO)*Lsky;
	FX[2] = FP_FX + (2+gpu_nIFO)*Lsky;

	while(count<pixelcount)
	{
		k = k_array[count] - 1;
		V = V_tsize[k];
		tsize = V_tsize[k+constK];
		msk = V%4;	
		msk = (msk>0);
		vDim = V*tsize;	

		vtd[0] = vtd_vTD_nr + v_ptr;
		vtd[1] = vtd_vTD_nr + vDim + v_ptr;
		vtd[2] = vtd_vTD_nr + 2*vDim + v_ptr;
		vTD[0] = vtd_vTD_nr + gpu_nIFO*vDim + v_ptr;
		vTD[1] = vtd_vTD_nr + gpu_nIFO*vDim + vDim + v_ptr;
		vTD[2] = vtd_vTD_nr + gpu_nIFO*vDim + 2*vDim + v_ptr;
		nr = vtd_vTD_nr + 2*gpu_nIFO*vDim + v_ptr;
		// Get eTD
		/*for(l=tid; l<vDim; l+=grid_size)
		{
			eTD[l] = vtd[0][l]*vtd[0][l] + vTD[0][l]*vTD[0][l];
			eTD[l + vDim] = vtd[1][l]*vtd[1][l] + vTD[1][l]*vTD[1][l];
			eTD[l + 2*vDim] = vtd[2][l]*vtd[2][l] + vTD[2][l]*vTD[2][l];
		}*/
		// Wait for all threads to update eTD
	//	__syncthreads();
		
/*		if(k==4)
		for(l=tid; l<vDim; l+=grid_size)
		{
			gpu_output[l] = eTD[l];
			gpu_output[l+vDim] = eTD[l+vDim];
			gpu_output[l+vDim*2] = eTD[l+2*vDim];
		}*/

			
		pa[0] = vtd[0] + (tsize/2)*V;
		pa[1] = vtd[1] + (tsize/2)*V;
		pa[2] = vtd[2] + (tsize/2)*V;
		pA[0] = vTD[0] + (tsize/2)*V;
		pA[1] = vTD[1] + (tsize/2)*V;
		pA[2] = vTD[2] + (tsize/2)*V;
	
		for(l=tid; l<Lsky; l+=grid_size)
		{
			if(!mm[l])	continue;
			pe[0] = eTD + etd_ptr + (tsize/2)*V;
			pe[1] = eTD + vDim + etd_ptr + (tsize/2)*V;
			pe[2] = eTD + 2*vDim + etd_ptr + (tsize/2)*V;
//			pe[0] = pe[0] + (tsize/2)*V;
//			pe[1] = pe[1] + (tsize/2)*V;
//			pe[2] = pe[2] + (tsize/2)*V;
                        pe[0] = pe[0] + ml[0][l] * (int)V;
                        pe[1] = pe[1] + ml[1][l] * (int)V;
                        pe[2] = pe[2] + ml[2][l] * (int)V;
			
			/*if(k == 4 && l == 256)
			{
				gpu_output[0] = pe[0][0];
				gpu_output[1] = pe[1][0];
				gpu_output[2] = pe[2][0];
				gpu_output[3] = pe[0][0] + pe[1][0] + pe[2][0];
				gpu_output[4] = ml[0][l];
				gpu_output[5] = pe[0][1];
				gpu_output[6] = pe[1][1];
				gpu_output[7] = pe[2][1];
				gpu_output[8] = pe[0][1] + pe[1][1] + pe[2][1];
				gpu_output[9] = ml[1][l];
				gpu_output[10] = pe[0][2];
				gpu_output[11] = pe[1][2];
				gpu_output[12] = pe[2][2];
				gpu_output[13] = pe[0][2] + pe[1][2] + pe[2][2];
				gpu_output[14] = ml[2][l];
			}*/
/*			if(k==4)
			{
				gpu_output[l] = pe[0][0];
				gpu_output[l+Lsky] = pe[1][0];
				gpu_output[l+Lsky*2] = pe[2][0];
				gpu_output[l+Lsky*3] = ml[0][l];
				gpu_output[l+Lsky*4] = ml[1][l];
			}*/
				
			kernel_skyloop_calculate(ml, nr, FP, FX, gpu_BB, gpu_bb, gpu_fp, gpu_fx, gpu_Fp, gpu_Fx, pa, pA, pe[0], pe[1], pe[2], V, gpu_output, l, &_stat, tid, k, output_ptr);
		}
	
//		kernel_store_result_to_tmp(gpu_tmp, tid, &_stat);
		//Wait for all threads to finish calculation
//		__syncthreads();
/*		if(tid < 32)
		{
			kernel_store_stat(gpu_tmp, tid);	
			__syncthreads();
		}
		if(tid == 0)
		{
			stat = kernel_store_final_stat(gpu_tmp, gpu_output, output_ptr);
			if(stat > 0)
			{
				l = _stat.lm;
				pe[0] = eTD + (tsize/2)*V;
				pe[1] = eTD + vDim + (tsize/2)*V;
				pe[2] = eTD + 2*vDim + (tsize/2)*V;
				pe[3] = eTD + 3*vDim + (tsize/2)*V;
                        	pe[0] = pe[0] + ml[0][l] * (int)V;
	                        pe[1] = pe[1] + ml[1][l] * (int)V;
        	                pe[2] = pe[2] + ml[2][l] * (int)V;
                	        pe[3] = pe[3] + ml[3][l] * (int)V;
				size_t v;
				for(v=0; v<V; v++)
					gpu_output[output_ptr+OutputSize+v] = pe[0][v] + pe[1][v] + pe[2][v] + pe[3][v];
			}
		}*/
//		output_ptr += (OutputSize + V); 
		output_ptr += Lsky;
		v_ptr += vDim*gpu_nIFO*2 + gpu_nIFO*V;
		etd_ptr += vDim*gpu_nIFO;
		count++;
	}
	
	return;
}
__inline__ __device__ void kernel_skyloop_calculate(short **ml, float *nr, double **FP, double **FX, float *gpu_BB, float *gpu_bb, float *gpu_fp, float *gpu_fx, float *gpu_Fp, float *gpu_Fx, float **pa, float **pA, float *PE_0, float *PE_1, float *PE_2, size_t V, float *gpu_output,  int l, struct STAT *_s, int tid, int k, int output_ptr)
{
        int msk;                                              // mask
        size_t v;                                  // indicate the pixel
	size_t V4;
	size_t j;
        float pe[gpu_nIFO];
	float *v00[gpu_nIFO];
	float *v90[gpu_nIFO];
        float _Eo[4], _Es[4], _En[4];
        float Ln, Ls, Eo, aa;
        float rE;                                               // energy array rNRG.data 
        float pE;                                               // energy array pNRG.data
        int count;
	int m; 
	float ee, em, Lo, AA;

        m = 0;                                                 // # of pixels above the threshold
        for(count=0; count<4; count++)
        {
                _Eo[count] = 0;
                _Es[count] = 0;
                _En[count] = 0;
        }

        count = 0;
        for(v=0; v<V; v++)                                      // loop over selected pixels    
        {
                // *_rE = _sse_sum_ps(_pe);
                pe[0] = PE_0[v];
                pe[1] = PE_1[v];
                pe[2] = PE_2[v];
                rE = pe[0] + pe[1] + pe[2] + 0.0;                                                             // get pixel energy
	/*	if(k==4 && l == 256)		
		{
			gpu_output[15+v*5] = pe[0];	
			gpu_output[15+v*5+1] = pe[1];	
			gpu_output[15+v*5+2] = pe[2];
			gpu_output[15+v*5+3] = pe[0] + pe[1] + pe[2];
			gpu_output[15+v*5+4] = rE;
		}*/
	        // E>En  0/1 mask
                msk = ( rE>=constEn );                                                                          // E>En  0/1 mask
                m += msk;                                                                                              // count pixels above threshold
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

        Ln = _En[0] + _En[1] + _En[2] + _En[3];                 // Write back to output
        Eo = _Eo[0] + _Eo[1] + _Eo[2] + _Eo[3] + 0.01;
        Ls = _Es[0] + _Es[1] + _Es[2] + _Es[3];
        m = m * 2 + 0.01;
        aa = Ls*Ln/(Eo-Ls);

        msk = ((aa-m)/(aa+m)<0.33);
//	gpu_output[output_ptr+l] = aa*(1-msk) - 1*msk;
/*	if(k==4)
	{
		int Lsky = 196608;
		gpu_output[l] = PE_0[0];
		gpu_output[l+Lsky] = PE_1[0];
		gpu_output[l+2*Lsky] = PE_2[0];
		gpu_output[l+3*Lsky] = PE_0[0] + PE_1[0] + PE_2[0];
		gpu_output[l+4*Lsky] = aa;
	}*/
	//float *bb, *BB, *fp, *Fp, *fx, *Fx;
	if(msk)	return;

	// after skyloop

	v00[0] = pa[0] + ml[0][l] * (int)V;
	v00[1] = pa[1] + ml[1][l] * (int)V;
	v00[2] = pa[2] + ml[2][l] * (int)V;
	v90[0] = pA[0] + ml[0][l] * (int)V;
	v90[1] = pA[1] + ml[1][l] * (int)V;
	v90[2] = pA[2] + ml[2][l] * (int)V;
	
	msk = V%4;
	msk = (msk>0);
	V4 = V + msk*(4-V%4);
	// point to the memory
	
	for(j=0; j<V; j++)
	{
		//cpp_
		gpu_bb[tid+j*NIFO*GRID_SIZE] = v00[0][j];
		gpu_bb[tid+GRID_SIZE+j*NIFO*GRID_SIZE] = v00[1][j];
		gpu_bb[tid+2*GRID_SIZE+j*NIFO*GRID_SIZE] = v00[2][j];
		gpu_bb[tid+3*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_BB[tid+j*NIFO*GRID_SIZE] = v90[0][j];
		gpu_BB[tid+GRID_SIZE+j*NIFO*GRID_SIZE] = v90[1][j];
		gpu_BB[tid+2*GRID_SIZE+j*NIFO*GRID_SIZE] = v90[2][j];
		gpu_BB[tid+3*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;

		//cpf_
		gpu_fp[tid+j*NIFO*GRID_SIZE] = FP[0][l];
		gpu_fp[tid+GRID_SIZE+j*NIFO*GRID_SIZE] = FP[1][l];
		gpu_fp[tid+2*GRID_SIZE+j*NIFO*GRID_SIZE] = FP[2][l];
		gpu_fp[tid+3*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_fx[tid+j*NIFO*GRID_SIZE] = FX[0][l];
		gpu_fx[tid+GRID_SIZE+j*NIFO*GRID_SIZE] = FX[1][l];
		gpu_fx[tid+2*GRID_SIZE+j*NIFO*GRID_SIZE] = FX[2][l];
		gpu_fx[tid+3*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
//		kernel_cpf_(fp+j*NIFO, FP, l);
//		kernel_cpf_(fx+j*NIFO, FX, l);
	}
/*	if(k==4)
	{
		int Lsky = 196608;
		gpu_output[l] = *(gpu_fp+tid);
		gpu_output[l+Lsky] = *(gpu_fp+tid+GRID_SIZE);
		gpu_output[l+2*Lsky] = *(gpu_fp+tid+2*GRID_SIZE);
		gpu_output[l+3*Lsky] = *(gpu_fp+tid+3*GRID_SIZE);
		gpu_output[l+4*Lsky] = *(gpu_fx+tid);
		gpu_output[l+5*Lsky] = *(gpu_fx+tid+GRID_SIZE);
		gpu_output[l+6*Lsky] = *(gpu_fx+tid+2*GRID_SIZE);
		gpu_output[l+7*Lsky] = *(gpu_fx+tid+3*GRID_SIZE);
	}*/

	for(j=V; j<V4; j++)
	{
		gpu_bb[tid+j*NIFO*GRID_SIZE] = 0.;
		gpu_bb[tid+GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_bb[tid+2*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_bb[tid+3*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_BB[tid+j*NIFO*GRID_SIZE] = 0.;
		gpu_BB[tid+GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_BB[tid+2*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_BB[tid+3*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;

		gpu_fp[tid+j*NIFO*GRID_SIZE] = 0.;
		gpu_fp[tid+GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_fp[tid+2*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_fp[tid+3*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_fx[tid+j*NIFO*GRID_SIZE] = 0.;
		gpu_fx[tid+GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_fx[tid+2*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_fx[tid+3*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
	}
	/*for(j=0; j<V; j++)
	{
		//cpp_
		gpu_bb[tid+j*GRID_SIZE] = v00[0][j];
		gpu_bb[tid+j*GRID_SIZE+VMAX*GRID_SIZE] = v00[1][j];
		gpu_bb[tid+j*GRID_SIZE+2*VMAX*GRID_SIZE] = v00[1][j];
		gpu_BB[tid+j*GRID_SIZE] = v90[0][j];
		gpu_BB[tid+j*GRID_SIZE+VMAX*GRID_SIZE] = v90[1][j];
		gpu_BB[tid+j*GRID_SIZE+2*VMAX*GRID_SIZE] = v90[1][j];

		//cpf_
		gpu_fp[tid+j*GRID_SIZE] = FP[0][l];
		gpu_fp[tid+j*GRID_SIZE+VMAX*GRID_SIZE] = FP[1][l];
		gpu_fp[tid+j*GRID_SIZE+2*VMAX*GRID_SIZE] = FP[1][l];
		gpu_fx[tid+j*GRID_SIZE] = FX[0][l];
		gpu_fx[tid+j*GRID_SIZE+VMAX*GRID_SIZE] = FX[1][l];
		gpu_fx[tid+j*GRID_SIZE+2*VMAX*GRID_SIZE] = FX[1][l];
//		kernel_cpf_(fp+j*NIFO, FP, l);
//		kernel_cpf_(fx+j*NIFO, FX, l);
	}*/
	
/*	if(k == 4&& l==0)
	{
		for(j=0; j<V; j++)
		{
			gpu_output[j*NIFO] = gpu_bb[tid+j*gpu_nIFO*GRID_SIZE];
			gpu_output[j*NIFO+1] = gpu_bb[tid+GRID_SIZE+j*gpu_nIFO*GRID_SIZE];
			gpu_output[j*NIFO+2] = gpu_bb[tid+2*GRID_SIZE+j*gpu_nIFO*GRID_SIZE];
			gpu_output[j*NIFO+16] = gpu_BB[tid+j*gpu_nIFO*GRID_SIZE];
			gpu_output[j*NIFO+1+16] = gpu_BB[tid+GRID_SIZE+j*gpu_nIFO*GRID_SIZE];
			gpu_output[j*NIFO+2+16] = gpu_BB[tid+2*GRID_SIZE+j*gpu_nIFO*GRID_SIZE];
		}
	}*/
//	if(k == 4)
//		gpu_output[l] = gpu_bb[tid];
	
	m = 0; Ls=Ln=Eo=0;
	for(j=0; j<V; j++)
	{
		ee = kernel_sse_abs_ps(gpu_bb+tid+j*NIFO*GRID_SIZE, gpu_BB+tid+j*NIFO*GRID_SIZE);
		if(ee<constEn)	continue;
		kernel_sse_cpf_ps(gpu_bb+tid+m*NIFO*GRID_SIZE, gpu_bb+tid+j*NIFO*GRID_SIZE);
		kernel_sse_cpf_ps(gpu_BB+tid+m*NIFO*GRID_SIZE, gpu_BB+tid+j*NIFO*GRID_SIZE);
		kernel_sse_cpf_ps(gpu_Fx+tid+m*NIFO*GRID_SIZE, gpu_fx+tid+j*NIFO*GRID_SIZE);
		kernel_sse_cpf_ps(gpu_Fp+tid+m*NIFO*GRID_SIZE, gpu_fp+tid+j*NIFO*GRID_SIZE);
		kernel_sse_mul_ps(gpu_Fp+tid+m*NIFO*GRID_SIZE, nr+j*gpu_nIFO);
		kernel_sse_mul_ps(gpu_Fx+tid+m*NIFO*GRID_SIZE, nr+j*gpu_nIFO);
		m++;
		em = kernel_sse_maxE_ps(gpu_bb+tid+j*NIFO*GRID_SIZE, gpu_BB+tid+j*NIFO*GRID_SIZE);
		Ls += ee-em;	Eo += ee;
		msk = ( (ee-em)>constEs );
		Ln += msk*ee;
	}

	/*if(k==4)
	{
		int Lsky = 196608;
		gpu_output[l] = ee;
		gpu_output[l+Lsky] = em;
		gpu_output[l+2*Lsky] = Ls;
		gpu_output[l+3*Lsky] = Eo;
		gpu_output[l+4*Lsky] = Ln;
		gpu_output[l+5*Lsky] = m;
	}*/

	/*if(k==4 && l==0)
	{
		float *Fp = gpu_Fp+tid+(V-1)*gpu_nIFO*GRID_SIZE;
		float *Fx= gpu_Fx+tid+(V-1)*gpu_nIFO*GRID_SIZE;
		gpu_output[0] = Fp[0];
		gpu_output[1] = Fp[GRID_SIZE];
		gpu_output[2] = Fp[2*GRID_SIZE];
		gpu_output[3] = Fx[0];
		gpu_output[4] = Fx[GRID_SIZE];
		gpu_output[5] = Fx[2*GRID_SIZE];
	}
	if(k==4 && l==1)
	{
		float *Fp = gpu_Fp+tid+(V-1)*gpu_nIFO*GRID_SIZE;
		float *Fx= gpu_Fx+tid+(V-1)*gpu_nIFO*GRID_SIZE;
		gpu_output[6] = Fp[0];
		gpu_output[7] = Fp[GRID_SIZE];
		gpu_output[8] = Fp[2*GRID_SIZE];
		gpu_output[9] = Fx[0];
		gpu_output[10] = Fx[GRID_SIZE];
		gpu_output[11] = Fx[2*GRID_SIZE];
	}*/
	
	msk = m%4;
	msk = (msk>0);
	size_t m4 = m + msk*(4-m%4);
	_En[0] = _En[1] = _En[2] = _En[3] = 0;
	
	gpu_output[l+9*196608] = m;
	
	for(j=m; j<m4; j++)
	{
		gpu_Fp[tid+j*NIFO*GRID_SIZE] = 0.;
		gpu_Fp[tid+GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_Fp[tid+2*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_Fp[tid+3*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_Fx[tid+j*NIFO*GRID_SIZE] = 0.;
		gpu_Fx[tid+GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_Fx[tid+2*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
		gpu_Fx[tid+3*GRID_SIZE+j*NIFO*GRID_SIZE] = 0.;
	}
/*	if(k==4)
	{
		int Lsky = 196608;
		gpu_output[l] = *(gpu_Fp+tid+3*NIFO*GRID_SIZE);
		gpu_output[l+Lsky] = *(gpu_Fp+tid+GRID_SIZE+3*NIFO*GRID_SIZE);
 		gpu_output[l+2*Lsky] = *(gpu_Fp+tid+2*GRID_SIZE+3*NIFO*GRID_SIZE);
		gpu_output[l+3*Lsky] = *(gpu_Fp+tid+3*GRID_SIZE+3*NIFO*GRID_SIZE);
		gpu_output[l+4*Lsky] = *(gpu_fp+tid+3*NIFO*GRID_SIZE);
		gpu_output[l+5*Lsky] = *(gpu_fp+tid+GRID_SIZE+3*NIFO*GRID_SIZE);
		gpu_output[l+6*Lsky] = *(gpu_fp+tid+2*GRID_SIZE+3*NIFO*GRID_SIZE);
		gpu_output[l+7*Lsky] = *(gpu_fp+tid+3*GRID_SIZE+3*NIFO*GRID_SIZE);
	}*/

	for(j=0; j<m4; j+=4)
	{
		/*if(k==4)
		{
			int Lsky = 196608;
			gpu_output[l] = *(gpu_Fp+tid+NIFO*GRID_SIZE);
			gpu_output[l+Lsky] = *(gpu_Fp+tid+GRID_SIZE+NIFO*GRID_SIZE);
			gpu_output[l+2*Lsky] = *(gpu_Fp+tid+2*GRID_SIZE+NIFO*GRID_SIZE);
			gpu_output[l+3*Lsky] = *(gpu_Fp+tid+3*GRID_SIZE+NIFO*GRID_SIZE);
			gpu_output[l+4*Lsky] = *(gpu_Fx+tid+NIFO*GRID_SIZE);
			gpu_output[l+5*Lsky] = *(gpu_Fx+tid+GRID_SIZE+NIFO*GRID_SIZE);
			gpu_output[l+6*Lsky] = *(gpu_Fx+tid+2*GRID_SIZE+NIFO*GRID_SIZE);
			gpu_output[l+7*Lsky] = *(gpu_Fx+tid+3*GRID_SIZE+NIFO*GRID_SIZE);
		}*/
		kernel_sse_dpf4_ps(gpu_Fp+tid+j*NIFO*GRID_SIZE, gpu_Fx+tid+j*NIFO*GRID_SIZE, gpu_fp+tid+j*NIFO*GRID_SIZE, gpu_fx+tid+j*NIFO*GRID_SIZE, k, l, gpu_output);
		kernel_sse_like4_ps(gpu_fp+tid+j*gpu_nIFO*GRID_SIZE, gpu_fx+tid+j*gpu_nIFO*GRID_SIZE, gpu_bb+tid+j*gpu_nIFO*GRID_SIZE, gpu_BB+tid+j*gpu_nIFO*GRID_SIZE, _Es);
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
	
	/*if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l] = Lo;
		gpu_output[l+Lsky] = AA;
		gpu_output[l+2*Lsky] = ee;
		gpu_output[l+3*Lsky] = em;
		gpu_output[l+4*Lsky] = aa;
	}*/	

	// atomic operate!!!
/*
	msk = (AA > _s->stat);
	_s->stat = _s->stat+AA - _s->stat*msk - AA*(1-msk);
	_s->Lm = _s->Lm+Lo - _s->Lm*msk - Lo*(1-msk);
	_s->Em = _s->Em+Eo - _s->Em*msk - Eo*(1-msk);
	_s->Am = _s->Am+aa - _s->Am*msk - aa*(1-msk);
	_s->lm = _s->lm+l - _s->lm*msk - l*(1-msk);
	_s->Vm = _s->Vm+m - _s->Vm*msk - m*(1-msk);
	_s->suball = _s->suball+ee - _s->suball*msk - ee*(1-msk);
	_s->EE = _s->EE+em - _s->EE*msk - em*(1-msk);*/
	
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
	for(int i=0; i<num_blocks*num_threads; i+=32)
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
	
__inline__ __device__ float kernel_minSNE_ps(float pE, float *pe)
{
        float a, b, c;
        int ab, ac, bc;
        float temp;
        int flag;

        a = pe[0];
        b = pe[1];
        c = pe[2];
        ab = ( a>=b );                                                                                  // if a>=b, ab 1/0
        ac = ( a>=c );                                                                                  // if a>=c, ac 1/0
        bc = ( b>=c );                                                                                  // if b>=c, bc 1/0

        temp = a+b+c - ab*ac*a - (1-ab)*bc*b - (1-ac)*(1-bc)*c;
        flag = ( temp>=pE );                                                                            // if temp>=pE, flag 1/0
        temp = temp + pE - flag*temp - (1-flag)*pE;
        return temp;
}
/*__inline__ __device__ void kernel_cpf_(float *a, double **p, size_t i)
{
	a[0] = p[0][i];
	a[1] = p[1][i];
	a[2] = p[2][i];
	a[3] = p[3][i];
	return;
}*/
__inline__ __device__ void kernel_sse_cpf_ps(float *a, float *p)
{
	a[0] = p[0];
	a[GRID_SIZE] = p[GRID_SIZE];
	a[2*GRID_SIZE] = p[2*GRID_SIZE];
	a[3*GRID_SIZE] = p[3*GRID_SIZE];	// can be removed latter
	return;
}
__inline__ __device__ void kernel_sse_mul_ps(float *a, float *b)
{
	a[0] = a[0]*b[0];
	a[GRID_SIZE] = a[GRID_SIZE]*b[1];
	a[2*GRID_SIZE] = a[2*GRID_SIZE]*b[2];
	return;
}
__inline__ __device__ float kernel_sse_abs_ps(float *bb, float *BB)
{
	float out;
	out = bb[0]*bb[0] + BB[0]*BB[0];
	out += bb[GRID_SIZE]*bb[GRID_SIZE] + BB[GRID_SIZE]*BB[GRID_SIZE];
	out += bb[2*GRID_SIZE]*bb[2*GRID_SIZE] + BB[2*GRID_SIZE]*BB[2*GRID_SIZE];
	return out;
}
__inline__ __device__ float kernel_sse_maxE_ps(float *a, float *A)
{
	float out;
	float temp;
	bool flag;
	out = a[0]*a[0] + A[0]*A[0];
	temp = a[GRID_SIZE]*a[GRID_SIZE] + A[GRID_SIZE]*A[GRID_SIZE];
	flag = (temp>out);
	out = temp+out - (1-flag)*temp - flag*out;
	temp = a[2*GRID_SIZE]*a[2*GRID_SIZE] + A[2*GRID_SIZE]*A[2*GRID_SIZE];
	flag = (temp>out);
	out = temp+out - (1-flag)*temp - flag*out;
	return out;
}
__inline__ __device__ void kernel_sse_dpf4_ps(float *Fp, float *Fx, float *fp, float *fx, int k, int l, float *gpu_output)
{
	float _c[NIFO];					// cos
	float _s[NIFO];					// sin
	kernel_sse_ort4_ps(Fp, Fx, _s, _c, k, l, gpu_output);
/*	if(k==4)
	{
		int Lsky = 196608;
		gpu_output[l] = _c[0];
		gpu_output[l+Lsky] = _c[1];
		gpu_output[l+2*Lsky] = 5;
		gpu_output[l+3*Lsky] = _c[3];
		gpu_output[l+4*Lsky] = _s[0];
		gpu_output[l+5*Lsky] = _s[1];
		gpu_output[l+6*Lsky] = 5;
		gpu_output[l+7*Lsky] = _s[3];
	}*/
	kernel_sse_rot4p_m_ps(Fp, _c, Fx, _s, fp, fx);		// get fp = Fp*c+Fx*s, fx = Fx*c - Fp*s
//	kernel_sse_rot4m_ps(Fx, _c, Fp, _s, fx);		// get fx = Fx*c-Fp*s
}
__inline__ __device__ void kernel_sse_ort4_ps(float *u, float *v, float *_s, float *_c, int k, int l, float *gpu_output)
{
	const float sm[NIFO] = {0.f, 0.f, 0.f, 0.f};	
	const float _o[NIFO] = {1.e-24, 1.e-24, 1.e-24, 1.e-24};
	float _n[NIFO], _m[NIFO], gI[NIFO], gR[NIFO], _p[NIFO], _q[NIFO];	
	float _out[NIFO];
	
	kernel_sse_dot4_ps(u, v, _out);
	gI[0] = _out[0]*2.; gI[1] = _out[1]*2.; gI[2] = _out[2]*2.; gI[3] = _out[3]*2.; 		// ~sin(2*psi) or 2*u*v
/*	if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l] = gI[0];
		gpu_output[l+Lsky] = gI[1];
		gpu_output[l+2*Lsky] = gI[2];
		gpu_output[l+3*Lsky] = gI[3];
	}*/
	kernel_sse_dot4_ps(u, u, _out);
	gR[0] = _out[0]; gR[1] = _out[1]; gR[2] = _out[2]; gR[3] = _out[3]; 
/*	if(k==4)
	{
		int Lsky = 196608;
		gpu_output[l+4*Lsky] = gR[0];
		gpu_output[l+5*Lsky] = gR[1];
		gpu_output[l+6*Lsky] = gR[2];
		gpu_output[l+7*Lsky] = gR[3];
	}*/
	
	kernel_sse_dot4_ps(v, v, _out);
	gR[0] -= _out[0]; gR[1] -= _out[1]; gR[2] -= _out[2]; gR[3] -= _out[3]; 		// u^2-v^2
/*	if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l] = gR[0];
		gpu_output[l+Lsky] = gR[1];
		gpu_output[l+2*Lsky] = gR[2];
		gpu_output[l+3*Lsky] = gR[3];
	}*/
	
	_p[0] = (gR[0]>=0); _p[1] = (gR[1]>=0); _p[2] = (gR[2]>=0); _p[3] = (gR[3]>=0);		// 1 if gR>0. or 0 if gR<0.
/*	if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l] = _p[0];
		gpu_output[l+Lsky] = _p[1];
		gpu_output[l+2*Lsky] = _p[2];
		gpu_output[l+3*Lsky] = _p[3];
	}*/
	_q[0] = 1-_p[0]; _q[1] = 1-_p[1]; _q[2] = 1-_p[2]; _q[3] = 1-_p[3]; 			// 0 if gR>0. or 1 if gR<0.
/*	if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l+4*Lsky] = _q[0];
		gpu_output[l+5*Lsky] = _q[1];
		gpu_output[l+6*Lsky] = _q[2];
		gpu_output[l+7*Lsky] = _q[3];
	}	*/
	
	_n[0] = sqrt(gI[0]*gI[0] + gR[0]*gR[0]); _n[1] = sqrt(gI[1]*gI[1] + gR[1]*gR[1]); _n[2] = sqrt(gI[2]*gI[2] + gR[2]*gR[2]); _n[3] = sqrt(gI[3]*gI[3] + gR[3]*gR[3]); // go
/*	if(k == 4)
	{
		float a = 0.225992;
		float b = -0.449986;
		int Lsky = 196608;
		gpu_output[l] = _n[0];
		gpu_output[l+Lsky] = _n[1];
		gpu_output[l+2*Lsky] = _n[2];
//		gpu_output[l+3*Lsky] = sqrt(a*a + b*b);
		gpu_output[l+3*Lsky] = _n[3];
	}*/

	gR[0] = abs(gR[0]) + (_n[0]+_o[0]); gR[1] = abs(gR[1]) + (_n[1]+_o[1]); gR[2] = abs(gR[2]) + (_n[2]+_o[2]); gR[3] = abs(gR[3]) + (_n[3]+_o[3]); 
	/*if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l] = gR[0];
		gpu_output[l+Lsky] = gR[1];
		gpu_output[l+2*Lsky] = gR[2];
		gpu_output[l+3*Lsky] = gR[3];
	}*/
	
	_n[0] = _n[0]*2. + _o[0]; _n[1] = _n[1]*2. + _o[1]; _n[2] = _n[2]*2. + _o[2]; _n[3] = _n[3]*2. + _o[3]; 	// 2*go + eps
/*	if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l] = _n[0];
		gpu_output[l+Lsky] = _n[1];
		gpu_output[l+2*Lsky] = _n[2];
		gpu_output[l+3*Lsky] = _n[3];
	}	*/
	gI[0] = gI[0]/_n[0]; gI[1] = gI[1]/_n[1]; gI[2] = gI[2]/_n[2]; gI[3] = gI[3]/_n[3]; 	// sin(2*psi)
/*	if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l] = gI[0];
		gpu_output[l+Lsky] = gI[1];
		gpu_output[l+2*Lsky] = gI[2];
		gpu_output[l+3*Lsky] = gI[3];
	}*/

	_n[0] = sqrt(gR[0]/_n[0]); _n[1] = sqrt(gR[1]/_n[1]); _n[2] = sqrt(gR[2]/_n[2]); _n[3] = sqrt(gR[3]/_n[3]);	// sqrt((gc+|gR|)/(2gc+eps)
/*	if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l+4*Lsky] = _n[0];
		gpu_output[l+5*Lsky] = _n[1];
		gpu_output[l+6*Lsky] = _n[2];
		gpu_output[l+7*Lsky] = _n[3];
	}*/

	_m[0] = (gI[0]>=0); _m[1] = (gI[1]>=0); _m[2] = (gI[2]>=0); _m[3] = (gI[3]>=0); 		// if gI>0. or 0 if gI<0.
/*	if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l+4*Lsky] = _m[0];
		gpu_output[l+5*Lsky] = _m[1];
		gpu_output[l+6*Lsky] = _m[2];
		gpu_output[l+7*Lsky] = _m[3];
	}*/
	
	_m[0] = (_m[0]*2.-1) * _n[0]; _m[1] = (_m[1]*2.-1) * _n[1]; _m[2] = (_m[2]*2.-1) * _n[2]; _m[3] = (_m[3]*2.-1) * _n[3];	// _n if gI>0 or -_n if gI<
	// sin(psi)
	_s[0] = _q[0]*_m[0] + _p[0]*(gI[0]/_n[0]); 
	_s[1] = _q[1]*_m[1] + _p[1]*(gI[1]/_n[1]); 	
	_s[2] = _q[2]*_m[2] + _p[2]*(gI[2]/_n[2]); 
	_s[3] = _q[3]*_m[3] + _p[3]*(gI[3]/_n[3]); 
/*	if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l] = _s[0];
		gpu_output[l+Lsky] = _s[1];
		gpu_output[l+2*Lsky] = _s[2];
		gpu_output[l+3*Lsky] = _s[3];
	}*/
	gI[0] = abs(gI[0]);  gI[1] = abs(gI[1]); gI[2] = abs(gI[2]); gI[3] = abs(gI[3]); 	// |gI|
	
	// cos(psi)
	_c[0] = _p[0]*_n[0] + _q[0]*(gI[0]/_n[0]); 
	_c[1] = _p[1]*_n[1] + _q[1]*(gI[1]/_n[1]); 
	_c[2] = _p[2]*_n[2] + _q[2]*(gI[2]/_n[2]); 
	_c[3] = _p[3]*_n[3] + _q[3]*(gI[3]/_n[3]); 
/*	if(k == 4)
	{
		int Lsky = 196608;
		gpu_output[l+4*Lsky] = _c[0];
		gpu_output[l+5*Lsky] = _c[1];
		gpu_output[l+6*Lsky] = _c[2];
		gpu_output[l+7*Lsky] = _c[3];
	}*/
	return;
}
__inline__ __device__ void kernel_sse_dot4_ps(float *u, float *v, float *out)
{
	out[0] = u[0]*v[0];
	out[0] += u[GRID_SIZE]*v[GRID_SIZE];
	out[0] += u[2*GRID_SIZE]*v[2*GRID_SIZE];

	out[1] = u[NIFO*GRID_SIZE]*v[NIFO*GRID_SIZE];
	out[1] += u[NIFO*GRID_SIZE+GRID_SIZE]*v[NIFO*GRID_SIZE+GRID_SIZE];
	out[1] += u[NIFO*GRID_SIZE+2*GRID_SIZE]*v[NIFO*GRID_SIZE+2*GRID_SIZE];

	out[2] = u[2*NIFO*GRID_SIZE]*v[2*NIFO*GRID_SIZE];
	out[2] += u[2*NIFO*GRID_SIZE+GRID_SIZE]*v[2*NIFO*GRID_SIZE+GRID_SIZE];
	out[2] += u[2*NIFO*GRID_SIZE+2*GRID_SIZE]*v[2*NIFO*GRID_SIZE+2*GRID_SIZE];

	out[3] = u[3*NIFO*GRID_SIZE]*v[3*NIFO*GRID_SIZE];
	out[3] += u[3*NIFO*GRID_SIZE+GRID_SIZE]*v[3*NIFO*GRID_SIZE+GRID_SIZE];
	out[3] += u[3*NIFO*GRID_SIZE+2*GRID_SIZE]*v[3*NIFO*GRID_SIZE+2*GRID_SIZE];
	return;
}
__inline__ __device__ void kernel_sse_rot4p_m_ps(float *Fp, float *_c, float *Fx, float *_s, float *fp, float *fx)
{
	fp[0] = Fp[0]*_c[0] + Fx[0]*_s[0];	
	fx[0] = Fx[0]*_c[0] - Fp[0]*_s[0];
	fp[GRID_SIZE] = Fp[GRID_SIZE]*_c[0] + Fx[GRID_SIZE]*_s[0];	
	fx[GRID_SIZE] = Fx[GRID_SIZE]*_c[0] - Fp[GRID_SIZE]*_s[0];	
	fp[2*GRID_SIZE] = Fp[2*GRID_SIZE]*_c[0] + Fx[2*GRID_SIZE]*_s[0];	
	fx[2*GRID_SIZE] = Fx[2*GRID_SIZE]*_c[0] - Fp[2*GRID_SIZE]*_s[0];	
	
	fp[gpu_nIFO*GRID_SIZE] = Fp[gpu_nIFO*GRID_SIZE]*_c[1] + Fx[gpu_nIFO*GRID_SIZE]*_s[1];	
	fx[gpu_nIFO*GRID_SIZE] = Fx[gpu_nIFO*GRID_SIZE]*_c[1] - Fp[gpu_nIFO*GRID_SIZE]*_s[1];	
	fp[gpu_nIFO*GRID_SIZE+GRID_SIZE] = Fp[gpu_nIFO*GRID_SIZE+GRID_SIZE]*_c[1] + Fx[gpu_nIFO*GRID_SIZE+GRID_SIZE]*_s[1];	
	fx[gpu_nIFO*GRID_SIZE+GRID_SIZE] = Fx[gpu_nIFO*GRID_SIZE+GRID_SIZE]*_c[1] - Fp[gpu_nIFO*GRID_SIZE+GRID_SIZE]*_s[1];	
	fp[gpu_nIFO*GRID_SIZE+2*GRID_SIZE] = Fp[gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_c[1] + Fx[gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_s[1];	
	fx[gpu_nIFO*GRID_SIZE+2*GRID_SIZE] = Fx[gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_c[1] - Fp[gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_s[1];	

	fp[2*gpu_nIFO*GRID_SIZE] = Fp[2*gpu_nIFO*GRID_SIZE]*_c[1] + Fx[2*gpu_nIFO*GRID_SIZE]*_s[1];	
	fx[2*gpu_nIFO*GRID_SIZE] = Fx[2*gpu_nIFO*GRID_SIZE]*_c[1] - Fp[2*gpu_nIFO*GRID_SIZE]*_s[1];	
	fp[2*gpu_nIFO*GRID_SIZE+GRID_SIZE] = Fp[2*gpu_nIFO*GRID_SIZE+GRID_SIZE]*_c[1] + Fx[2*gpu_nIFO*GRID_SIZE+GRID_SIZE]*_s[1];	
	fx[2*gpu_nIFO*GRID_SIZE+GRID_SIZE] = Fx[2*gpu_nIFO*GRID_SIZE+GRID_SIZE]*_c[1] - Fp[2*gpu_nIFO*GRID_SIZE+GRID_SIZE]*_s[1];	
	fp[2*gpu_nIFO*GRID_SIZE+2*GRID_SIZE] = Fp[2*gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_c[1] + Fx[2*gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_s[1];	
	fx[2*gpu_nIFO*GRID_SIZE+2*GRID_SIZE] = Fx[2*gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_c[1] - Fp[2*gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_s[1];	

	fp[3*gpu_nIFO*GRID_SIZE] = Fp[3*gpu_nIFO*GRID_SIZE]*_c[1] + Fx[3*gpu_nIFO*GRID_SIZE]*_s[1];	
	fx[3*gpu_nIFO*GRID_SIZE] = Fx[3*gpu_nIFO*GRID_SIZE]*_c[1] - Fp[3*gpu_nIFO*GRID_SIZE]*_s[1];	
	fp[3*gpu_nIFO*GRID_SIZE+GRID_SIZE] = Fp[3*gpu_nIFO*GRID_SIZE+GRID_SIZE]*_c[1] + Fx[3*gpu_nIFO*GRID_SIZE+GRID_SIZE]*_s[1];	
	fx[3*gpu_nIFO*GRID_SIZE+GRID_SIZE] = Fx[3*gpu_nIFO*GRID_SIZE+GRID_SIZE]*_c[1] - Fp[3*gpu_nIFO*GRID_SIZE+GRID_SIZE]*_s[1];	
	fp[3*gpu_nIFO*GRID_SIZE+2*GRID_SIZE] = Fp[3*gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_c[1] + Fx[3*gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_s[1];	
	fx[3*gpu_nIFO*GRID_SIZE+2*GRID_SIZE] = Fx[3*gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_c[1] - Fp[3*gpu_nIFO*GRID_SIZE+2*GRID_SIZE]*_s[1];	
	/*fp[12] = Fp[12]*_c[3] + Fx[12]*_s[3];	
	fp[13] = Fp[13]*_c[3] + Fx[13]*_s[3];	
	fp[14] = Fp[14]*_c[3] + Fx[14]*_s[3];*/
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
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *post_gpu_data)
{
	FILE *fpt = fopen("./new_debug/myk4_fptone", "a");
	FILE *fpt1 = fopen("./new_debug/myk4_fpttwo", "a");
//
	int Lsky = gpu_Lsky;
	int k;
	size_t V;
	int pixelcount=0;
	int streamNum;
	size_t output_ptr = 0;
	float *aa;
	//cout<<"Callback"<<endl;
	for(int i=0; i<StreamNum; i++)
	{
		k = ((post_data*)post_gpu_data)->other_data.k[pixelcount] - 1;
		
		while(k != -1)
		{
			V = ((post_data*)post_gpu_data)->other_data.V[pixelcount];
			/*for(int l=0; l<Lsky; l++)
			{	
				aa = ((post_data*)post_gpu_data)->output.output[l+output_ptr];
				sif(aa != -1)
				{
					fprintf(fpt, "k = %d l = %d aa = %f\n", k, l, aa);
					cc++;
				}
			}*/	
			//after_skyloop((void*)&post_gpu_data[i], gpu_net, gpu_hist, pwc, FP, FX, pa[streamNum][pixelcount], pA[streamNum][pixelcount], pixelcount, output_ptr, Lsky, gpu_time, streamCount);
			//cout<<"k = "<<k<<" V = "<<V<<endl;
			//aa = ((post_data*)post_gpu_data)->output.output[l+output_ptr];
			/*if(k==4)
			{
				aa = ((post_data*)post_gpu_data)->output.output;
				for(int l=0; l<Lsky; l++)
				{
					fprintf(fpt, "l = %d Lo = %f AA = %f ee = %f em = %f aa = %f\n", l, aa[l], aa[l+Lsky], aa[l+2*Lsky], aa[l+3*Lsky], aa[l+4*Lsky]);
				}
			}*/
			if(k==4)
			{
				aa = ((post_data*)post_gpu_data)->output.output;
				float z = 0.0;
				for(int l=0; l<Lsky; l++)
				{
		//			fprintf(fpt, "k = %d l = %d ee = %f em = %f Ls = %f Eo = %f Ln = %f m = %f\n", k, l, aa[l], aa[l+Lsky], aa[l+2*Lsky], aa[l+3*Lsky], aa[l+4*Lsky], aa[l+5*Lsky]);
				//	fprintf(fpt, "k = %d l = %d gI[0] = %f gI[1] = %f gI[2] = %f gI[3] = %f\n", k, l, aa[l], aa[l+Lsky], aa[l+2*Lsky], aa[l+3*Lsky]);
				//	fprintf(fpt1, "k = %d l = %d gR[0] = %f gR[1] = %f gR[2] = %f gR[3] = %f\n", k, l, aa[l+4*Lsky], aa[l+5*Lsky], aa[l+6*Lsky], aa[l+7*Lsky]);
			//		fprintf(fpt, "k = %d l = %d Fp[0] = %f Fp[1] = %f Fp[2] = %f Fp[3] = %f\n", k, l, aa[l], aa[l+Lsky], aa[l+2*Lsky], aa[l+3*Lsky]);
//					fprintf(fpt1, "k = %d l = %d m = %d Fx[0] = %f Fx[1] = %f Fx[2] = %f Fx[3] = %f\n", k, l, aa[l+9*Lsky], aa[l+4*Lsky], aa[l+5*Lsky], aa[l+6*Lsky], aa[l+7*Lsky]);
					fprintf(fpt, "k = %d l = %d _s[0] = %f _s[1] = %f _s[2] = %f _s[3] = %f\n", k, l, aa[l], aa[l+Lsky], aa[l+2*Lsky], aa[l+3*Lsky]);
					fprintf(fpt1, "k = %d l = %d _c[0] = %f _c[1] = %f _c[2] = %f _c[3] = %f\n", k, l, aa[l+4*Lsky], aa[l+5*Lsky], aa[l+6*Lsky], aa[l+7*Lsky]);
				}
				cout<<"finish"<<endl;
			}
/*			if(k==151)
			{
			int vDim = 43*3;
			aa = ((post_data*)post_gpu_data)->output.output;
			for(int l=0; l<Lsky; l++)
			{	
				fprintf(fpt, "l = %d pe[0] = %f pe[1] = %f pe[2] = %f ml[0] = %f ml[1] = %f\n", aa[l], aa[l+Lsky], aa[l+2*Lsky], aa[l+3*Lsky], aa[l+4*Lsky]);
//				fprintf(fpt, "%f %f %f %f %f\n", aa[l], aa[l+1], aa[l+2], aa[l+3], aa[l+4], aa[l+5]);
				
			}
			}*/

			output_ptr += Lsky;
			pixelcount++;
			if(pixelcount<MaxPixel)
				k = ((post_data*)post_gpu_data)->other_data.k[pixelcount] - 1;
			else 
				break;
		}
	}
	fclose(fpt);
	fclose(fpt1);

	
}
/*void MyCallback(struct post_data *post_gpu_data)
{
	FILE *fpt = fopen("/home/hpc/cWB/Big_input_file/my_k4etd", "a");
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

			//after_skyloop((void*)&post_gpu_data[i], gpu_net, gpu_hist, pwc, FP, FX, pa[streamNum][pixelcount], pA[streamNum][pixelcount], pixelcount, output_ptr, Lsky, gpu_time, streamCount);
			cout<<"k = "<<k<<" V4 = "<<V4<<endl;
			if(k == 4)
				for(int l=0; l<V4MAX+OutputSize; l++)
					fprintf(fpt, "%f\n", post_gpu_data[i].output.output[l]); 
			
			output_ptr = output_ptr + V4*Lsky + Lsky;
			pixelcount++;
			if(pixelcount<MaxPixel)
				k = post_gpu_data[i].other_data.k[pixelcount] - 1;
			else 
				break;
		}
	}
//	fclose(fpt);
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
void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int vDim, int Vmax, int Lsky, size_t K)// allocate locked memory on CPU 
{
        for(int i = 0; i<BufferNum; i++)
        {
                CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.vtd_vTD_nr), 2*gpu_nIFO*vDim*sizeof(float) + gpu_nIFO*Vmax*sizeof(float) + HEAD_SIZE*sizeof(float), cudaHostAllocMapped ) );
                CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.eTD), gpu_nIFO*vDim*sizeof(float), cudaHostAllocMapped ) );
        }
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.FP_FX), 2 * gpu_nIFO* Lsky * sizeof(double), cudaHostAllocMapped ) );
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.ml_mm), (1 + gpu_nIFO) * Lsky * sizeof(short), cudaHostAllocMapped ) );
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.V_tsize), K * 2 * sizeof(size_t), cudaHostAllocMapped ) );
        for(int i = 0; i<StreamNum; i++)
        {
     //           CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.output), OutputSize*sizeof(float) + Vmax*sizeof(float), cudaHostAllocMapped ) );
                CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.output), MaxPixel*Lsky*sizeof(float), cudaHostAllocMapped ) );	//used for debug
                post_gpu_data[i].other_data.ml_mm = (short*)malloc(sizeof(size_t) * (1 + gpu_nIFO) * Lsky);
        }
        return;
}
void cleanup_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, cudaStream_t *stream)
{
        for(int i = 0; i<BufferNum; i++)
        {
                CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.vtd_vTD_nr));
                CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.eTD));
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
void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int vDim, int Vmax, int Lsky, size_t K)// allocate the memory on GPU
{
        for(int i = 0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].vtd_vTD_nr), 2*gpu_nIFO*vDim*sizeof(float) + gpu_nIFO*Vmax*sizeof(float) + HEAD_SIZE*sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].eTD), gpu_nIFO * vDim * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].BB), num_blocks * num_threads * Vmax * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].bb), num_blocks * num_threads * Vmax * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].fp), num_blocks * num_threads * Vmax * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].fx), num_blocks * num_threads * Vmax * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].Fp), num_blocks * num_threads * Vmax * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].Fx), num_blocks * num_threads * Vmax * NIFO * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].tmp), num_blocks*num_threads*OutputSize*sizeof(float) ) );
//                CUDA_CHECK(cudaMalloc(&(skyloop_output[i].output), OutputSize*sizeof(float) + Vmax*sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_output[i].output), MaxPixel*Lsky*sizeof(float) ) );
        }
        CUDA_CHECK(cudaMalloc(&(skyloop_other[0].FP_FX), 2 * gpu_nIFO * Lsky * sizeof(double) ) );
        CUDA_CHECK(cudaMalloc(&(skyloop_other[0].ml_mm), (1 + gpu_nIFO) * Lsky * sizeof(short) ) );
        CUDA_CHECK(cudaMalloc(&(skyloop_other[0].V_tsize), K * 2 * sizeof(size_t) ) );
	return;
}
void cleanup_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, cudaStream_t *stream)// cleanup the memory on GPU
{
        for(int i = 0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaFree(skyloop_other[i].vtd_vTD_nr) );
                CUDA_CHECK(cudaFree(skyloop_other[i].eTD) );
                CUDA_CHECK(cudaFree(skyloop_other[i].BB) );
                CUDA_CHECK(cudaFree(skyloop_other[i].bb) );
                CUDA_CHECK(cudaFree(skyloop_other[i].fp) );
                CUDA_CHECK(cudaFree(skyloop_other[i].fx) );
                CUDA_CHECK(cudaFree(skyloop_other[i].Fp) );
                CUDA_CHECK(cudaFree(skyloop_other[i].Fx) );
                CUDA_CHECK(cudaFree(skyloop_other[i].tmp) );
                CUDA_CHECK(cudaFree(skyloop_output[i].output) );
        }
        CUDA_CHECK(cudaFree(skyloop_other[0].FP_FX) );
        CUDA_CHECK(cudaFree(skyloop_other[0].ml_mm) );
        CUDA_CHECK(cudaFree(skyloop_other[0].V_tsize) );
        return;
}


