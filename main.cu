#include "/home/hpc/cWB/TEST/S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2/macro/gpu_struct.h"
#include "main.cuh"
#include <xmmintrin.h>
#include "wavearray.hh"
#include "gpu_network.hh"
//#include "/home/hpc/cWB/TEST/S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2/macro/gpu_function.h"
//#include "/home/hpc/cWB/trunk/wat/network.hh"
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

#define StreamNum 4 
#define BufferNum 4  
#define CONSTANT_SIZE 1500

network *gpu_net;
TH2F *gpu_hist;
netcluster *pwc;
double *FP[NIFO];
double *FX[NIFO];
float *pa[NIFO];
float *pA[NIFO];
double gpu_d[10];
size_t gpu_nIFO;
size_t streamCount[StreamNum]; // the result of each stream

// GPU constant memory
__constant__ float constEn, constEs;	// two threshold
__constant__ size_t constV[CONSTANT_SIZE], consttsize[CONSTANT_SIZE];
#define CUDA_CHECK(value) {                                             \
    cudaError_t _m_cudaStat = value;                                    \
    if (_m_cudaStat != cudaSuccess) {                                   \
        fprintf(stderr, "Error %s at line %d in file %s\n",             \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
        exit(1); }}

extern long Callback(void *post_gpu_data, network *gpu_net,  TH2F *gpu_hist, netcluster *pwc, double **FP, double **FX, float **pa, float **pA, size_t *streamCount, double *d);
long gpu_subNetCut(network *net, int lag, float snc, TH2F *hist, double *d)
{
	//define variables
	size_t nIFO = net->ifoList.size();
	
	float En = 2*net->acor*net->acor*nIFO;	// network energy threshold in the sky loop
	float Es = 2*net->e2or;			// subnet energy threshold in the sky loop
	float TH = fabs(snc);			// sub network threshold
	
	int l;
	float aa, AA;	
	size_t i, j, k, V, V4, id, K;
	int Lsky = int(net->index.size());
	short *mm = net->skyMask.data;
	
	short *ml[NIFO];

	clock_t start[10], finish[10];
	
	for(i=0; i<10; i++)
		gpu_d[i] = d[i];

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
	
	// allocate buffers
	std::vector<int> pI;			// buffer for pixel IDs	
	wavearray<double> cid;			// buffers for cluster ID
	netpixel *pix;	
	std::vector<int> *vint;
	pwc = &net->wc_List[lag];
	size_t count = 0;
	size_t tsize = 0;
	size_t V4max = 0;				// store the maximum of V4
	size_t Tmax = 0;				// store the maximum of tsize
	size_t *V_array, *tsize_array;
	size_t k_array[StreamNum];

//++++++++++++++++++++++++++++++++
// find out the maximum of V and tsize 
//++++++++++++++++++++++++++++++++

   	cid = pwc->get((char*)"ID",  0,'S',0);                  // get cluster ID

	K = cid.size();
	
	V_array = (size_t*)malloc(sizeof(size_t) * K);
    tsize_array = (size_t*)malloc(sizeof(size_t) * K);
	
	//cout<<"1"<<endl;
	for(k=0; k<K; k++)				// loop over clusters
	{
		V_array[k] = 0;
		tsize_array[k] = 0;
		id = size_t(cid.data[k]+0.1);
		if(pwc->sCuts[id-1] != -2) continue;	// skip rejected/processed culster
		vint = &(pwc->cList[id-1]);		// pixel list
		V = vint->size();			// pixel list size
		if(!V) continue;
		
		pI = net->wdmMRA.getXTalk(pwc, id);
	
		V = pI.size();				// number of loaded pixels
		if(!V) continue;
		
		pix = pwc->getPixel(id, pI[0]);
		tsize = pix->tdAmp[0].size();
		if(!tsize || tsize&1)			// tsize%1 = 1/0 = power/amplitude 
		{					 
			cout<<"network::subNetCut() error: wrong pixel TD data\n";
			exit(1);
		}
	
		tsize /= 2;
	    V4 = V + (V%4 ? 4 - V%4 : 0);     
		V_array[k] = V;
		tsize_array[k] = tsize;
		if( tsize > Tmax )
			Tmax = tsize;
		if( V4 > V4max )
			V4max = V4;
	}
	
	if(K<=CONSTANT_SIZE)
	{
		cudaMemcpyToSymbol(constV, V_array, sizeof(size_t) * K);
		cudaMemcpyToSymbol(consttsize, tsize_array, sizeof(size_t) * K);
	}
	else
	{
		cudaMemcpyToSymbol(constV, V_array, sizeof(size_t) * CONSTANT_SIZE);
		cudaMemcpyToSymbol(consttsize, tsize_array, sizeof(size_t) * CONSTANT_SIZE);
	}
	
//++++++++++++++++++++++++++++++++
// declare the variables used for gpu calculation 
//++++++++++++++++++++++++++++++++
	struct pre_data pre_gpu_data[BufferNum];	// store the data before gpu cal
	struct post_data post_gpu_data[StreamNum];	// store the data transfer from gpu
	struct skyloop_output skyloop_output[StreamNum];// store the skyloop_output data
	struct other skyloop_other[StreamNum];		// store the data which is not output
	
	int eTDDim = 0;					// the size of each eTD
	int alloced_gpu = 0;				// the number of gpu which has been allocated data
	int constptr = CONSTANT_SIZE;
	
	//cudaEvent_t start, stop;		// CUDA event for recording time
	//float elapsedTime;				// 
	//float wholeTime = 0;				// 
	start[0] = clock();
	
	eTDDim = Tmax * V4max;
	for(int i=0; i<StreamNum; i++)
	{
		streamCount[i] = 0;
	}	
	// allocate the memory on cpu and gpu
	allocate_cpu_mem(pre_gpu_data, post_gpu_data, eTDDim, V4max, Lsky);
	allocate_gpu_mem(skyloop_output, skyloop_other, eTDDim, V4max, Lsky);
	
	gpu_net = net;
	gpu_hist = hist;
	gpu_nIFO = nIFO;
	//cout<<"XIFO = "<<XIFO<<endl;
	//cout<<"Lsky = "<<Lsky<<endl;
	cudaStream_t stream[StreamNum];			// define the stream
	for(int i=0; i<StreamNum; i++)			
		CUDA_CHECK(cudaStreamCreate(&stream[i]));	// create the new stream
	cudaMemcpyToSymbol(constEn, &En, sizeof(float));
	cudaMemcpyToSymbol(constEs, &Es, sizeof(float));

	for(int i=0; i<BufferNum; i++)		// initialize the data
	{
		for(int l=0; l<Lsky; l++)
		{
			for(int j=0; j<NIFO; j++)
			{
				int mlptr;
				mlptr = j*Lsky;
				post_gpu_data[i].other_data.ml_mm[mlptr + l] = ml[j][l];
			}
			post_gpu_data[i].other_data.ml_mm[NIFO*Lsky + l] = mm[l];
		}
		post_gpu_data[i].other_data.T_En = En;
		post_gpu_data[i].other_data.T_Es = Es;
		post_gpu_data[i].other_data.TH = TH;
		post_gpu_data[i].other_data.le = Lsky - 1;
		post_gpu_data[i].other_data.lag = lag;
		post_gpu_data[i].other_data.nIFO = nIFO;
	}
	for(int l=0; l<Lsky; l++)
	{
		for(int j=0; j<NIFO; j++)
		{
			int mlptr;
			mlptr = j*Lsky;
			pre_gpu_data[0].other_data.ml_mm[mlptr + l] = ml[j][l];	
		}
		pre_gpu_data[0].other_data.ml_mm[NIFO*Lsky+ l] = mm[l];
	}
	cudaMemcpyAsync(skyloop_other[0].ml_mm, pre_gpu_data[0].other_data.ml_mm, (1 + NIFO) * Lsky * sizeof(short), cudaMemcpyHostToDevice, stream[0] );
	finish[0] = clock();
	d[3] += (double)(finish[0] - start[0])/CLOCKS_PER_SEC;
//++++++++++++++++++++++++++++++++
// loop over cluster
//++++++++++++++++++++++++++++++++
   	cid = pwc->get((char*)"ID",  0,'S',0);                 // get cluster ID
   	K = cid.size();                                                         
	
	start[1] = clock();
	//cout<<"2"<<endl;
	for(k=0; k<K; k++)				// loop over clusters
	{
		if(!V_array[k])	continue;
		start[2] = clock();
		if(k>=constptr)
		{
			if((K-constptr)>CONSTANT_SIZE)
			{
				cudaMemcpyToSymbol(constV, V_array + constptr, sizeof(size_t) * CONSTANT_SIZE);
				cudaMemcpyToSymbol(consttsize, tsize_array + constptr, sizeof(size_t) * CONSTANT_SIZE);	
				constptr += CONSTANT_SIZE;
			}
			else
			{
				cudaMemcpyToSymbol(constV, V_array + constptr, sizeof(size_t) * (K-constptr));
				cudaMemcpyToSymbol(consttsize, tsize_array + constptr, sizeof(size_t) * (K-constptr));	
			}
		}
		id = size_t(cid.data[k]+0.1);
		pI = net->wdmMRA.getXTalk(pwc, id);
		V = V_array[k];
		tsize = tsize_array[k];
		V4 = V + (V%4 ? 4 - V%4 : 0);     
		//cout<<"3"<<endl;
     		std::vector<wavearray<float> > vtd;     // vectors of TD energies  
     		std::vector<wavearray<float> > vTD;     // vectors of TD energies  
     		std::vector<wavearray<float> > eTD;     // vectors of TD energies  
		wavearray<float> tmp(tsize*V4); tmp=0;  // aligned array for TD amplitude	
	
		for(i=0; i<NIFO; i++)
		{
			vtd.push_back(tmp);
			vTD.push_back(tmp);
			eTD.push_back(tmp);					// array of aligned energy vectors
		}
		for(i=0; i<NIFO; i++)
		{
			pa[i] = vtd[i].data + (tsize/2)*V4;
			pA[i] = vTD[i].data + (tsize/2)*V4;
		}

		net->pList.clear();
		for( j=0; j<V; j++)
		{  
			pix = pwc->getPixel(id,pI[j]);          // get pixel pointer   
         		net->pList.push_back(pix);      // store pixel pointers for MRA
			for(i=0; i<nIFO; i++) {
            			for( l=0; l<tsize; l++) 
						{                                
              			   aa = pix->tdAmp[i].data[l];             // copy TD 00 data 
		                   AA = pix->tdAmp[i].data[l+tsize];       // copy TD 90 data 
				   vtd[i].data[l*V4+j] = aa;
				   vTD[i].data[l*V4+j] = AA;
					// assign the data 
				   			if(alloced_gpu<BufferNum)
				   			{
								int etddim = V4 * tsize;
				   				pre_gpu_data[alloced_gpu].other_data.eTD[i*etddim + l*V4+j] = aa*aa+AA*AA;
								if(i == nIFO - 1 && NIFO > nIFO)
									for(int I = nIFO; I<NIFO; I++)
					   					pre_gpu_data[alloced_gpu].other_data.eTD[I*etddim + l*V4+j] = 0;
				   			}
            			}
			}
		}
		finish[2] = clock();
		d[5] += (double)(finish[2] - start[2])/CLOCKS_PER_SEC;
		/*int etddim = V4 * tsize;
		FILE *fpt1 = fopen("skyloop_myeTD", "a");
		for(int l=0; l<etddim; l++)
            fprintf(fpt1,"%f %f %f %f\n", pre_gpu_data[alloced_gpu].other_data.eTD[l], pre_gpu_data[alloced_gpu].other_data.eTD[etddim + l], pre_gpu_data[alloced_gpu].other_data.eTD[2*etddim + l], pre_gpu_data[alloced_gpu].other_data.eTD[3*etddim + l]);
		fclose(fpt1);*/

//++++++++++++++++++++++++++++++++
// assign the data 
//++++++++++++++++++++++++++++++++
		if(alloced_gpu < BufferNum)
		{
			start[3] = clock();
			
			int i = alloced_gpu;
			
			if(i<StreamNum)
			{
				post_gpu_data[i].other_data.stream = i;
				post_gpu_data[i].other_data.id = id;
				post_gpu_data[i].other_data.V = V;
				post_gpu_data[i].other_data.V4 = V4;
				post_gpu_data[i].other_data.tsize = tsize;
				post_gpu_data[i].other_data.k = k;
				k_array[i] = k;
			}
			alloced_gpu++;
			if(alloced_gpu == StreamNum)		// if all streams' data have been assigned
			{
				push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, V_array, tsize_array, k_array, Lsky, StreamNum, stream);
				for(int i=0; i<StreamNum; i++)				// wait for all commands in the stream to complete
					CUDA_CHECK(cudaStreamSynchronize(stream[i]));
				alloced_gpu = 0;
			}
			
			finish[3] =  clock();
			d[6] += (double)(finish[3] - start[3])/CLOCKS_PER_SEC;
		 }
	}							// end of loop
	if(alloced_gpu != 0)		// if there are some clusters waiting for GPU calculation
	{	
		push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, V_array, tsize_array, k_array, Lsky, alloced_gpu, stream);
		for(int i=0; i<alloced_gpu; i++)				// wait for all commands in the stream to complete
			CUDA_CHECK(cudaStreamSynchronize(stream[i]));
		alloced_gpu = 0;
	}
	finish[1] = clock();
	d[4] += (double)(finish[1] - start[1])/CLOCKS_PER_SEC;
	cleanup_cpu_mem(pre_gpu_data, post_gpu_data, stream);
	cleanup_gpu_mem(skyloop_output, skyloop_other, stream);
	for(int i=0; i<StreamNum; i++)	
		cudaStreamDestroy(stream[i]);
	for(int i=0; i<StreamNum; i++)
		count += streamCount[i];
//	cout<<"count = "<<count<<endl;
	for(i=0; i<3; i++)
		d[i] = gpu_d[i];
	return count;
}

__host__ void push_work_into_gpu(struct pre_data *input_data, struct post_data *post_gpu_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, size_t *V_array, size_t *tsize_array, size_t *k_array, int Lsky, int work_size, cudaStream_t *stream)
{
	size_t V, V4;
	int etddim;
	int k;
	for(int i=0; i<work_size; i++)// transfer the data from CPU to GPU
	{
		k = k_array[i];
		V = V_array[k];
		V4 = V + (V%4 ? 4 - V%4 : 0);
        etddim = tsize_array[k] * V4;
		cudaMemcpyAsync(skyloop_other[i].eTD, input_data[i].other_data.eTD, NIFO * etddim * sizeof(float), cudaMemcpyHostToDevice, stream[i] );
	}

	for(int i=0; i<work_size; i++)// call for gpu caculation
		kernel_skyloop<<<num_blocks, num_threads, shared_memory_usage, stream[i]>>>(skyloop_other[i].eTD, skyloop_other[0].ml_mm, skyloop_output[i].output, Lsky, k_array[i]);

	for(int i=0; i<work_size; i++)// transfer the data back from GPU to CPU
	{
		k = k_array[i];
		V = V_array[k];
		V4 =  V4 = V + (V%4 ? 4 - V%4 : 0);
        cudaMemcpyAsync(post_gpu_data[i].output.output, skyloop_output[i].output, Lsky * V4 * sizeof(float) + Lsky * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
	}
	for(int i=0; i<work_size; i++)
		cudaStreamAddCallback(stream[i], MyCallback, (void*)&post_gpu_data[i], 0);
//	cout<<"Push work into gpu success."<<endl;
}

__global__ void kernel_skyloop(float *eTD, short *ml_mm, float *gpu_output, int Lsky, size_t k) 
{
	const int grid_size = blockDim.x * gridDim.x;
	int l = blockIdx.x * blockDim.x + threadIdx.x;
	float *pe[NIFO];
	short *ml[NIFO];
	short *mm;
	int msk;	
	size_t V, V4, tsize;
	int le = Lsky - 1;

	V = constV[k];
	tsize = consttsize[k];
	msk = V%4;
	msk = (msk>0);
	V4 = V + msk*(4-V%4);

	pe[0] = eTD;
	pe[1] = eTD + V4*tsize;
	pe[2] = eTD + 2*V4*tsize;
	pe[3] = eTD + 3*V4*tsize;
	ml[0] = ml_mm;
	ml[1] = ml_mm + Lsky;
	ml[2] = ml_mm + 2*Lsky;
	ml[3] = ml_mm + 3*Lsky;
	mm = ml_mm + 4*Lsky;

	for(; l<=le; l+=grid_size)		// loop over sky locations
	{
		if(!mm[l]) continue;		// skip delay configurations
		// _sse_point_ps 
		pe[0] = pe[0] + (tsize/2)*V4;
		pe[1] = pe[1] + (tsize/2)*V4;
		pe[2] = pe[2] + (tsize/2)*V4;
		pe[3] = pe[3] + (tsize/2)*V4;

		pe[0] = pe[0] + ml[0][l] * (int)V4;
		pe[1] = pe[1] + ml[1][l] * (int)V4;
		pe[2] = pe[2] + ml[2][l] * (int)V4;
		pe[3] = pe[3] + ml[3][l] * (int)V4;
		// inner skyloop
		kernel_skyloop_calculate(pe[0], pe[1], pe[2], pe[3], V, V4, V4*Lsky, gpu_output, l);
	}
		
}

__inline__ __device__ void kernel_skyloop_calculate(float *PE_0, float *PE_1, float *PE_2, float *PE_3, size_t V, size_t V4, size_t rEDim, float *gpu_output,  int l) 
{
	float msk;						// mask
	size_t v = 0;					// indicate the pixel
	size_t ptr;						// indicate the location 
	float pe[NIFO];
	float _Eo[4], _Es[4], _En[4];
	float En, Es, Eo, aa;
	int Mm;
	float rE;						// energy array rNRG.data 
	float pE;						// energy array pNRG.data
	int count;
	
	Mm = 0;							// # of pixels above the threshold

	for(count=0; count<4; count++)
	{
		_Eo[count] = 0;
		_Es[count] = 0;
		_En[count] = 0;
	}
	
	ptr = l*V4;
	count = 0;
	while( v<V )					// loop over selected pixels	
	{
		// *_rE = _sse_sum_ps(_pe);
		pe[0] = PE_0[v];
		pe[1] = PE_1[v];
		pe[2] = PE_2[v];
		pe[3] = PE_3[v];
		rE = pe[0] + pe[1] + pe[2] + pe[3];								// get pixel energy
		//assign the value to the local memory
		gpu_output[ptr+v] = rE;
      	// E>En  0/1 mask
		msk = ( rE>=constEn );										// E>En  0/1 mask
		Mm += msk;												// count pixels above threshold
		///*new
		pE = rE * msk;											// zero sub-threshold pixels
		_Eo[count] += pE;
		//Eo += pE;												// network energy
		pE = kernel_minSNE_ps(pE, pe);						// subnetwork energy
		_Es[count] += pE;
		//Es += pE;												// subnetwork energy
		msk = ( pE>=constEs );										// subnet energy > Es 0/1 mask
		rE *= msk;											   
		_En[count] += rE;
		// assign the value to the local memory
		//new*/
		v++;	
		count++;
		count = count%4;
	}

	En = _En[0] + _En[1] + _En[2] + _En[3];			// Write back to output
	Eo = _Eo[0] + _Eo[1] + _Eo[2] + _Eo[3] + 0.01;
	Es = _Es[0] + _Es[1] + _Es[2] + _Es[3];
	Mm = Mm * 2 + 0.01;
	aa = Es*En/(Eo-Es);
	
	msk = ((aa-Mm)/(aa+Mm)<0.33);
	aa = aa*(1-msk) + (-1)*msk;
	gpu_output[rEDim + l] = aa;
	
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
	ab = ( a>=b );											// if a>=b, ab 1/0
	ac = ( a>=c );											// if a>=c, ac 1/0
	ad = ( a>=d );											// if a>=d, ad 1/0
	bc = ( b>=c );											// if b>=c, bc 1/0
	bd = ( b>=d );											// if b>=d, bd 1/0
	cd = ( c>=d );											// if c>=d, cd 1/0
	 
	temp = a+b+c+d - ab*ac*ad*a - (1-ab)*bc*bd*b - (1-ac)*(1-bc)*cd*c - (1-ad)*(1-bd)*(1-cd)*d;
	flag = ( temp>=pE );										// if temp>=pE, flag 1/0
	temp = (temp - flag*temp) + (pE - (1-flag)*pE);
	return temp;
}
 
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void* post_gpu_data)
{
	Callback(post_gpu_data, gpu_net, gpu_hist, pwc, FP, FX, pa, pA, streamCount, gpu_d);
}

void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int eTDDim, int V4max, int Lsky)// allocate locked memory on CPU 
{
	for(int i = 0; i<BufferNum; i++)
	{	

		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.eTD), NIFO * eTDDim * sizeof(float), cudaHostAllocMapped ) );
	}
	CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.ml_mm), (1 + NIFO) * Lsky * sizeof(short), cudaHostAllocMapped ) );
	for(int i = 0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.output), Lsky * V4max * sizeof(float) + Lsky * sizeof(float), cudaHostAllocMapped ) );
		post_gpu_data[i].other_data.ml_mm = (short*)malloc(sizeof(size_t) * (1 + NIFO) * Lsky);
	}
		return;
}

void cleanup_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, cudaStream_t *stream)
{
 	for(int i = 0; i<BufferNum; i++)
	{
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.eTD));
	}		
	CUDA_CHECK(cudaFreeHost(pre_gpu_data[0].other_data.ml_mm));
	for(int i=0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].output.output));
		free(post_gpu_data[i].other_data.ml_mm);
	}
	return;
}

void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int eTDDim, int V4max, int Lsky)// allocate the memory on GPU
{
	for(int i = 0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].eTD), NIFO * eTDDim * sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_output[i].output), Lsky * V4max * sizeof(float) + Lsky * sizeof(float) ) );
	}
	CUDA_CHECK(cudaMalloc(&(skyloop_other[0].ml_mm), (1 + NIFO) * Lsky * sizeof(short) ) );	
}

void cleanup_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, cudaStream_t *stream)// cleanup the memory on GPU
{
	for(int i = 0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaFree(skyloop_other[i].eTD) );
		CUDA_CHECK(cudaFree(skyloop_output[i].output) );
	}
	CUDA_CHECK(cudaFree(skyloop_other[0].ml_mm) );
	return;
}


