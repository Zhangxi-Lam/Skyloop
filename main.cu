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

#define TSize 1000
#define StreamNum 4
#define BufferNum 4  

network *gpu_net;
TH2F *gpu_hist;
netcluster *pwc;
double *FP[NIFO];
double *FX[NIFO];
bool finish[StreamNum];
size_t gpu_nIFO;
#define CUDA_CHECK(value) {                                             \
    cudaError_t _m_cudaStat = value;                                    \
    if (_m_cudaStat != cudaSuccess) {                                   \
        fprintf(stderr, "Error %s at line %d in file %s\n",             \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
        exit(1); }}

inline void gpu_pnt_(float** q, float** p, short** m, int l, int n); 
inline void gpu_cpp_(float*& a, float** p);
inline void gpu_cpf_(float*& a, double** p, size_t i); //GV
extern long Callback(void *post_gpu_data, network *gpu_net, netcluster *pwc);
long gpu_subNetCut(network *net, int lag, float snc, TH2F *hist)
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
	
	float diff = 0;					// indicate the differences

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

//++++++++++++++++++++++++++++++++
// find out the maximum of V and tsize 
//++++++++++++++++++++++++++++++++

   	cid = pwc->get((char*)"ID",  0,'S',0);                  // get cluster ID

	K = cid.size();
	for(k=0; k<K; k++)				// loop over clusters
	{
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
		if( tsize > Tmax )
			Tmax = tsize;
		if( V4 > V4max )
			V4max = V4;
	}
	cout<<"new main.cu inside gpu_subNetCut"<<endl;
	cout<<"V4max = "<<V4max<<" Tmax = "<<Tmax<<endl;

//++++++++++++++++++++++++++++++++
// declare the variables used for gpu calculation 
//++++++++++++++++++++++++++++++++
	struct pre_data pre_gpu_data[BufferNum];	// store the data before gpu cal
	struct post_data post_gpu_data[StreamNum];	// store the data transfer from gpu
	struct skyloop_output skyloop_output[StreamNum];// store the skyloop_output data
	struct other skyloop_other[StreamNum];		// store the data which is not output
	
	size_t streamCount[StreamNum];	// the result of each stream
//	bool finish[StreamNum];			// indicate whether the stream is finished
	int eTDDim = 0;					// the size of each eTD
	int alloced_gpu = 0;				// the number of gpu which has been allocated data
	
	eTDDim = Tmax * V4max;
	for(int i=0; i<StreamNum; i++)
	{
		streamCount[i] = i;
		finish[i] = true;		
	}
	// allocate the memory on cpu and gpu
	allocate_cpu_mem(pre_gpu_data, post_gpu_data, eTDDim, V4max, Lsky);
	allocate_gpu_mem(skyloop_output, skyloop_other, eTDDim, V4max, Lsky);
	
	gpu_net = net;
	gpu_hist = hist;
	gpu_nIFO = nIFO;
	cout<<"XIFO = "<<XIFO<<endl;
	cout<<"En = "<<En<<endl;
	cudaStream_t stream[StreamNum];			// define the stream
	for(int i=0; i<StreamNum; i++)			
		CUDA_CHECK(cudaStreamCreate(&stream[i]));	// create the new stream
		
	for(int i=0; i<BufferNum; i++)		// initialize the data
	{
		*(pre_gpu_data[i].other_data.T_En) = En;
		*(pre_gpu_data[i].other_data.T_Es) = Es; 
		*(pre_gpu_data[i].other_data.TH) = TH;
		*(pre_gpu_data[i].other_data.le) = Lsky - 1;
		*(pre_gpu_data[i].other_data.lag) = lag;
		*(pre_gpu_data[i].other_data.nIFO) = nIFO;
		if( i<StreamNum )
		{
			for(int l=0; l<Lsky; l++)
			{
				for(int j=0; j<nIFO; j++)
				{
					pre_gpu_data[i].other_data.ml[j][l] = ml[j][l];
//					post_gpu_data[i].other_data.ml[j][l] = ml[j][l];
				}
				pre_gpu_data[i].other_data.mm[l] = mm[l];
//				post_gpu_data[i].other_data.mm[l] = mm[l];
			}
			*(post_gpu_data[i].other_data.T_En) = En;
			*(post_gpu_data[i].other_data.T_Es) = Es;
			*(post_gpu_data[i].other_data.TH) = TH;
			*(post_gpu_data[i].other_data.le) = Lsky - 1;
			*(post_gpu_data[i].other_data.lag) = lag;
			*(post_gpu_data[i].other_data.nIFO) = nIFO;
		}
		else 
		{
			for(int l=0; l<Lsky; l++)
			{
				for(int j=0; j<nIFO; j++)
					pre_gpu_data[i].other_data.ml[j][l] = ml[j][l];
				pre_gpu_data[i].other_data.mm[l] = mm[l];
			}
		}
				
	}
//++++++++++++++++++++++++++++++++
// loop over cluster
//++++++++++++++++++++++++++++++++
   	cid = pwc->get((char*)"ID",  0,'S',0);                 // get cluster ID
   	K = cid.size();                                                         
	for(k=0; k<K; k++)				// loop over clusters
	{
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
	
     	std::vector<wavearray<float> > eTD;     // vectors of TD energies  
		wavearray<float> tmp(tsize*V4); tmp=0;  // aligned array for TD amplitude	
	
		for(i=0; i<NIFO; i++)
			eTD.push_back(tmp);					// array of aligned energy vectors

		net->pList.clear();
		for( j=0; j<V; j++)
		{   
			pix = pwc->getPixel(id,pI[j]);          // get pixel pointer   
         		net->pList.push_back(pix);      // store pixel pointers for MRA
			for(i=0; i<nIFO; i++) {
            			for( l=0; l<tsize; l++) {                                
              			   aa = pix->tdAmp[i].data[l];             // copy TD 00 data 
		                   AA = pix->tdAmp[i].data[l+tsize];       // copy TD 90 data 
		                   eTD[i].data[l*V4+j] = aa*aa+AA*AA;      // copy power      
					// assign the data 
							if(alloced_gpu<BufferNum)
				           		pre_gpu_data[alloced_gpu].other_data.eTD[i][l*V4+j] = aa*aa+AA*AA;
            			}
			}
		}
		/*if(alloced_gpu<StreamNum)
		{
			FILE *fpt = fopen("skyloop_before", "a");	
			for(int i=0; i<eTDDim; i++)
				fprintf(fpt, "k = %d, l = %d, eTD[0] = %f eTD[1] = %f eTD[2] = %f\n", k, i, pre_gpu_data[alloced_gpu].other_data.eTD[0][i], pre_gpu_data[alloced_gpu].other_data.eTD[1][i],pre_gpu_data[alloced_gpu].other_data.eTD[2][i]);
			for(int i=0; i<Lsky; i++)
				fprintf(fpt, "k = %d, l = %d, ml[0] = %hd ml[1] = %hd ml[2] = %hd\n", k, i, ml[0][i],  ml[1][i], ml[2][i]);
			fclose(fpt);
				
		}*/
//++++++++++++++++++++++++++++++++
// assign the data 
//++++++++++++++++++++++++++++++++
		if(alloced_gpu < BufferNum)
		{
			int i = alloced_gpu;
			finish[i] = false;
			*(pre_gpu_data[i].other_data.id) = id;
			*(pre_gpu_data[i].other_data.V) = V;
			*(pre_gpu_data[i].other_data.V4) = V4;
			*(pre_gpu_data[i].other_data.tsize) = tsize;
			*(pre_gpu_data[i].other_data.count) = k;
			*(pre_gpu_data[i].other_data.finish) = finish[i];
			
			if(i<StreamNum)
			{
				*(post_gpu_data[i].other_data.id) = id;
				*(post_gpu_data[i].other_data.V) = V;
				*(post_gpu_data[i].other_data.V4) = V4;
				*(post_gpu_data[i].other_data.tsize) = tsize;
				*(post_gpu_data[i].other_data.count) = k;
				*(post_gpu_data[i].other_data.finish) = finish[i];
			}
			alloced_gpu++;
			if(alloced_gpu == StreamNum)		// if all streams' data have been assigned
			{
				push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, eTDDim, V4max, Lsky, StreamNum, stream);
				for(int i=0; i<StreamNum; i++)				// wait for all commands in the stream to complete
					CUDA_CHECK(cudaStreamSynchronize(stream[i]));
				alloced_gpu = 0;
			}

			
			
		}
			
	}							// end of loop
	cout<<"here"<<endl;
	cleanup_cpu_mem(pre_gpu_data, post_gpu_data, stream);
	cleanup_gpu_mem(skyloop_output, skyloop_other, stream);
	for(int i=0; i<StreamNum; i++)	
		cudaStreamDestroy(stream[i]);
	return count;
}

void test_function(void)
{
	cout<<"test function's nIFO = "<<gpu_net->ifoListSize()<<endl;
	return;
}
void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int eTDDim, int V4max, int Lsky)// allocate locked memory on CPU 
{
	for(int i = 0; i<BufferNum; i++)
	{
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.eTD[0]), eTDDim * sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.eTD[1]), eTDDim * sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.eTD[2]), eTDDim * sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.ml[0]), Lsky * sizeof(short), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.ml[1]), Lsky * sizeof(short), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.ml[2]), Lsky * sizeof(short), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.mm), Lsky * sizeof(short), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.T_En), sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.T_Es), sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.TH), sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.le), sizeof(int), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.lag), sizeof(int), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.id), sizeof(size_t), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.nIFO), sizeof(size_t), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.V), sizeof(size_t), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.V4), sizeof(size_t), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.tsize), sizeof(size_t), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.count), sizeof(size_t), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.finish), sizeof(bool), cudaHostAllocMapped ) );
	}
	for(int i = 0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.rE), Lsky * V4max * sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.pE), Lsky * V4max * sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.Eo), Lsky * sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.En), Lsky * sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.Es), Lsky * sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.Mm), Lsky * sizeof(int), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.eTD[0]), eTDDim * sizeof(float), cudaHostAllocMapped ) );
	    CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.eTD[1]), eTDDim * sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.eTD[2]), eTDDim * sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.ml[0]), Lsky * sizeof(short), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.ml[1]), Lsky * sizeof(short), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.ml[2]), Lsky * sizeof(short), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.mm), Lsky * sizeof(short), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.T_En), sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.T_Es), sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.TH), sizeof(float), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.le), sizeof(int), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.lag), sizeof(int), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.id), sizeof(size_t), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.nIFO), sizeof(size_t), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.V), sizeof(size_t), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.V4), sizeof(size_t), cudaHostAllocMapped ) );
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.tsize), sizeof(size_t), cudaHostAllocMapped ) );
	        CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.count), sizeof(size_t), cudaHostAllocMapped ) );
        	CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].other_data.finish), sizeof(bool), cudaHostAllocMapped ) );
		cout<<" alloc cpu"<<endl;
	}
	
		return;
}

void cleanup_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, cudaStream_t *stream)
{
 	for(int i = 0; i<BufferNum; i++)
	{	
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.eTD[0]));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.eTD[1]));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.eTD[2]));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.ml[0]));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.ml[1]));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.ml[2]));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.mm));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.T_En));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.T_Es));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.TH));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.le));
	        CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.lag));
	        CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.id));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.nIFO));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.V));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.V4));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.tsize));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.count));
		CUDA_CHECK(cudaFreeHost(pre_gpu_data[i].other_data.finish));
	}		
	for(int i=0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].output.rE));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].output.pE));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].output.Eo));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].output.En));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].output.Es));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].output.Mm));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.eTD[0]));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.eTD[1]));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.eTD[2]));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.ml[0]));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.ml[1]));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.ml[2]));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.mm));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.T_En));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.T_Es));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.TH));
        	CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.le));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.lag));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.id));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.nIFO));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.V));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.V4));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.tsize));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.count));
		CUDA_CHECK(cudaFreeHost(post_gpu_data[i].other_data.finish));
		cout<<"cleanup cpu"<<endl;
	}
	return;
}

void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int eTDDim, int V4max, int Lsky)// allocate the memory on GPU
{
	for(int i = 0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaMalloc(&(skyloop_output[i].rE), Lsky * V4max * sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_output[i].pE), Lsky * V4max * sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_output[i].Eo), Lsky * sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_output[i].En), Lsky * sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_output[i].Es), Lsky * sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_output[i].Mm), Lsky * sizeof(int) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].eTD[0]), eTDDim * sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].eTD[1]), eTDDim * sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].eTD[2]), eTDDim * sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].ml[0]), Lsky * sizeof(short) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].ml[1]), Lsky * sizeof(short) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].ml[2]), Lsky * sizeof(short) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].mm), Lsky * sizeof(short) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].T_En), sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].T_Es), sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].TH), sizeof(float) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].le), sizeof(int) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].lag), sizeof(int) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].id), sizeof(size_t) ) );
	   	CUDA_CHECK(cudaMalloc(&(skyloop_other[i].nIFO), sizeof(size_t) ) );
       		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].V), sizeof(size_t) ) );
	   	CUDA_CHECK(cudaMalloc(&(skyloop_other[i].V4), sizeof(size_t) ) );
	   	CUDA_CHECK(cudaMalloc(&(skyloop_other[i].tsize), sizeof(size_t) ) );
	     	CUDA_CHECK(cudaMalloc(&(skyloop_other[i].count), sizeof(size_t) ) );
		CUDA_CHECK(cudaMalloc(&(skyloop_other[i].finish), sizeof(bool) ) );
		cout<<"alloc gpu"<<endl;
	}
}

void cleanup_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, cudaStream_t *stream)// cleanup the memory on GPU
{
	for(int i = 0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaFree(skyloop_output[i].rE) );
		CUDA_CHECK(cudaFree(skyloop_output[i].pE) );
		CUDA_CHECK(cudaFree(skyloop_output[i].Eo) );
		CUDA_CHECK(cudaFree(skyloop_output[i].En) );
		CUDA_CHECK(cudaFree(skyloop_output[i].Es) );
		CUDA_CHECK(cudaFree(skyloop_output[i].Mm) );
		CUDA_CHECK(cudaFree(skyloop_other[i].eTD[0]) );
		CUDA_CHECK(cudaFree(skyloop_other[i].eTD[1]) );
		CUDA_CHECK(cudaFree(skyloop_other[i].eTD[2]) );
		CUDA_CHECK(cudaFree(skyloop_other[i].ml[0]) );
		CUDA_CHECK(cudaFree(skyloop_other[i].ml[1]) );
		CUDA_CHECK(cudaFree(skyloop_other[i].ml[2]) );
		CUDA_CHECK(cudaFree(skyloop_other[i].mm) );
		CUDA_CHECK(cudaFree(skyloop_other[i].T_En) );
		CUDA_CHECK(cudaFree(skyloop_other[i].T_Es) );
		CUDA_CHECK(cudaFree(skyloop_other[i].TH) );
		CUDA_CHECK(cudaFree(skyloop_other[i].le) );
		CUDA_CHECK(cudaFree(skyloop_other[i].lag) );
		CUDA_CHECK(cudaFree(skyloop_other[i].id) );
		CUDA_CHECK(cudaFree(skyloop_other[i].nIFO) );
		CUDA_CHECK(cudaFree(skyloop_other[i].V) );
		CUDA_CHECK(cudaFree(skyloop_other[i].V4) );
		CUDA_CHECK(cudaFree(skyloop_other[i].tsize) );
		CUDA_CHECK(cudaFree(skyloop_other[i].count) );
		CUDA_CHECK(cudaFree(skyloop_other[i].finish) );
		cout<<"cleanup gpu"<<endl;
	}
	return;
}

__host__ void push_work_into_gpu(struct pre_data *input_data, struct post_data *post_gpu_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, int eTDDim, int V4max, int Lsky, int work_size, cudaStream_t *stream)
{
	for(int i=0; i<work_size; i++)// transfer the data from CPU to GPU
	{
		for(int j = 0; j<gpu_nIFO ; j++)
		{
			cudaMemcpyAsync(skyloop_other[i].eTD[j], input_data[i].other_data.eTD[j], eTDDim * sizeof(float), cudaMemcpyHostToDevice, stream[i] );
			cudaMemcpyAsync(skyloop_other[i].ml[j], input_data[i].other_data.ml[j], Lsky * sizeof(short), cudaMemcpyHostToDevice, stream[i] );
		}
		//
		FILE *fpt = fopen("skyloop_before","a");
		for(int k=0; k<eTDDim; k++)
			fprintf(fpt, "k = %d, l = %d, eTD[0] = %f eTD[1] = %f eTD[2] = %f\n", i, k, input_data[i].other_data.eTD[0][k], input_data[i].other_data.eTD[1][k], input_data[i].other_data.eTD[2][k]);
		fclose(fpt);
		cudaMemcpyAsync(skyloop_other[i].mm, input_data[i].other_data.mm, Lsky * sizeof(short), cudaMemcpyHostToDevice, stream[i] );
		cudaMemcpyAsync(skyloop_other[i].T_En, input_data[i].other_data.T_En, sizeof(float), cudaMemcpyHostToDevice, stream[i] );
		cudaMemcpyAsync(skyloop_other[i].T_Es, input_data[i].other_data.T_Es, sizeof(float), cudaMemcpyHostToDevice, stream[i] );
//		cudaMemcpyAsync(skyloop_other[i].TH, input_data[i].other_data.TH, sizeof(float), cudaMemcpyHostToDevice, stream[i] );
		cudaMemcpyAsync(skyloop_other[i].le, input_data[i].other_data.le, sizeof(int), cudaMemcpyHostToDevice, stream[i] );
//		cudaMemcpyAsync(skyloop_other[i].lag, input_data[i].other_data.lag, sizeof(int), cudaMemcpyHostToDevice, stream[i] );
//		cudaMemcpyAsync(skyloop_other[i].id, input_data[i].other_data.id, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
		cudaMemcpyAsync(skyloop_other[i].nIFO, input_data[i].other_data.nIFO, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
		cudaMemcpyAsync(skyloop_other[i].V, input_data[i].other_data.V, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
		cudaMemcpyAsync(skyloop_other[i].V4, input_data[i].other_data.V4, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
		cudaMemcpyAsync(skyloop_other[i].tsize, input_data[i].other_data.tsize, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
//		cudaMemcpyAsync(skyloop_other[i].count, input_data[i].other_data.count, sizeof(size_t), cudaMemcpyHostToDevice, stream[i] );
//		cudaMemcpyAsync(skyloop_other[i].finish, input_data[i].other_data.finish, sizeof(bool), cudaMemcpyHostToDevice, stream[i] );
	}

	for(int i=0; i<work_size; i++)// call for gpu caculation
		kernel_skyloop<<<num_blocks, num_threads, shared_memory_usage, stream[i]>>>(skyloop_other[i].eTD[0], skyloop_other[i].eTD[1], skyloop_other[i].eTD[2], skyloop_other[i].ml[0], skyloop_other[i].ml[1], skyloop_other[i].ml[2], skyloop_other[i].mm, skyloop_other[i].V, skyloop_other[i].V4, skyloop_other[i].tsize, skyloop_other[i].T_En, skyloop_other[i].T_Es, skyloop_output[i].rE, skyloop_output[i].pE, skyloop_output[i].Eo, skyloop_output[i].En, skyloop_output[i].Es, skyloop_output[i].Mm, Lsky);

	for(int i=0; i<work_size; i++)// transfer the data back from GPU to CPU
	{
		for(int j=0; j<gpu_nIFO; j++)
		{
                        cudaMemcpyAsync(post_gpu_data[i].other_data.eTD[j], skyloop_other[i].eTD[j], eTDDim * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                        cudaMemcpyAsync(post_gpu_data[i].other_data.ml[j], skyloop_other[i].ml[j], Lsky * sizeof(short), cudaMemcpyDeviceToHost, stream[i] );
		}
                cudaMemcpyAsync(post_gpu_data[i].other_data.mm, skyloop_other[i].mm, Lsky * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output.rE, skyloop_output[i].rE, Lsky * V4max * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output.pE, skyloop_output[i].pE, Lsky * V4max * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output.Eo, skyloop_output[i].Eo, Lsky * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output.En, skyloop_output[i].En, Lsky * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output.Es, skyloop_output[i].Es, Lsky * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
                cudaMemcpyAsync(post_gpu_data[i].output.Mm, skyloop_output[i].Mm, Lsky * sizeof(int), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.T_En, skyloop_other[i].T_En, sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.T_Es, skyloop_other[i].T_Es, sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.TH, skyloop_other[i].TH, sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
//               cudaMemcpyAsync(post_gpu_data[i].other_data.le, skyloop_other[i].le, sizeof(int), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.lag, skyloop_other[i].lag, sizeof(int), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.id, skyloop_other[i].id, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.nIFO, skyloop_other[i].nIFO, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.V, skyloop_other[i].V, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.V4, skyloop_other[i].V4, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.tsize, skyloop_other[i].tsize, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.count, skyloop_other[i].count, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
//                cudaMemcpyAsync(post_gpu_data[i].other_data.finish, skyloop_other[i].finish, sizeof(size_t), cudaMemcpyDeviceToHost, stream[i] );
				cudaStreamAddCallback(stream[i], MyCallback, (void*)&post_gpu_data[i], 0);
	}
//	cout<<"Push work into gpu success."<<endl;
}

__global__ void kernel_skyloop(float *eTD_0, float *eTD_1, float *eTD_2, short *ml_0, short *ml_1, short *ml_2, short *gpu_mm, size_t *gpu_V, size_t *gpu_V4, size_t *gpu_tsize, float *gpu_T_En, float *gpu_T_Es, float *gpu_rE, float *gpu_pE, float *gpu_Eo, float *gpu_En, float *gpu_Es, int *gpu_Mm, int Lsky) 
{
	const int grid_size = blockDim.x * gridDim.x;
	int l = blockIdx.x * blockDim.x + threadIdx.x;
	float *pe[NIFO];
	short *ml[NIFO];
	short *mm;
	float T_En, T_Es;								// two threshold
	
	size_t V, V4, tsize;
	int le = Lsky - 1;

	pe[0] = eTD_0;
	pe[1] = eTD_1;
	pe[2] = eTD_2;
	ml[0] = ml_0;
	ml[1] = ml_1;
	ml[2] = ml_2;
	mm = gpu_mm;
	V = *gpu_V;
	V4 = *gpu_V4;
	tsize = *gpu_tsize;
	T_En = *gpu_T_En;
	T_Es = *gpu_T_Es;

	for(; l<=le; l+=grid_size)		// loop over sky locations
	{
		if(!mm[l]) continue;		// skip delay configurations
		// _sse_point_ps 
		pe[0] = pe[0] + (tsize/2)*V4;
		pe[1] = pe[1] + (tsize/2)*V4;
		pe[2] = pe[2] + (tsize/2)*V4;

		pe[0] = pe[0] + ml[0][l] * (int)V4;
		pe[1] = pe[1] + ml[1][l] * (int)V4;
		pe[2] = pe[2] + ml[2][l] * (int)V4;
		// inner skyloop
		kernel_skyloop_calculate(pe[0], pe[1], pe[2], V, V4, T_En, T_Es, gpu_rE, gpu_pE, gpu_Eo, gpu_En, gpu_Es, gpu_Mm, l);
		///*debug	
//		if(l<(tsize*V4))
		
/*		for(int i=0; i<V4; i++)
		{
		gpu_En[l] = pe[0][0]; 
		gpu_Eo[l] = pe[1][0]; 
		gpu_Es[l] = pe[2][0]; 
		gpu_Mm[l] = (tsize/2) * V4 + ml[1][l] * (int)V4; 	
		}
		//debug*/
	}
		
}

__inline__ __device__ void kernel_skyloop_calculate(float *PE_0, float *PE_1, float *PE_2, size_t V, size_t V4, float T_En, float T_Es, float *gpu_rE, float *gpu_pE, float *gpu_Eo, float *gpu_En, float *gpu_Es, int *gpu_Mm, int l) 
{
	float msk;						// mask
	size_t v = 0;					// indicate the pixel
	size_t ptr;						// indicate the location 
	float pe[NIFO];
	float Eo, En, Es;
	int Mm;
	float rE;						// energy array rNRG.data 
	float pE;						// energy array pNRG.data
	
	Eo = 0;							// total network energy
	En = 0;							// network energy above the threshold
	Es = 0;							// subnet energy above the threshold
	Mm = 0;							// # of pixels above the threshold
	
	ptr = l*V4;
	while( v<V )					// loop over selected pixels	
	{
		// *_rE = _sse_sum_ps(_pe);
		pe[0] = PE_0[v];
		pe[1] = PE_1[v];
		pe[2] = PE_2[v];
		rE = pe[0] + pe[1] + pe[2];								// get pixel energy
      	// E>En  0/1 mask
		msk = ( rE>=T_En );										// E>En  0/1 mask
		Mm += msk;												// count pixels above threshold
		///*new
		pE = rE * msk;											// zero sub-threshold pixels
		Eo += pE;												// network energy
		pE = kernel_minSNE_ps(pE, pe, msk);						// subnetwork energy
		Es += pE;												// subnetwork energy
		msk = ( pE>=T_Es );										// subnet energy > Es 0/1 mask
		rE *= msk;												
		En += rE;												// network energy
		// assign the value to the local memory
		gpu_rE[ptr+v] = rE;
		gpu_pE[ptr+v] = pE;
		//new*/
		v++;
	}
	Eo += 0.01;
	Mm = Mm *2 +0.01;

/*debug
	En = pe[0];
	Eo = rE;
	Es = pE;
// debug*/
	gpu_En[l] = En;												// Write back to output
	gpu_Eo[l] = Eo;												
	gpu_Es[l] = Es;
	gpu_Mm[l] = Mm;
	
}
__inline__ __device__ float kernel_minSNE_ps(float pE, float *pe, float msk)
{
	float a, b, c, ab, bc, ac;
	float temp;
	float flag;
	
	a = pe[0];
	b = pe[1];
	c = pe[2];
	ab = ( a>=b );											// if a>=b, ab 1/0
	ac = ( a>=c );											// if a>=c, ac 1/0
	bc = ( b>=c );											// if b>=c, bc 1/0
	temp = a+b+c - ab*ac*a - (1-ab)*bc*b - (1-ac)*(1-bc)*c;
	flag = ( temp>=pE );
	temp = temp + pE - flag*temp - (1-flag)*pE;
	return temp;
} 
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void* post_gpu_data)
{
	cout<<"use extern"<<endl;
	size_t count = 0;
	count =	Callback(post_gpu_data, gpu_net, pwc);
	cout<<"after extern "<<count<<endl;
}
inline void gpu_pnt_(float** q, float** p, short** m, int l, int n) {
	// point 0-7 float pointers to first network pixel
   NETX(q[0] = (p[0] + m[0][l]*n);,
        q[1] = (p[1] + m[1][l]*n);,
        q[2] = (p[2] + m[2][l]*n);,
        q[3] = (p[3] + m[3][l]*n);,
        q[4] = (p[4] + m[4][l]*n);,
        q[5] = (p[5] + m[5][l]*n);,
        q[6] = (p[6] + m[6][l]*n);,
        q[7] = (p[7] + m[7][l]*n);)
      return;
}
inline void gpu_cpp_(float*& a, float** p) 
{
// copy to a data defined by array of pointers p and increment pointer
	NETX(*(a++) = *p[0]++;,
	     *(a++) = *p[1]++;,
	     *(a++) = *p[2]++;,
	     *(a++) = *p[3]++;,
	     *(a++) = *p[4]++;,
	     *(a++) = *p[5]++;,
	     *(a++) = *p[6]++;,
	     *(a++) = *p[7]++;)
	return;
}
inline void gpu_cpf_(float*& a, double** p, size_t i) //GV
{ 
// copy to a data defined by array of pointers p and increment target pointer
	NETX(*(a++) = p[0][i];,
	     *(a++) = p[1][i];,
	     *(a++) = p[2][i];,                             
	     *(a++) = p[3][i];,  
	     *(a++) = p[4][i];,
	     *(a++) = p[5][i];,
	     *(a++) = p[6][i];,
	     *(a++) = p[7][i];)
	return;
}
/*
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
}*/


