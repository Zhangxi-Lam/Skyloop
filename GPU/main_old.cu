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

//inline int _sse_MRA_ps(network *net, float *amp, float *AMP, float Eo, int K);

// debug
int cc = 0;
//
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
	CombineSize = V4max;
//++++++++++++++++++++++++++++++++
// declare the variables used for gpu calculation 
//++++++++++++++++++++++++++++++++
	struct pre_data pre_gpu_data[BufferNum];		
	struct post_data post_gpu_data[StreamNum];      // store the data transfer from gpu
        struct skyloop_output skyloop_output[StreamNum];// store the skyloop_output data
        struct other skyloop_other[StreamNum];          // store the data which is not output

        int eTDDim = 0;                                 // the size of each eTD
	int vDim = 0;
        int alloced_gpu = 0;                            // the number of gpu which has been allocated data

        eTDDim = Tmax * V4max;
	vDim = eTDDim;
        for(int i=0; i<StreamNum; i++)
                streamCount[i] = 0;
	// allocate the memory on cpu and gpu
	allocate_cpu_mem(pre_gpu_data, post_gpu_data, eTDDim, V4max, Lsky, K);
	allocate_gpu_mem(skyloop_output, skyloop_other, eTDDim, V4max, Lsky, K);
	
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
                                int mlptr;
                                mlptr = j*Lsky;
                                post_gpu_data[i].other_data.ml_mm[mlptr + l] = ml[j][l];
				if(i==0)
					pre_gpu_data[0].other_data.ml_mm[mlptr + l] = ml[j][l];
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
	cudaMemcpyAsync(skyloop_other[0].ml_mm, pre_gpu_data[0].other_data.ml_mm, (1 + NIFO) * Lsky * sizeof(short), cudaMemcpyHostToDevice, stream[0] );
	cudaMemcpyAsync(skyloop_other[0].V_tsize, pre_gpu_data[0].other_data.V_tsize, K * 2 * sizeof(size_t), cudaMemcpyHostToDevice, stream[0] );
//++++++++++++++++++++++++++++++++
// loop over cluster
//++++++++++++++++++++++++++++++++
	std::vector<wavearray<float> > vtd;     	// vectors of TD energies  
        std::vector<wavearray<float> > vTD;     	// vectors of TD energies  
        wavearray<float> tmp(StreamNum*Tmax*V4max); tmp=0;	// aligned array for TD amplitude   

	QuickSort(V_array, k_sortArray, 0, kcount-1);
	cid = pwc->get((char*)"ID", 0,'S',0);		// get cluster ID
	K = cid.size();
	
	for(i=0; i<NIFO; i++)
        {
                vtd.push_back(tmp);
                vTD.push_back(tmp);
        }
	alloced_V4 = 0;					// initialize
	etd_ptr = MaxPixel;
	v_ptr = 0;
	pixelCount = 0;
	for(int z=0; z<kcount;)			// loop over unskiped clusters
	{
		while(!CombineFinish && z<kcount)
		{
		k = k_sortArray[z];
		V = V_array[k];
		V4 = V4_array[k];
		tsize = tsize_array[k];
		etddim = V4 * tsize;
		if((alloced_V4+V4) <= V4max && (alloced_V4+V4+pixelCount+1) <= (V4max+1) )
		{
			alloced_V4 += V4;
			id = size_t(cid.data[k]+0.1);
                        pI = net->wdmMRA.getXTalk(pwc, id);

                        for(i=0; i<NIFO; i++)
                        {
                                pa[alloced_gpu][pixelCount][i] = vtd[i].data + (tsize/2)*V4 + v_ptr + alloced_gpu*vDim;
                                pA[alloced_gpu][pixelCount][i] = vTD[i].data + (tsize/2)*V4 + v_ptr + alloced_gpu*vDim;
                        }
			
			for(j=0; j<V; j++)
			{
				pix = pwc->getPixel(id,pI[j]);
				for(i=0; i<nIFO; i++)
                                {
                                        for( l=0; l<tsize; l++)
                                        {
                                                aa = pix->tdAmp[i].data[l];             // copy TD 00 data 
                                                AA = pix->tdAmp[i].data[l+tsize];       // copy TD 90 data 
                                                vtd[i].data[l*V4+j+v_ptr + alloced_gpu*vDim] = aa;
                                                vTD[i].data[l*V4+j+v_ptr + alloced_gpu*vDim] = AA;
                                                // assign the data 
                                                pre_gpu_data[alloced_gpu].other_data.eTD[i*etddim + l*V4+j + etd_ptr] = aa*aa+AA*AA;
                                                if(i == nIFO - 1 && NIFO > nIFO)
                                                	for(int I = nIFO; I<NIFO; I++)
								{
                                                        		pre_gpu_data[alloced_gpu].other_data.eTD[I*etddim + l*V4+j + etd_ptr] = 0;
									if(j==(V-1))
										for(int J=V; J<V4; J++)
                                                        				pre_gpu_data[alloced_gpu].other_data.eTD[I*etddim + l*V4+J + etd_ptr] = 0;
								}
                                        }
                                }
				
			}
/*			if(k == 125)
                        {
                                FILE *fpt = fopen("./debug_files/skyloop_myinput", "a");
				fprintf(fpt, "V = %d V4 = %d tsize = %d\n", V, V4, tsize);
                                for(int l=0; l<etddim; l++)
                                        fprintf(fpt, "lag = %d k = %d l = %d pe[0] = %f pe[1] = %f pe[2] = %f pe[3] = %f\n", k, l, pre_gpu_data[alloced_gpu].other_data.eTD[l + etd_ptr], pre_gpu_data[alloced_gpu].other_data.eTD[1*etddim + l + etd_ptr], pre_gpu_data[alloced_gpu].other_data.eTD[2*etddim + l + etd_ptr], pre_gpu_data[alloced_gpu].other_data.eTD[3*etddim + l + etd_ptr]);
                                fclose(fpt);            
                        }
*/
			i = alloced_gpu;
			etd_ptr += NIFO*etddim;	
			v_ptr += V4*tsize;
			pre_gpu_data[i].other_data.eTD[pixelCount] = k+1;
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
//		if( pixelCount<MaxPixel && alloced_V4<CombineSize )	continue;
		//cout<<"list "<<i<<" gross V4 = "<<alloced_V4<<endl;
		//for(int z=0; z<pixelCount; z++)
                        //cout<<"k = "<<pre_gpu_data[i].other_data.eTD[z]-1<<endl;
		post_gpu_data[i].other_data.stream = i;
		etddim_array[i] = etd_ptr;
		alloced_V4_array[i] = alloced_V4;
		pixel_array[i] = pixelCount;
		alloced_gpu++;
//++++++++++++++++++++++++++++++++
// assign the data 
//++++++++++++++++++++++++++++++++
		if(alloced_gpu == StreamNum)
		{
			push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, alloced_V4_array, etddim_array, Lsky, pixel_array, StreamNum, stream);
			for(int i=0; i<StreamNum; i++)
				CUDA_CHECK(cudaStreamSynchronize(stream[i]));
			MyCallback(post_gpu_data);
			//clear
			alloced_gpu = 0;
			for(int j=0; j<StreamNum; j++)
				for(int i=0; i<pixel_array[j]; i++)
				{
					post_gpu_data[j].other_data.k[i] = 0;
                                        post_gpu_data[j].other_data.V4[i] = 0;
                                        post_gpu_data[j].other_data.tsize[i] = 0;	
				}
			vtd.clear();
			vTD.clear();
			for(int i=0; i<NIFO; i++)
			{
				vtd.push_back(tmp);
				vTD.push_back(tmp);
			}
		}
		// clear
		etd_ptr = MaxPixel;
		v_ptr = 0;	
		pixelCount = 0;
		alloced_V4 = 0;	
		CombineFinish = false;
		
	}
	if(alloced_gpu != 0)
	{
		
		push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, alloced_V4_array, etddim_array, Lsky, pixel_array, StreamNum, stream);
		for(int i=0; i<StreamNum; i++)
			CUDA_CHECK(cudaStreamSynchronize(stream[i]));
		MyCallback(post_gpu_data);
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
	cout<<"net = "<<net->rNRG.data[0]<<endl;
	cout<<"cc = "<<cc<<endl;
	cc = 0;
	return count;
}

__host__ void push_work_into_gpu(struct pre_data *input_data, struct post_data *post_gpu_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, size_t *alloced_V4_array, size_t *etddim_array, int Lsky, int *pixel_array, int work_size, cudaStream_t *stream)
{
        for(int i=0; i<work_size; i++)// transfer the data from CPU to GPU
                cudaMemcpyAsync(skyloop_other[i].eTD, input_data[i].other_data.eTD, etddim_array[i] * sizeof(float), cudaMemcpyHostToDevice, stream[i] );

        for(int i=0; i<work_size; i++)// call for gpu caculation
                kernel_skyloop<<<num_blocks, num_threads, shared_memory_usage, stream[i]>>>(skyloop_other[i].eTD, skyloop_other[0].ml_mm, skyloop_other[0].V_tsize, skyloop_output[i].output, pixel_array[i]);
        for(int i=0; i<work_size; i++)// transfer the data back from GPU to CPU
                cudaMemcpyAsync(post_gpu_data[i].output.output, skyloop_output[i].output, Lsky * alloced_V4_array[i] * sizeof(float) + Lsky * pixel_array[i] * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
        //for(int i=0; i<work_size; i++)
                //cudaStreamAddCallback(stream[i], Callback, (void*)&post_gpu_data[i], 0);
}


__global__ void kernel_skyloop(float *eTD, short *ml_mm, size_t *V_tsize, float *gpu_output, int pixelcount)
{
        const int grid_size = blockDim.x * gridDim.x;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        float *pe[NIFO];
	float *k_array;
        short *ml[NIFO];
        short *mm;
        size_t V, V4, tsize;
        int Lsky = constLsky;
	int l;
	int k;
        int msk;
	int count = 0;				// indicate the pixel
	size_t etd_ptr = MaxPixel;
	size_t output_ptr = 0;
	k_array = eTD;

	k = k_array[count] - 1;
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
		
		for(l=tid; l<Lsky; l+=grid_size)		// loop over sky locations
		{
			if(!mm[l])	continue;	// skip delay configurations
			pe[0] = eTD + etd_ptr;
			pe[1] = eTD + V4*tsize + etd_ptr;
			pe[2] = eTD + 2*V4*tsize + etd_ptr;
			pe[3] = eTD + 3*V4*tsize + etd_ptr;
		/*	if(k == 83)
			{
				gpu_output[tid*4] = pe[0][tid];	
				gpu_output[tid*4+1] = pe[1][tid];	
				gpu_output[tid*4+2] = pe[2][tid];	
				gpu_output[tid*4+3] = pe[3][tid];	
			}*/
			pe[0] = pe[0] + (tsize/2)*V4;
                        pe[1] = pe[1] + (tsize/2)*V4;
                        pe[2] = pe[2] + (tsize/2)*V4;
                        pe[3] = pe[3] + (tsize/2)*V4;
                        pe[0] = pe[0] + ml[0][l] * (int)V4;
                        pe[1] = pe[1] + ml[1][l] * (int)V4;
                        pe[2] = pe[2] + ml[2][l] * (int)V4;
                        pe[3] = pe[3] + ml[3][l] * (int)V4;
			
			kernel_skyloop_calculate(pe[0], pe[1], pe[2], pe[3], V, V4, V4*Lsky, gpu_output, l, output_ptr);
		}
		etd_ptr = etd_ptr + NIFO*V4*tsize;
		output_ptr = output_ptr + V4*Lsky + Lsky;
		count++;
//		gpu_output[tid*2] = count;
//		gpu_output[tid*2+1] = pixelcount;
        }

}

__inline__ __device__ void kernel_skyloop_calculate(float *PE_0, float *PE_1, float *PE_2, float *PE_3, size_t V, size_t V4, size_t rEDim, float *gpu_output,  int l, size_t output_ptr)
{
        int msk;                                              // mask
        size_t v;                                  // indicate the pixel
        size_t ptr;                                                // indicate the location 
        float pe[NIFO];
        float _Eo[4], _Es[4], _En[4];
        float En, Es, Eo, aa;
        int Mm;
        float rE;                                               // energy array rNRG.data 
        float pE;                                               // energy array pNRG.data
        int count;

        Mm = 0;                                                 // # of pixels above the threshold
        for(count=0; count<4; count++)
        {
                _Eo[count] = 0;
                _Es[count] = 0;
                _En[count] = 0;
        }

        count = 0;
        ptr = l*V4 + output_ptr;
        for(v=0; v<V; v++)                                      // loop over selected pixels    
        {
                // *_rE = _sse_sum_ps(_pe);
                pe[0] = PE_0[v];
                pe[1] = PE_1[v];
                pe[2] = PE_2[v];
                pe[3] = PE_3[v];
                rE = pe[0] + pe[1] + pe[2] + pe[3];                                                             // get pixel energy
                //assign the value to the local memory
                gpu_output[ptr + v] = rE;
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
        //aa = aa*(1-msk) - 1*msk;
        //gpu_output[rEDim + l + output_ptr] = aa;

	if(msk)	return;
	
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
			after_skyloop((void*)&post_gpu_data[i], gpu_net, gpu_hist, pwc, FP, FX, pa[streamNum][pixelcount], pA[streamNum][pixelcount], pixelcount, output_ptr, Lsky, gpu_time, streamCount, cc);
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
		after_skyloop(post_gpu_data, gpu_net, gpu_hist, pwc, FP, FX, pa[streamNum][pixelcount], pA[streamNum][pixelcount], pixelcount, output_ptr, Lsky, gpu_time, streamCount, cc);
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
void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int eTDDim, int V4max, int Lsky, size_t K)// allocate locked memory on CPU 
{
        for(int i = 0; i<BufferNum; i++)
        {
                CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.eTD), NIFO * eTDDim * sizeof(float) + MaxPixel * sizeof(float), cudaHostAllocMapped ) );
        }
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.ml_mm), (1 + NIFO) * Lsky * sizeof(short), cudaHostAllocMapped ) );
        CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.V_tsize), K * 2 * sizeof(size_t), cudaHostAllocMapped ) );
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
        CUDA_CHECK(cudaFreeHost(pre_gpu_data[0].other_data.V_tsize));
        for(int i=0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaFreeHost(post_gpu_data[i].output.output));
                free(post_gpu_data[i].other_data.ml_mm);
        }
        return;
}
void allocate_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, int eTDDim, int V4max, int Lsky, size_t K)// allocate the memory on GPU
{
        for(int i = 0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].eTD), NIFO * eTDDim * sizeof(float) + MaxPixel * sizeof(float) ) );
                CUDA_CHECK(cudaMalloc(&(skyloop_output[i].output), Lsky * V4max * sizeof(float) + Lsky * sizeof(float) ) );
        }
        CUDA_CHECK(cudaMalloc(&(skyloop_other[0].ml_mm), (1 + NIFO) * Lsky * sizeof(short) ) );
        CUDA_CHECK(cudaMalloc(&(skyloop_other[0].V_tsize), K * 2 * sizeof(size_t) ) );
	
}
void cleanup_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, cudaStream_t *stream)// cleanup the memory on GPU
{
        for(int i = 0; i<StreamNum; i++)
        {
                CUDA_CHECK(cudaFree(skyloop_other[i].eTD) );
                CUDA_CHECK(cudaFree(skyloop_output[i].output) );
        }
        CUDA_CHECK(cudaFree(skyloop_other[0].ml_mm) );
        CUDA_CHECK(cudaFree(skyloop_other[0].V_tsize) );
        return;
}

/*void copyToSymbol(size_t *V_array, size_t *tsize_array, size_t &const_ptr, size_t size)
{
	size_t *V_ptr, *tsize_ptr;
	v_ptr = V_array + const_ptr;
	tsize_ptr = tsize_array + const_ptr;
	cudaMemcpyToSymbol(constV, V_ptr, sizeof(size_t) * size);
	cudaMemcpyToSymbol(consttsize, tsize_ptr, sizeof(size_t) * size);
	const_ptr += size;
}*/

