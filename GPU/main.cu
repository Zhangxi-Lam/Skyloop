#include "main.cuh"
#include "gpu_network.hh"
#include "wavearray.hh"
#include <xmmintrin.h>
#include "/home/hpc/cWB/trunk/wat/GPU/gpu_struct.h"

#define num_blocks 16
#define num_threads 256
#define shared_memory_usage 0

#define StreamNum 4
#define BufferNum 4
#define CONSTANT_SIZE 1500
#define MaxPixel 10
#define CLOCK_SIZE 10

inline int _sse_MRA_ps(network *net, float *amp, float *AMP, float Eo, int K);

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
	cout<<"Lsky = "<<Lsky<<endl;
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
	CombineSize = V4max / 2;
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
	for(int z=0; z<kcount; z++)			// loop over unskiped clusters
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
		}
		if( pixelCount<MaxPixel && alloced_V4<CombineSize )	continue;
		//cout<<"list "<<i<<" gross V4 = "<<alloced_V4<<endl;
		//for(int z=0; z<pixelCount; z++)
                //        cout<<"k = "<<pre_gpu_data[i].other_data.eTD[z]-1<<endl;
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
		
	}
	if(alloced_gpu != 0)
	{
		push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, alloced_V4_array, etddim_array, Lsky, pixel_array, StreamNum, stream);
		for(int i=0; i<StreamNum; i++)
			CUDA_CHECK(cudaStreamSynchronize(stream[i]));
		alloced_gpu = 0;
	}		
	
	cleanup_cpu_mem(pre_gpu_data, post_gpu_data, stream);
        cleanup_gpu_mem(skyloop_output, skyloop_other, stream);
	for(i=0; i<StreamNum; i++)
		cudaStreamDestroy(stream[i]);
	for(int i=0; i<StreamNum; i++)				// add count
                count += streamCount[i];
        cout<<"count = "<<count<<endl;
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
//        for(int i=0; i<work_size; i++)
//                cudaStreamAddCallback(stream[i], MyCallback, (void*)&post_gpu_data[i], 0);
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
        aa = aa*(1-msk) - 1*msk;
        gpu_output[rEDim + l + output_ptr] = aa;
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
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *post_gpu_data)
{
//	FILE *fpt = fopen("./debug_files/skyloop_output", "a");
	int Lsky = gpu_Lsky;
	int k;
	size_t V4;
	int pixelcount=0;
	size_t output_ptr = 0;
	//cout<<"Callback"<<endl;
	k = ((post_data*)post_gpu_data)->other_data.k[pixelcount] - 1;
	
	while(k != -1)
	{
		//cout<<"k = "<<k<<endl;
		V4 = ((post_data*)post_gpu_data)->other_data.V4[pixelcount];

		after_skyloop(post_gpu_data, pixelcount, output_ptr);
		output_ptr = output_ptr + V4*Lsky + Lsky;
		pixelcount++;
		if(pixelcount<MaxPixel)
			k = ((post_data*)post_gpu_data)->other_data.k[pixelcount] - 1;
		else 
			break;
	}
//	fclose(fpt);
}
void after_skyloop(void *post_gpu_data, int pixelcount, size_t output_ptr)
{
//	FILE *fpt = fopen("./debug_files/skyloop_output", "a");
	bool mra = false;
	float vvv[NIFO], *v00[NIFO], *v90[NIFO];
	float *rE, *aa_array;
	float Ln, Eo, Ls;
	float aa, AA, En, Es, ee, em, stat, Lm, Em, Am, Lo, EE, rHo, To, TH;
	int l, lb, le, lag, stream, lm, Vm, Lsky;
	int f_ = NIFO/4;
	int m;
	size_t id, nIFO, V, V4, tsize, count;
	size_t k = 0;
	short *ml[NIFO], *mm;
	double suball, submra;
	double xx[NIFO];
	
	count = lb = m = Ln = Eo = Ls = 0;	suball = submra = 0;
	stat=Lm=Em=Am=EE=0.;    lm=Vm= -1;	Lsky = gpu_Lsky;	le = Lsky - 1;	
	k = ((post_data*)post_gpu_data)->other_data.k[pixelcount] - 1;
	V = ((post_data*)post_gpu_data)->other_data.V[pixelcount];
	V4 = ((post_data*)post_gpu_data)->other_data.V4[pixelcount];
	tsize = ((post_data*)post_gpu_data)->other_data.tsize[pixelcount];
	id = ((post_data*)post_gpu_data)->other_data.id[pixelcount];
	En = ((post_data*)post_gpu_data)->other_data.T_En;
	Es = ((post_data*)post_gpu_data)->other_data.T_Es;
	TH = ((post_data*)post_gpu_data)->other_data.TH;
	lag = ((post_data*)post_gpu_data)->other_data.lag;
	nIFO = ((post_data*)post_gpu_data)->other_data.nIFO;
	stream = ((post_data*)post_gpu_data)->other_data.stream;
	
	for(int i=0; i<NIFO; i++)
		ml[i] = ((post_data*)post_gpu_data)->other_data.ml_mm + i*Lsky;                      
        mm = ((post_data*)post_gpu_data)->other_data.ml_mm + NIFO*Lsky;                                 
        rE = ((post_data*)post_gpu_data)->output.output + output_ptr;
        aa_array = ((post_data*)post_gpu_data)->output.output + V4*Lsky + output_ptr;
	
	//cout<<"1"<<endl;
	std::vector<int> pI;                      // buffer for pixel TDs                      
        wavearray<float> fp(NIFO*V4);  fp=0;     // aligned array for + antenna pattern             
        wavearray<float> fx(NIFO*V4);  fx=0;     // aligned array for x antenna pattern 
        wavearray<float> nr(NIFO*V4);  nr=0;     // aligned array for inverse rms            
        wavearray<float> Fp(NIFO*V4);  Fp=0;     // aligned array for pattern                
        wavearray<float> Fx(NIFO*V4);  Fx=0;     // aligned array for pattern
        wavearray<float> am(NIFO*V4);  am=0;     // aligned array for TD amplitudes                 
        wavearray<float> AM(NIFO*V4);  AM=0;     // aligned array for TD amplitudes                 
        wavearray<float> bb(NIFO*V4);  bb=0;     // temporary array for MRA amplitudes              
        wavearray<float> BB(NIFO*V4);  BB=0;     // temporary array for MRA amplitudes  
        wavearray<float> xi(NIFO*V4);  xi=0;     // 00 array for reconctructed responses            
        wavearray<float> XI(NIFO*V4);  XI=0;     // 90 array for reconstructed responses
	
	__m128* _Fp = (__m128*) Fp.data;                                                             
        __m128* _Fx = (__m128*) Fx.data;                                                             
        __m128* _am = (__m128*) am.data;                                                             
        __m128* _AM = (__m128*) AM.data;                                                             
        __m128* _xi = (__m128*) xi.data;                                                             
        __m128* _XI = (__m128*) XI.data;                                                             
        __m128* _fp = (__m128*) fp.data;                                                             
        __m128* _fx = (__m128*) fx.data;                                                             
        __m128* _bb = (__m128*) bb.data;                                                             
        __m128* _BB = (__m128*) BB.data;                                                             
        __m128* _nr = (__m128*) nr.data;  
	__m128 _E_n = _mm_setzero_ps();         // network energy above the threshold                
	__m128 _E_s = _mm_setzero_ps();         // subnet energy above the threshold    

	netpixel *pix;
	std::vector<int> *vint;
	
	//cout<<"2"<<endl;
	// initialize data
	gpu_net->a_00.resize(NIFO*V4);  gpu_net->a_00=0.;                                            
        gpu_net->a_90.resize(NIFO*V4);  gpu_net->a_90=0.;   
        __m128* _aa = (__m128*) gpu_net->a_00.data;         // set pointer to 00 array               
	__m128* _AA = (__m128*) gpu_net->a_90.data;         // set pointer to 90 array

        gpu_net->rNRG.resize(V4);       gpu_net->rNRG=0.;
        gpu_net->pNRG.resize(V4);       gpu_net->pNRG=0.;
	
	pI = gpu_net->wdmMRA.getXTalk(pwc, id);
	gpu_net->pList.clear();
	for(int j=0; j<V; j++)                  // loop over selected pixels
        {
                pix = pwc->getPixel(id, pI[j]); // get pixel pointer
                gpu_net->pList.push_back(pix);
                double rms = 0.;
                for(int i=0; i<nIFO; i++)
                {
                        xx[i] = 1./pix->data[i].noiserms;
                        rms += xx[i]*xx[i];     // total inverse variance
                }
                for(int i=0; i<nIFO; i++)
                        nr.data[j*NIFO+i]=(float)xx[i]/sqrt(rms);       // normalized 1/rms
        }
	//cout<<"3"<<endl;

skyloop:
	for(l=lb; l<=le; l++)
	{
		if(!mm[l] || l<0)	continue;
		aa = aa_array[l];
		if(aa == -1) 	continue;
		for(int j=0; j<V; j++)
			gpu_net->rNRG.data[j] = rE[l*V4+j];
		
		gpu_net->pnt_(v00, pa[stream][pixelcount], ml, (int)l, (int)V4);	// pointers to first pixel 00 data
		gpu_net->pnt_(v90, pA[stream][pixelcount], ml, (int)l, (int)V4);	// pointers to first pixel 90 data
	
		float *pfp = fp.data;
		float *pfx = fx.data;
		float *p00 = gpu_net->a_00.data;
		float *p90 = gpu_net->a_90.data;
		
		m = 0;
		for(int j=0; j<V; j++)
                {
                        int jf= j*f_;
                        gpu_net->cpp_(p00,v00); gpu_net->cpp_(p90,v90);                 // copy amplitudes with target increment
                        gpu_net->cpf_(pfp,FP,l);gpu_net->cpf_(pfx,FX,l);                // copy antenna with target increment
                        _sse_zero_ps(_xi+jf);                      // zero MRA amplitudes
                        _sse_zero_ps(_XI+jf);                      // zero MRA amplitudes
                        _sse_cpf_ps(_am+jf,_aa+jf);                // duplicate 00
                        _sse_cpf_ps(_AM+jf,_AA+jf);                // duplicate 90 

                        if(gpu_net->rNRG.data[j]>En) m++;              // count superthreshold pixels
                }
		
		__m128* _pp = (__m128*) am.data;              // point to multi-res amplitudes
                __m128* _PP = (__m128*) AM.data;              // point to multi-res amplitudes
		
		if(mra)
		{
			_sse_MRA_ps(gpu_net, xi.data, XI.data, En, m);  // get principal components
        	        _pp = (__m128*) xi.data;                                                // point to PC amplitudes
                	_PP = (__m128*) XI.data;                                                // point to Pc amplitudes
		}
		
		m = 0; Ls = Ln = Eo = 0;
		for(int j=0; j<V; j++)
		{
			int jf = j*f_;  // source sse pointer increment 
			int mf = m*f_;  // target sse pointer increment 
			_sse_zero_ps(_bb+jf);   // reset array for MRA amplitudes
			_sse_zero_ps(_BB+jf);       // reset array for MRA amplitudes
			ee = _sse_abs_ps(_pp+jf,_PP+jf);        // total pixel energy
			if(ee<En) continue;
			_sse_cpf_ps(_bb+mf,_pp+jf);         // copy 00 amplitude/PC
			_sse_cpf_ps(_BB+mf,_PP+jf);         // copy 90 amplitude/PC
			_sse_cpf_ps(_Fp+mf,_fp+jf);         // copy F+
			_sse_cpf_ps(_Fx+mf,_fx+jf);         // copy Fx
			_sse_mul_ps(_Fp+mf,_nr+jf);         // normalize f+ by rms
			_sse_mul_ps(_Fx+mf,_nr+jf);         // normalize fx by rms
			m++;
			em = _sse_maxE_ps(_pp+jf,_PP+jf);   // dominant pixel energy
			Ls += ee-em; Eo += ee;       // subnetwork energy, network energy
			if(ee-em>Es) Ln += ee;       // network energy above subnet threshold
		}
		
		size_t m4 = m + (m%4 ? 4 - m%4 : 0);
		_E_n = _mm_setzero_ps();	// + likelihood
	
		for(int j=0; j<m4; j+=4)
                {
                    int jf = j*f_;
                    _sse_dpf4_ps(_Fp+jf,_Fx+jf,_fp+jf,_fx+jf);  // go to DPF
                    _E_s = _sse_like4_ps(_fp+jf,_fx+jf,_bb+jf,_BB+jf);  // std likelihood
                    _E_n = _mm_add_ps(_E_n,_E_s);                       // total likelihood
                }
		_mm_storeu_ps(vvv,_E_n);

                Lo = vvv[0]+vvv[1]+vvv[2]+vvv[3];
                AA = aa/(fabs(aa)+fabs(Eo-Lo)+2*m*(Eo-Ln)/Eo);        //  subnet stat with threshold
                ee = Ls*Eo/(Eo-Ls);
                em = fabs(Eo-Lo)+2*m;   //  suball NULL
                ee = ee/(ee+em);        //  subnet stat without threshold
                aa = (aa-m)/(aa+m);

                if(AA>stat && !mra)
                {
                        stat=AA; Lm=Lo; Em=Eo; Am=aa; lm=l; Vm=m; suball=ee; EE=em;
                }
	}
	if(!mra && lm>=0) {mra=true; le=lb=lm; goto skyloop;}    // get MRA principle components
	
/*	vint = &(pwc->cList[id-1]);
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
	}*/
	streamCount[stream] += count;
//	fclose(fpt);
	//cout<<"4"<<endl;
	return;
	
}

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

                CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.eTD), NIFO * eTDDim * sizeof(float), cudaHostAllocMapped ) );
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
                CUDA_CHECK(cudaMalloc(&(skyloop_other[i].eTD), NIFO * eTDDim * sizeof(float) ) );
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
inline int _sse_MRA_ps(network* net, float* amp, float* AMP, float Eo, int K) {
// fast multi-resolution analysis inside sky loop
// select max E pixel and either scale or skip it based on the value of residual
// pointer to 00 phase amplitude of monster pixels
// pointer to 90 phase amplitude of monster pixels
// Eo - energy threshold
//  K - number of principle components to extract
// returns number of MRA pixels
   int j,n,mm;
   int k = 0;
   int m = 0;
   int f = NIFO/4;
   int V = (int)net->rNRG.size();
   float*  ee = net->rNRG.data;                            // residual energy
   float*  pp = net->pNRG.data;                            // residual energy
   float   EE = 0.;                                         // extracted energy
   float   E;
   float mam[NIFO];
   float mAM[NIFO];
   net->pNRG=-1;
   for(j=0; j<V; ++j) if(ee[j]>Eo) pp[j]=0;

   __m128* _m00 = (__m128*) mam;
   __m128* _m90 = (__m128*) mAM;
   __m128* _amp = (__m128*) amp;
   __m128* _AMP = (__m128*) AMP;
   __m128* _a00 = (__m128*) net->a_00.data;
   __m128* _a90 = (__m128*) net->a_90.data;

   while(k<K){

      for(j=0; j<V; ++j) if(ee[j]>ee[m]) m=j;               // find max pixel
      if(ee[m]<=Eo) break;  mm = m*f;

      //cout<<" V= "<<V<<" m="<<m<<" ee[m]="<<ee[m];

             E = _sse_abs_ps(_a00+mm,_a90+mm); EE += E;     // get PC energy
      int    J = net->wdmMRA.getXTalk(m)->size()/7;
      float* c = net->wdmMRA.getXTalk(m)->data;             // c1*c2+c3*c4=c1*c3+c2*c4=0

      if(E/EE < 0.01) break;                                // ignore small PC

      _sse_cpf_ps(mam,_a00+mm);                             // store a00 for max pixel
      _sse_cpf_ps(mAM,_a90+mm);                             // store a90 for max pixel
      _sse_add_ps(_amp+mm,_m00);                            // update 00 PC
      _sse_add_ps(_AMP+mm,_m90);                            // update 90 PC

      for(j=0; j<J; j++) {
         n = int(c[0]+0.1);
         if(ee[n]>Eo) {
            ee[n] = _sse_rotsub_ps(_m00,c[1],_m90,c[2],_a00+n*f);    // subtract PC from a00
            ee[n]+= _sse_rotsub_ps(_m00,c[3],_m90,c[4],_a90+n*f);    // subtract PC from a90
         }
         c += 7;
      }
      //cout<<" "<<ee[m]<<" "<<k<<" "<<E<<" "<<EE<<" "<<endl;
      pp[m] = _sse_abs_ps(_amp+mm,_AMP+mm);    // store PC energy
      k++;
   }
   return k;
}
