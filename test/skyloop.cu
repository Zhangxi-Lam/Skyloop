#define num_blocks 16
#define num_threads 256
#define shared_memory_usage 0

#define StreamNum 4
#define BufferNum 4
#define CONSTANT_SIZE 1000 
#define NIFO 4
#define nIFO 3
#define XIFO 4
#define CUDA_CHECK(value) {                                             \
    cudaError_t _m_cudaStat = value;                                    \
    if (_m_cudaStat != cudaSuccess) {                                   \
        fprintf(stderr, "Error %s at line %d in file %s\n",             \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
        exit(1); }}



#include "/home/hpc/cWB/TEST/S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2/macro/test/test_struct.h"
#include "/home/hpc/cWB/TEST/S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2/macro/test/main.cuh"
#include<fstream>
#include<iostream>
using namespace std;

bool flag;
__constant__ float constEn, constEs;				// GPU constant memory
__constant__ size_t constV[CONSTANT_SIZE], constV4[CONSTANT_SIZE], consttsize[CONSTANT_SIZE];
int main(void)
{
	struct pre_data pre_gpu_data[BufferNum];	// store the data before gpu calculation
	struct post_data post_gpu_data[StreamNum];	// store the data transfer from gpu
	struct skyloop_output skyloop_output[StreamNum];	// store the skyloop_output data
	struct other skyloop_other[StreamNum];		// store the data which is not output
	
	ifstream file1("../../skyloop_mm");
	ifstream file2("../../skyloop_ml");
	ifstream file3("../../skyloop_eTD");
	ifstream file4("../../skyloop_inputV");
	
	short temp1;
	short temp2;
	
	int Tmax, V4max, eTDDim, Lsky;
	int alloced_gpu = 0;
	int K;
	size_t V, V4, tsize;
	size_t *V_array, *V4_array, *tsize_array;
	size_t k_array[StreamNum];
	float En, Es;
	int etddim;
	Tmax = 300;
	V4max = 219;
	Lsky = 3072;
	flag = false;
	eTDDim = Tmax * V4max;
	allocate_cpu_mem(pre_gpu_data, post_gpu_data, eTDDim, V4max, Lsky);
	allocate_gpu_mem(skyloop_output, skyloop_other, eTDDim, V4max, Lsky);

	cudaStream_t stream[StreamNum];			// define the stream
	for(int i=0; i<StreamNum; i++)			
		CUDA_CHECK(cudaStreamCreate(&stream[i]));	// create the new stream

	for(int l=0; l<Lsky; l++)
	{
		for(int j=0; j<NIFO; j++)
		{
			file2>>temp2;
			int mlptr;
			mlptr = j * Lsky;
			pre_gpu_data[0].other_data.ml_mm[mlptr + l] = temp2;
		}
		file1>>temp1;
		pre_gpu_data[0].other_data.ml_mm[NIFO * Lsky + l] = temp1;
	}

	cudaMemcpyAsync(skyloop_other[0].ml_mm, pre_gpu_data[0].other_data.ml_mm, (1+NIFO) * Lsky * sizeof(short), cudaMemcpyHostToDevice, stream[0] );
	
	K = 115;
	V_array = (size_t*)malloc(sizeof(size_t) * K);
	V4_array = (size_t*)malloc(sizeof(size_t) * K);
	tsize_array = (size_t*)malloc(sizeof(size_t) * K);
	 
	clock_t start[10], finish[10];
	double d[10];
	for(int i=0; i<10; i++)
		d[i] = 0;
	for(int k=0; k<K; k++)
	{
		file4>>V;
		file4>>V4;
		file4>>tsize;	
		file4>>En;
		file4>>Es;
		if(k==0)
		{
			cudaMemcpyToSymbol(constEn, &En, sizeof(float));
			cudaMemcpyToSymbol(constEs, &Es, sizeof(float));
		}
		V_array[k] = V;
		V4_array[k] = V4;
		tsize_array[k] = tsize;
				
	}
	cudaMemcpyToSymbol(constV, V_array, sizeof(size_t) * K);
	cudaMemcpyToSymbol(constV4, V4_array, sizeof(size_t) * K);
	cudaMemcpyToSymbol(consttsize, tsize_array, sizeof(size_t) * K);

	start[0] = clock();
	for(int k=0; k<K; k++)
	{
		start[1] = clock();
		
		etddim = V4_array[k] * tsize_array[k];
		for(int l=0; l<eTDDim; l++)
		{
			float temp;
			if(l<etddim)
			{
				file3>>pre_gpu_data[alloced_gpu].other_data.eTD[l];
				file3>>pre_gpu_data[alloced_gpu].other_data.eTD[l + etddim];
				file3>>pre_gpu_data[alloced_gpu].other_data.eTD[l + 2 * etddim];
				file3>>pre_gpu_data[alloced_gpu].other_data.eTD[l + 3 * etddim];
			}
			else
			{
				file3>>temp;
				file3>>temp;
				file3>>temp;
				file3>>temp;
			}
			
		}
		finish[1] = clock();
		d[1] += (double)(finish[1] - start[1]);
		if(alloced_gpu < BufferNum)
		{	
			start[2] = clock();
			int i = alloced_gpu;
			k_array[i] = k;
			
			post_gpu_data[i].other_data.k = k;
			post_gpu_data[i].other_data.V4 = V4_array[k];
			post_gpu_data[i].other_data.tsize = tsize_array[k];
			alloced_gpu++;
			if(alloced_gpu == StreamNum)		// if all streams' data have been assigned
			{
				push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, V4_array, tsize_array, k_array, Lsky, StreamNum, stream);
				for(int i=0; i<StreamNum; i++)				// wait for all commands in the stream to complete
					CUDA_CHECK(cudaStreamSynchronize(stream[i]));
				alloced_gpu = 0;
			}
			finish[2] = clock();
			d[2] += (double)(finish[2] - start[2]);
		 }
	}
	if(alloced_gpu != 0)
	{
		push_work_into_gpu(pre_gpu_data, post_gpu_data, skyloop_output, skyloop_other, V4_array, tsize_array, k_array, Lsky, alloced_gpu, stream);
		for(int i=0; i<alloced_gpu; i++)				// wait for all commands in the stream to complete
			CUDA_CHECK(cudaStreamSynchronize(stream[i]));
		alloced_gpu = 0;
	}
	finish[0] = clock();
	
//	printf("diff time = %f\n", (double)(diff)/CLOCKS_PER_SEC);
	
	printf("time = %f\n", (double)(d[2])/CLOCKS_PER_SEC);
	cleanup_cpu_mem(pre_gpu_data, post_gpu_data, stream);
	cleanup_gpu_mem(skyloop_output, skyloop_other, stream);
	for(int i=0; i<StreamNum; i++)	
		cudaStreamDestroy(stream[i]);
	cout<<"Finish!"<<endl;

	return 0;
}
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void* post_gpu_data)
{
	//FILE *fpt = fopen("./output/test_skyloopOutput", "a");
	float aa;
	int m, l, lb;
	int le = 3071;
	int rEDim;
	lb = 0;
	int k;
	size_t V4, tsize;
	k = ((post_data*)post_gpu_data)->other_data.k;
	V4 = ((post_data*)post_gpu_data)->other_data.V4;
	rEDim = 3072 * V4;
	for(l=lb; l<=le; l++)
	{
	       	aa = ((post_data*)post_gpu_data)->output.output[rEDim + l];
	//	fprintf(fpt, "k = %d l = %d aa = %f\n", k, l, aa);
	}
	//fclose(fpt);
}
void allocate_cpu_mem(struct pre_data *pre_gpu_data, struct post_data *post_gpu_data, int eTDDim, int V4max, int Lsky)// allocate locked memory on CPU 
{
	for(int i = 0; i<BufferNum; i++)
	{	
		CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[i].other_data.eTD), NIFO * eTDDim * sizeof(float), cudaHostAllocMapped ) );
	}
	CUDA_CHECK(cudaHostAlloc(&(pre_gpu_data[0].other_data.ml_mm), (1+NIFO) * Lsky * sizeof(short), cudaHostAllocMapped ) );
	
	for(int i = 0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaHostAlloc(&(post_gpu_data[i].output.output), Lsky * V4max * sizeof(float) + Lsky * sizeof(float), cudaHostAllocMapped ) );
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
	CUDA_CHECK(cudaMalloc(&(skyloop_other[0].ml_mm), (1+NIFO) * Lsky * sizeof(short) ) );	
}

void cleanup_gpu_mem(struct skyloop_output *skyloop_output, struct other *skyloop_other, cudaStream_t *stream)// cleanup the memory on GPU
{
	for(int i = 0; i<StreamNum; i++)
	{
		CUDA_CHECK(cudaFree(skyloop_other[i].eTD) );
		CUDA_CHECK(cudaFree(skyloop_output[i].output) );
		//cout<<"cleanup gpu"<<endl;
	}
	CUDA_CHECK(cudaFree(skyloop_other[0].ml_mm) );
	return;
}

__host__ void push_work_into_gpu(struct pre_data *input_data, struct post_data *post_gpu_data, struct skyloop_output *skyloop_output, struct other *skyloop_other, size_t *V4_array, size_t *tsize_array, size_t *k_array, int Lsky, int work_size, cudaStream_t *stream)
{
	int etddim;
	int V4;
	int k; 
	for(int i=0; i<work_size; i++)// transfer the data from CPU to GPU
	{
		k = k_array[i];
		etddim = tsize_array[k] * V4_array[k];
		cudaMemcpyAsync(skyloop_other[i].eTD, input_data[i].other_data.eTD, NIFO * etddim * sizeof(float), cudaMemcpyHostToDevice, stream[i] );
	}

	for(int i=0; i<work_size; i++)// call for gpu caculation
		kernel_skyloop<<<num_blocks, num_threads, shared_memory_usage, stream[i]>>>(skyloop_other[i].eTD, skyloop_other[0].ml_mm, skyloop_output[i].output, Lsky, k_array[i]);

	for(int i=0; i<work_size; i++)// transfer the data back from GPU to CPU
	{
		k = k_array[i];
		V4 = V4_array[k];
        	cudaMemcpyAsync(post_gpu_data[i].output.output, skyloop_output[i].output, Lsky * V4 * sizeof(float) + Lsky * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
	        //cudaMemcpyAsync(post_gpu_data[i].output.aa, skyloop_output[i].aa, Lsky * sizeof(float), cudaMemcpyDeviceToHost, stream[i] );
	}
	for(int i=0; i<work_size; i++)
		cudaStreamAddCallback(stream[i], MyCallback, (void*)&post_gpu_data[i], 0);
	flag = true;
//	cout<<"Push work into gpu success."<<endl;
}

__global__ void kernel_skyloop(float *eTD, short *ml_mm, float *gpu_output, int Lsky, size_t k) 
{
	const int grid_size = blockDim.x * gridDim.x;
	int l = blockIdx.x * blockDim.x + threadIdx.x;
	float *pe[NIFO];
	short *ml[NIFO];
	short *mm;
	size_t V, V4, tsize;
	int le = Lsky - 1;

	V = constV[k];
	V4 = constV4[k];
	tsize = consttsize[k];
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

__inline__ __device__ void kernel_skyloop_calculate(float *PE_0, float *PE_1, float *PE_2, float *PE_3, size_t V, size_t V4, size_t rEDim, float *gpu_output, int l) 
{
	int msk;						// mask
	size_t v = 0;					// indicate the pixel
	size_t ptr;						// indicate the location 
	float pe[NIFO];
	float _Eo[4], _Es[4], _En[4];
	float En, Es, Eo, aa;
	int Mm;
	float rE;						// energy array rNRG.data 
	float pE;						// energy array pNRG.data
	int count;
	
	//Eo = 0;							// total network energy
	//En = 0;							// network energy above the threshold
	//Es = 0;							// subnet energy above the threshold
	Mm = 0;							// # of pixels above the threshold
	
	for(count=0; count<4; count++)
	{
		_Eo[count] = 0;
		_Es[count] = 0;
		_En[count] = 0;
	}
	count = 0;
	ptr = l*V4;
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
		msk = ( rE>=constEn );										// E>En  0/1 mask
		Mm += msk;												// count pixels above threshold
		///*new
		pE = rE * msk;											// zero sub-threshold pixels
		_Eo[count] += pE;												// network energy
		pE = kernel_minSNE_ps(pE, pe);						// subnetwork energy
		_Es[count] += pE;												// subnetwork energy
		msk = ( pE>=constEs );										// subnet energy > Es 0/1 mask
		rE *= msk;												
		_En[count] +=rE;											// network energy
		//En += rE;												// network energy
		// assign the value to the local memory
		v++;
		count++;
		count = count%4;
	}

	En = _En[0] + _En[1] + _En[2] + _En[3];												// Write back to output
	Eo = _Eo[0] + _Eo[1] + _Eo[2] + _Eo[3] + 0.01;												
	Es = _Es[0] + _Es[1] + _Es[2] + _Es[3];
	Mm = Mm *2 +0.01;
	aa = Es*En/(Eo-Es);
	
	msk = ((aa-Mm)/(aa+Mm)<0.33);								// if need continue 1/0
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
	temp = temp + pE - flag*temp - (1-flag)*pE;
	return temp;
} 


