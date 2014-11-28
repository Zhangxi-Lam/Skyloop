#define MaxPixel 10 
struct skyloop_output                           // the output of skyloop
{
	float *output;
};
struct STAT
{
	float stat, Lm, Em, Am, suball, EE;
	int lm, Vm;
};
struct other                                    //the variable that not use in GPU
{
	float *vtd_vTD_nr;
	float *eTD;
	float *BB;
	float *bb;
	float *fp;
	float *fx;
	float *Fp;
	float *Fx;
	float *tmp;
	double *FP_FX;
	short *ml_mm;
	size_t *V_tsize;
	float T_En, T_Es, TH;	// same
	int le, lag;	// le, lag same
	int stream;	// indicate which stream
	size_t nIFO;
	size_t id[MaxPixel];
	size_t k[MaxPixel];
	size_t V[MaxPixel];
	size_t V4[MaxPixel];
	size_t tsize[MaxPixel]; 
	size_t count;			// result of each stream
};
struct pre_data
{
    struct other other_data;
};
struct post_data
{
        struct skyloop_output output;
        struct other other_data;
};
