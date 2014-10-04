/*struct skyloop_input                            // the input of skyloop
{
        float *eTD[NIFO];
        short *ml[NIFO];
		short *mm;								// Lsky
        size_t *V, *V4;
        float *T_En, *T_Es;
		int *le;		
};*/
struct skyloop_output                           // the output of skyloop
{
	float *rE;				// pointer of rNRG.data	V4max
	float *pE;				// pointer of pNRG.data	V4max
	float *aa;				// float
	int *Mm;				// float
};
struct other                                    //the variable that not use in GPU
{
	float *eTD[NIFO];
	short *ml[NIFO];
	short *mm;				// skyMask.data
	float *T_En, *T_Es, *TH;	// same
	int *le, *lag;	// le, lag same
	int *stream;	// indicate which stream
	size_t *id, *k, *V, *V4, *tsize; // nIFO same
	size_t *count;			// result of each stream
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
