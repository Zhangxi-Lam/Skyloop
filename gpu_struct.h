#include "TH2F.h"
#include "wavearray.hh"
#include "netcluster.hh"
#include "netpixel.hh"
#include "skymap.hh"
#include "monster.hh"
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
        float *rE;				// pointer of rNRG.data
        float *pE;				// pointer of pNRG.data
        float *Eo;
        float *En;
        float *Es;
        int *Mm;
};
struct other                                    //the variable that not use in GPU
{
	float *eTD[NIFO];
	short *ml[NIFO];
	short *mm;				// skyMask.data
	float *T_En, *T_Es, *TH;
	int *le, *vint_size, *lag;
	size_t *id, *nIFO, *V, *V4;
	size_t *count;			// result of each stream
	bool *finish;			// indicate whether the caculation is finished
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
