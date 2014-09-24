#define XIFO 4

#pragma GCC system_header

#include "cwb.hh"
#include "cwb2G.hh"
#include "config.hh"
#include "/home/hpc/cWB/trunk/tools/install/inc/network.hh"
#include "wavearray.hh"
#include "TString.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TRandom.h"
#include "TComplex.h"

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#include "/home/hpc/cWB/TEST/S6A_BKG_LF_L1H1V1_2G_SUPERCLUSTER_run1a_bench2/macro/gpu_struct.h"
//!SUPERCLUSTER

long subNetCut(network* net, int lag, float snc, TH2F* hist);
inline int _sse_MRA_ps(network* net, float* amp, float* AMP, float Eo, int K);
void PrintElapsedTime(int job_elapsed_time, double cpu_time, TString info);
long gpu_subNetCut(network* net, int lag, float snc, TH2F* hist);

#define USE_LOCAL_SUBNETCUT	// comment to use the builtin implementation of subNetCut

void 
CWB_Plugin(TFile* jfile, CWB::config* cfg, network* net, WSeries<double>* x, TString ifo, int type)  {

// This plugin implements the standard supercluster stage & use a local implementation of subNetCut (only 2G)

  cout << endl;
  cout << "-----> CWB_Plugin_supercluster_bench.C" << endl;
  cout << "ifo " << ifo.Data() << endl;
  cout << "type " << type << endl;
  cout << endl;

  if(type==CWB_PLUGIN_CONFIG) {  
    cfg->scPlugin=true;  	// disable built-in supercluster function
  }

  if(type==CWB_PLUGIN_ISUPERCLUSTER) {

    cout << "type==CWB_PLUGIN_ISUPERCLUSTER" << endl;

    TStopwatch bench;
    bench.Stop();

    // import ifile
    void* gIFILE=NULL; IMPORT(void*,gIFILE)
    cout << "-----> CWB_Plugin_wavegraph.C -> " << " gIFILE : " << gIFILE << endl;
    TFile* ifile = (TFile*)gIFILE;

    // import ifactor
    int gIFACTOR=-1; IMPORT(int,gIFACTOR)
    cout << "-----> CWB_Plugin_wavegraph.C -> " << " gIFACTOR : " << gIFACTOR << endl;
    int ifactor = gIFACTOR;

    int nIFO = net->ifoListSize();			// number of detectors
    int rateANA=cfg->inRate>>cfg->levelR;
    int nRES = net->wdmListSize();			// number of resolution levels
    int lags = net->getifo(0)->lagShift.size();

    wavearray<double>* hot[NIFO_MAX];  			// temporary time series
    for(int i=0; i<nIFO; i++) hot[i] = net->getifo(i)->getHoT();

    int nevt = 0;
    int nnn = 0;
    int mmm = 0;
    size_t count = 0;
    netcluster  wc;
    netcluster* pwc;

    for(int j=0; j<(int)lags; j++) {

      int cycle = cfg->simulation ? ifactor : Long_t(net->wc_List[j].shift);

      // read cluster metadata
      if(ifile!=NULL) wc.read(ifile,"coherence","clusters",0,cycle);
      else            wc.read(jfile,"coherence","clusters",0,cycle);
      // read clusters from temporary job file, loop over TF resolutions
      if(ifile!=NULL) {
        for(int i=nRES-1; i>=0; i--)     // reverse loop is faster loading cluster (?)
          wc.read(ifile,"coherence","clusters",-1,cycle,rateANA>>(i+cfg->l_low));
      } else {
        for(int i=nRES-1; i>=0; i--)     // reverse loop is faster loading cluster (?)
          wc.read(jfile,"coherence","clusters",-1,cycle,rateANA>>(i+cfg->l_low));
      }
      if(!cfg->simulation) cout<<"process lag "   <<cycle<<" ..."<<endl;
      cout<<"loaded clusters|pixels: "<<wc.csize()<<"|"<<wc.size()<<endl;

      // supercluster analysis
      wc.supercluster('L',net->e2or,cfg->TFgap,false);  //likehood2G
      cout<<"super  clusters|pixels: "<<wc.esize(0)<<"|"<<wc.psize(0)<<endl;

      // release all pixels
      pwc = net->getwc(j);
      pwc->cpf(wc, false);

      net->setDelayIndex(hot[0]->rate());
      pwc->setcore(false);

      // apply cuts
      int psel = 0;
      while(1) {
        count = pwc->loadTDampSSE(*net, 'a', cfg->BATCH, cfg->LOUD);
        bench.Continue();
#ifdef USE_LOCAL_SUBNETCUT
        psel += subNetCut(net,(int)j,cfg->subnet,NULL);
#else
        psel += net->subNetCut((int)j,cfg->subnet,NULL);
#endif
        bench.Stop();
        PrintElapsedTime(bench.RealTime(),bench.CpuTime(),"subNetCut : Processing Time - ");
        int ptot = pwc->psize(1)+pwc->psize(-1);
        double pfrac = ptot>0 ? double(psel)/double(ptot) : 0.;
        cout<<"selected pixels: "<<psel<<", fraction: "<<pfrac<< endl;
        if(count<10000) break;
      }

      pwc->defragment(cfg->Tgap,cfg->Fgap);    // SK added defragmentation

      nevt = net->events();
      nnn += pwc->psize(-1);
      mmm += pwc->psize(1)+pwc->psize(-1);

      if(mmm) cout<<"events in the buffer: "<<net->events()<<"|"<<nnn<<"|"<<nnn/double(mmm)<<"\n";
      else    cout<<"events in the buffer: "<<net->events()<<"\n";

      // store cluster into temporary job file [NEWSS]
      pwc->write(jfile,"supercluster","clusters",0,cycle);
      pwc->write(jfile,"supercluster","clusters",-1,cycle);
      cout<<cycle<<"|"<<pwc->csize()<<"|"<<pwc->size()<<" ";cout.flush();

      pwc->clear();
      cout<<endl;
    }
  }

  return;
}

void
PrintElapsedTime(int job_elapsed_time, double cpu_time, TString info) {
//
// convert job_elapsed_time to (hh:mm:ss) format and print it
//
// job_elapsed_time : time (seconds)
//
// info             : info string added to (hh:mm:ss)
//

  int job_elapsed_hour  = int(job_elapsed_time/3600);
  int job_elapsed_min   = int((job_elapsed_time-3600*job_elapsed_hour)/60);
  int job_elapsed_sec   = int(job_elapsed_time-3600*job_elapsed_hour-60*job_elapsed_min);
  char buf[1024];
  sprintf(buf,"%s %02d:%02d:%02d (hh:mm:ss) : cpu time : %f (sec)\n",info.Data(),job_elapsed_hour,job_elapsed_min,job_elapsed_sec,cpu_time);
  cout << buf;

  return;
}

long subNetCut(network* net, int lag, float snc, TH2F* hist)
{
                                     
// sub-network cut with dsp regulator                  
//  lag: lag index                                     
//  snc: sub network threshold, if snc<0 use weak constraint
// hist: diagnostic histogram                               
// return number of processed pixels                        
   if(!net->wc_List[lag].size()) return 0;

   size_t nIFO = net->ifoList.size();
  
   if(nIFO>NIFO) {
      cout<<"network::subNetCut(): invalid network.\n";
      exit(0);                                         
   }
		
	size_t count = 0;
	count = gpu_subNetCut(net, lag, snc, hist);
	return count;
}

long Callback(void* post_gpu_data, network *gpu_net, netcluster *pwc, double **FP, double **FX)
{
	bool mra = false;
	float *v00[NIFO];
	float *v90[NIFO];
	float *pa[NIFO];
	float *pA[NIFO];
	float *rE, *pE;				//pointers of rNRG.data and pNRG.data
	float Ln = 0;
	float Eo = 0;
	float Ls = 0;
	float aa, AA, En, Es;
	int m = 0;
	size_t l, id, nIFO, V, V4, tsize;
	size_t i = 0;
	int f_ =NIFO/4;
	int lb=0;
	int le, lag;
	short *ml[NIFO]; 
	float *eTD[NIFO];
	FILE *fpt = fopen("skyloop_my", "a");

	En = *((post_data*)post_gpu_data)->other_data.T_En;
	Es = *((post_data*)post_gpu_data)->other_data.T_Es;
	le = *((post_data*)post_gpu_data)->other_data.le;
	lag = *((post_data*)post_gpu_data)->other_data.lag;
	id = *((post_data*)post_gpu_data)->other_data.id;
	nIFO = *((post_data*)post_gpu_data)->other_data.nIFO;
	V = *((post_data*)post_gpu_data)->other_data.V;
	V4 = *((post_data*)post_gpu_data)->other_data.V4;
	tsize = *((post_data*)post_gpu_data)->other_data.tsize;
	i = *((post_data*)post_gpu_data)->other_data.count;
	rE = ((post_data*)post_gpu_data)->output.rE;
	pE = ((post_data*)post_gpu_data)->output.pE;
	for(int i=0; i<NIFO; i++)
		ml[i] = ((post_data*)post_gpu_data)->other_data.ml[i];
	
	
	std::vector<wavearray<float> > vtd;		// vectors of TD amplitudes
	std::vector<wavearray<float> > vTD;     // vectors of TD amplitudes
	std::vector<int> pI;				// buffer for pixel TDs
	wavearray<float> tmp(tsize*V4); tmp=0;
	wavearray<float>  fp(NIFO*V4);  fp=0;     // aligned array for + antenna pattern 
	wavearray<float>  fx(NIFO*V4);  fx=0;     // aligned array for x antenna pattern 
	wavearray<float>  nr(NIFO*V4);  nr=0;     // aligned array for x antenna pattern 
    wavearray<float>  Fp(NIFO*V4);  Fp=0;            // aligned array for pattern
    wavearray<float>  Fx(NIFO*V4);  Fx=0;            // aligned array for pattern
	wavearray<float>  am(NIFO*V4);  am=0;     // aligned array for TD amplitudes     
	wavearray<float>  AM(NIFO*V4);  AM=0;     // aligned array for TD amplitudes     
	wavearray<float>  bb(NIFO*V4);  bb=0;     // temporary array for MRA amplitudes  
	wavearray<float>  BB(NIFO*V4);  BB=0;     // temporary array for MRA amplitudes  
	wavearray<float>  xi(NIFO*V4);  xi=0;     // 00 array for reconctructed responses 
	wavearray<float>  XI(NIFO*V4);  XI=0;     // 90 array for reconstructed responses 

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
	
	netpixel* pix;	

	// initialize data
	pI = gpu_net->wdmMRA.getXTalk(pwc, id);
	for(int i=0; i<NIFO; i++)
	{
		
		vtd.push_back(tmp);
		vTD.push_back(tmp);
	}
	for(int i=0; i<NIFO; i++)
	{
		pa[i] = vtd[i].data + (tsize/2)*V4;
		pA[i] = vTD[i].data + (tsize/2)*V4;
	}
	
///*new	
	gpu_net->a_00.resize(NIFO*V4);	gpu_net->a_00=0.;
	gpu_net->a_90.resize(NIFO*V4);	gpu_net->a_90=0.;
	__m128* _aa = (__m128*) gpu_net->a_00.data;         // set pointer to 00 array
    __m128* _AA = (__m128*) gpu_net->a_90.data;         // set pointer to 90 array

	for(int j=0; j<V; j++)			// loop over selected pixels
	{
		pix = pwc->getPixel(id, pI[j]);	// get pixel pointer
		gpu_net->pList.push_back(pix);	// store pixel pointers for MRA
		
		for(int i=0; i<nIFO; i++)
			for(int l=0; l<tsize; l++)
			{
				aa = pix->tdAmp[i].data[l];		// copy TD 00 data
				AA = pix->tdAmp[i].data[l+tsize];	// copy Td 90 data
				vtd[i].data[l*V4+j] = aa;		// copy 00 data
				vTD[i].data[l*V4+j] = AA;		// copy 90 data
			}
		
	}
	
	// after skyloop
	for(l=lb; l<=le; l++)
	{
//		fprintf(fpt, "k = %d l = %d eTD[0] = %f eTD[1] = %f eTD[2] = %f\n", i, l, eTD[0][l], eTD[1][l], eTD[2][l]);
//		fprintf(fpt, "k = %d l = %d ml[0] = %hd ml[1] = %hd ml[2] = %hd\n", i, l, ml[0][l], ml[1][l], ml[2][l]);
		Ln = ((post_data*)post_gpu_data)->output.En[l];
		Eo = ((post_data*)post_gpu_data)->output.Eo[l];
		Ls = ((post_data*)post_gpu_data)->output.Es[l];
		m = ((post_data*)post_gpu_data)->output.Mm[l];
		//fprintf(fpt, "k = %d l = %d Ln = %f Eo = %f Ls = %f m = %d\n", i, l, Ln, Eo, Ls, m);
			
		aa = Ls*Ln/(Eo-Ls);
		if((aa-m)/(aa+m)<0.33)	continue;	
		cout<<"1"<<endl;	
		gpu_net->pnt_(v00, pa, ml, (int)l, (int)V4);	// pointers to first pixel 00 data
		//fprintf(fpt,"k = %d l = %d v00[0] = %f v00[1] = %f v00[2] = %f\n", i, l, v00[0][0], v00[1][0], v00[2][0]);
		gpu_net->pnt_(v90, pA, ml, (int)l, (int)V4);	// pointers to first pixel 90 data
		
		float *pfp = fp.data;
		float *pfx = fx.data;
		float *p00 =gpu_net->a_00.data;
		float *p90 =gpu_net->a_90.data;
		/*
		m = 0;
		for(int j=0; j<V; j++)
		{
			int jf= j*f_;
			gpu_net->cpp_(p00,v00);	gpu_net->cpp_(p90,v90);			// copy amplitudes with target increment
			fprintf(fpt,"k = %d l = %d p00[0] = %f p00[1] = %f p00[2] = %f p00[3] = %f\n", i, l, p00[0], p00[1], p00[2], p00[3]);
			gpu_net->cpf_(pfp,FP,l);gpu_net->cpf_(pfx,FX,l);		// copy antenna with target increment
			fprintf(fpt,"k = %d l = %d pfp[00] = %f pfp[1] = %f pfp[2] = %f pfp[3] = %f\n", i, l, pfp[0], pfp[1], pfp[2], pfp[3]);
			_sse_zero_ps(_xi+jf);                      // zero MRA amplitudes
            _sse_zero_ps(_XI+jf);                      // zero MRA amplitudes
            _sse_cpf_ps(_am+jf,_aa+jf);                // duplicate 00
            _sse_cpf_ps(_AM+jf,_AA+jf);                // duplicate 90 
            if(rE[l*tsize+j]>En) m++;              // count superthreshold pixels
		}
		fprintf(fpt,"k = %d l = %d m = %d \n", i, l, m); */
	/*	
        __m128* _pp = (__m128*) am.data;              // point to multi-res amplitudes
        __m128* _PP = (__m128*) AM.data;              // point to multi-res amplitudes
		
		if(mra)										// do MRA
		{
			_sse_MRA_ps(gpu_net, xi.data, XI.data, En, m);	// get principal components
			_pp = (__m128*) xi.data;						// point to PC amplitudes
			_PP = (__m128*) XI.data;						// point to Pc amplitudes
		}		
		
		m = 0; Ls=Ln=Eo=0;
		for(j=0; j<V; j++)
		{
			int jf = j*f_;                             // source sse pointer increment 
            int mf = m*f_;                             // target sse pointer increment 
            _sse_zero_ps(_bb+jf);                      // reset array for MRA amplitudes
            _sse_zero_ps(_BB+jf);                      // reset array for MRA amplitudes
            ee = _sse_abs_ps(_pp+jf,_PP+jf);           // total pixel energy            
            if(ee<En) continue;                                                         
            _sse_cpf_ps(_bb+mf,_pp+jf);                // copy 00 amplitude/PC          
            _sse_cpf_ps(_BB+mf,_PP+jf);                // copy 90 amplitude/PC          
            _sse_cpf_ps(_Fp+mf,_fp+jf);                // copy F+                       
            _sse_cpf_ps(_Fx+mf,_fx+jf);                // copy Fx                       
            _sse_mul_ps(_Fp+mf,_nr+jf);                // normalize f+ by rms           
            _sse_mul_ps(_Fx+mf,_nr+jf);                // normalize fx by rms           
            m++;                                                                        
            em = _sse_maxE_ps(_pp+jf,_PP+jf);          // dominant pixel energy         
            Ls += ee-em; Eo += ee;                     // subnetwork energy, network energy
            if(ee-em>Es) Ln += ee;                     // network energy above subnet threshold
		}
		
		size_t m4 = m + (m%4 ? 4 - m%4 : 0);
        _E_n = _mm_setzero_ps();                     // + likelihood

        for(j=0; j<m4; j+=4) 
		{                                   
        	int jf = j*f_;                                        
            _sse_dpf4_ps(_Fp+jf,_Fx+jf,_fp+jf,_fx+jf);                // go to DPF
            _E_s = _sse_like4_ps(_fp+jf,_fx+jf,_bb+jf,_BB+jf);        // std likelihood
            _E_n = _mm_add_ps(_E_n,_E_s);                             // total likelihood
        }

	*/	
		//fprintf(fpt,"k = %d l = %d p00[0] = %f p00[1] = %f p00[2] = %f\n", i, l, p00[0], p00[1], p00[2]);
	}
         
	fclose(fpt);
	return 1;

}
               
/*                                     
// sub-network cut with dsp regulator                  
//  lag: lag index                                     
//  snc: sub network threshold, if snc<0 use weak constraint
// hist: diagnostic histogram                               
// return number of processed pixels                        

   if(!net->wc_List[lag].size()) return 0;

   size_t nIFO = net->ifoList.size();
  
   if(nIFO>NIFO) {
      cout<<"network::subNetCut(): invalid network.\n";
      exit(0);                                         
   }

   float   En = 2*net->acor*net->acor*nIFO;  // network energy threshold in the sky loop
   float   Es = 2*net->e2or;                 // subnet energy threshold in the sky loop 
   float   TH = fabs(snc);                   // sub network threshold                   
                                                                                        
   __m128 _En = _mm_set1_ps(En);                                                        
   __m128 _Es = _mm_set1_ps(Es);                                                        
   __m128 _oo = _mm_set1_ps(1.e-12);                                                    
   __m128 _0  = _mm_set1_ps(0.);                                                        
   __m128 _05 = _mm_set1_ps(0.5);                                                       
   __m128 _1  = _mm_set1_ps(1.);                                                        
   __m128* _pe[NIFO];                                                                   

   int f_ = NIFO/4;
   int l,lm,Vm;    
   float Lm,Em,Am,Lo,Eo,Co,Lr,Er,ee,em,To;
   float cc,aa,AA,rHo,stat,Ls,Ln,EE;      

   size_t i,j,k,m,V,V4,id,K,M;
   int  Lsky = int(net->index.size());             // total number of source locations 
   short* mm = net->skyMask.data;                                                      

   float  vvv[NIFO];
   float* v00[NIFO];
   float* v90[NIFO];
   float*  pe[NIFO];
   float*  pa[NIFO];
   float*  pA[NIFO];
   short*  ml[NIFO];
   double* FP[NIFO];
   double* FX[NIFO];
   double  xx[NIFO];

   for(i=0; i<NIFO; i++) {
      if(i<nIFO) {        
         ml[i] = net->getifo(i)->index.data;
         FP[i] = net->getifo(i)->fp.data;   
         FX[i] = net->getifo(i)->fx.data;   
      }                                
      else {                           
         ml[i] = net->getifo(0)->index.data;
         FP[i] = net->getifo(0)->fp.data;   
         FX[i] = net->getifo(0)->fx.data;   
      }                                
   }                                   

   // allocate buffers
   std::vector<int> pI;                      // buffer for pixel IDs
   wavearray<double> cid;                    // buffers for cluster ID
   netpixel* pix;                                                     
   std::vector<int>* vint;                                            
   netcluster* pwc = &net->wc_List[lag];                             
                                                                      
   size_t count = 0;                                                  
   size_t tsize = 0;                                                  

//+++++++++++++++++++++++++++++++++++++++
// loop over clusters                    
//+++++++++++++++++++++++++++++++++++++++

   cid = pwc->get((char*)"ID",  0,'S',0);                 // get cluster ID
                                                                           
   K = cid.size();                                                         
   for(k=0; k<K; k++) {                                   // loop over clusters 
      id = size_t(cid.data[k]+0.1);                                             
      if(pwc->sCuts[id-1] != -2) continue;                // skip rejected/processed clusters 
      vint = &(pwc->cList[id-1]);                         // pixel list                       
      V = vint->size();                                   // pixel list size                  
      if(!V) continue;                                                                        

      pI = net->wdmMRA.getXTalk(pwc, id);

      V = pI.size();                                      // number of loaded pixels
      if(!V) continue;                                                              

      pix = pwc->getPixel(id,pI[0]);
      tsize = pix->tdAmp[0].size(); 
      if(!tsize || tsize&1) {                          // tsize%1 = 1/0 = power/amplitude
         cout<<"network::subNetCut() error: wrong pixel TD data\n";                      
         exit(1);                                                                        
      }                                                                                  
      tsize /= 2;                                                                        
      V4 = V + (V%4 ? 4 - V%4 : 0);                                                      

      //cout<<En<<" "<<Es<<" "<<lag<<" "<<id<<" "<<V4<<" "<<" "<<tsize<<endl;
                                                                             
      std::vector<wavearray<float> > vtd;              // vectors of TD amplitudes
      std::vector<wavearray<float> > vTD;              // vectors of TD amplitudes
      std::vector<wavearray<float> > eTD;              // vectors of TD energies  

      wavearray<float> tmp(tsize*V4); tmp=0;           // aligned array for TD amplitudes 
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
      wavearray<float>  ww(NIFO*V4);  ww=0;            // 00 array for phase-shifted data vectors 
      wavearray<float>  WW(NIFO*V4);  WW=0;            // 90 array for phase-shifted data vectors 
      wavearray<float>  u4(NIFO*4);   u4=0;            // temp array                              
      wavearray<float>  U4(NIFO*4);   U4=0;            // temp array                              

      __m128* _Fp = (__m128*) Fp.data;
      __m128* _Fx = (__m128*) Fx.data;
      __m128* _am = (__m128*) am.data;
      __m128* _AM = (__m128*) AM.data;
      __m128* _xi = (__m128*) xi.data;
      __m128* _XI = (__m128*) XI.data;
      __m128* _fp = (__m128*) fp.data;
      __m128* _fx = (__m128*) fx.data;
      __m128* _nr = (__m128*) nr.data; 
      __m128* _ww = (__m128*) ww.data; 
      __m128* _WW = (__m128*) WW.data; 
      __m128* _bb = (__m128*) bb.data; 
      __m128* _BB = (__m128*) BB.data; 

      for(i=0; i<NIFO; i++) {                          
         vtd.push_back(tmp);                           // array of aligned energy vectors
         vTD.push_back(tmp);                           // array of aligned energy vectors
         eTD.push_back(tmp);                           // array of aligned energy vectors
      }                                                                                  

      for(i=0; i<NIFO; i++) {                          // set up zero deley pointers                   
         pa[i] = vtd[i].data + (tsize/2)*V4;                                                           
         pA[i] = vTD[i].data + (tsize/2)*V4;                                                           
         pe[i] = eTD[i].data + (tsize/2)*V4;                                                           
      }                                                                                                

      net->a_00.resize(NIFO*V4); net->a_00=0.;
      net->a_90.resize(NIFO*V4); net->a_90=0.;
      net->rNRG.resize(V4);      net->rNRG=0.;
      net->pNRG.resize(V4);      net->pNRG=0.;

      __m128* _aa = (__m128*) net->a_00.data;         // set pointer to 00 array
      __m128* _AA = (__m128*) net->a_90.data;         // set pointer to 90 array

      net->pList.clear();
      for(j=0; j<V; j++) {                             // loop over selected pixels 
         pix = pwc->getPixel(id,pI[j]);                // get pixel pointer         
         net->pList.push_back(pix);                    // store pixel pointers for MRA

         double rms = 0.;
         for(i=0; i<nIFO; i++) {
            xx[i] = 1./pix->data[i].noiserms;
            rms += xx[i]*xx[i];                        // total inverse variance
         }                                                                      

         for(i=0; i<nIFO; i++) {
            nr.data[j*NIFO+i]=(float)xx[i]/sqrt(rms);  // normalized 1/rms
            for(l=0; l<tsize; l++) {                                      
               aa = pix->tdAmp[i].data[l];             // copy TD 00 data 
               AA = pix->tdAmp[i].data[l+tsize];       // copy TD 90 data 
               vtd[i].data[l*V4+j] = aa;               // copy 00 data    
               vTD[i].data[l*V4+j] = AA;               // copy 90 data    
               eTD[i].data[l*V4+j] = aa*aa+AA*AA;      // copy power      
            }                                                             
         }                                                                
      }                                                                   

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// first sky loop                                                          
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      int lb = 0;
      int le = Lsky-1;
      bool mra = false;
      double suball=0; 
      double submra=0; 

      stat=Lm=Em=Am=EE=0.; lm=Vm= -1;    


  skyloop:

	FILE *fpt = fopen("skyloop", "a");
	fprintf(fpt, "Now in skyloop V4 = %u le=%d V=%u\n", V4, le, V);
	fclose(fpt);

      for(l=lb; l<=le; l++) {                         // loop over sky locations
         if(!mm[l] || l<0) continue;                  // skip delay configurations
                                                                                  
         _sse_point_ps(_pe, pe, ml, int(l), (int)V4); // point _pe to energy vectors
                                                                                    
         __m128 _msk;                                                               
         __m128 _E_o = _mm_setzero_ps();              // total network energy       
         __m128 _E_n = _mm_setzero_ps();              // network energy above the threshold
         __m128 _E_s = _mm_setzero_ps();              // subnet energy above the threshold 
         __m128 _M_m = _mm_setzero_ps();              // # of pixels above threshold       
         __m128* _rE = (__m128*) net->rNRG.data;      // m128 pointer to energy array      
         __m128* _pE = (__m128*) net->pNRG.data;      // m128 pointer to energy array      

         for(j=0; j<V4; j+=4) {                                // loop over selected pixels 
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

         aa = Ls*Ln/(Eo-Ls);
         if((aa-m)/(aa+m)<0.33) continue;
                                         
         net->pnt_(v00, pa, ml, (int)l, (int)V4);      // pointers to first pixel 00 data 
         net->pnt_(v90, pA, ml, (int)l, (int)V4);      // pointers to first pixel 90 data 
         float* pfp = fp.data;                         // set pointer to fp               
         float* pfx = fx.data;                         // set pointer tp fx               
         float* p00 = net->a_00.data;                 // set pointer for 00 array        
         float* p90 = net->a_90.data;                 // set pointer for 90 array        

         m = 0;
         for(j=0; j<V; j++) { 
            int jf = j*f_;                             // source sse pointer increment 
            net->cpp_(p00,v00);  net->cpp_(p90,v90);   // copy amplitudes with target increment
            net->cpf_(pfp,FP,l); net->cpf_(pfx,FX,l);  // copy antenna with target increment   
            _sse_zero_ps(_xi+jf);                      // zero MRA amplitudes                  
            _sse_zero_ps(_XI+jf);                      // zero MRA amplitudes                  
            _sse_cpf_ps(_am+jf,_aa+jf);                // duplicate 00                         
            _sse_cpf_ps(_AM+jf,_AA+jf);                // duplicate 90                         
            if(net->rNRG.data[j]>En) m++;              // count superthreshold pixels          
         }                                                                                     

         __m128* _pp = (__m128*) am.data;              // point to multi-res amplitudes
         __m128* _PP = (__m128*) AM.data;              // point to multi-res amplitudes

         if(mra) {                                     // do MRA
            _sse_MRA_ps(net,xi.data,XI.data,En,m);     // get principal components
            _pp = (__m128*) xi.data;                   // point to PC amplitudes  
            _PP = (__m128*) XI.data;                   // point to PC amplitudes  
         }                                                                        

         m = 0; Ls=Ln=Eo=0;
         for(j=0; j<V; j++) { 
            int jf = j*f_;                             // source sse pointer increment 
            int mf = m*f_;                             // target sse pointer increment 
            _sse_zero_ps(_bb+jf);                      // reset array for MRA amplitudes
            _sse_zero_ps(_BB+jf);                      // reset array for MRA amplitudes
            ee = _sse_abs_ps(_pp+jf,_PP+jf);           // total pixel energy            
            if(ee<En) continue;                                                         
            _sse_cpf_ps(_bb+mf,_pp+jf);                // copy 00 amplitude/PC          
            _sse_cpf_ps(_BB+mf,_PP+jf);                // copy 90 amplitude/PC          
            _sse_cpf_ps(_Fp+mf,_fp+jf);                // copy F+                       
            _sse_cpf_ps(_Fx+mf,_fx+jf);                // copy Fx                       
            _sse_mul_ps(_Fp+mf,_nr+jf);                // normalize f+ by rms           
            _sse_mul_ps(_Fx+mf,_nr+jf);                // normalize fx by rms           
            m++;                                                                        
            em = _sse_maxE_ps(_pp+jf,_PP+jf);          // dominant pixel energy         
            Ls += ee-em; Eo += ee;                     // subnetwork energy, network energy
            if(ee-em>Es) Ln += ee;                     // network energy above subnet threshold
         }                                                                                     

         size_t m4 = m + (m%4 ? 4 - m%4 : 0);
          _E_n = _mm_setzero_ps();                     // + likelihood

         for(j=0; j<m4; j+=4) {                                   
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

         if(AA>stat && !mra) {
            stat=AA; Lm=Lo; Em=Eo; Am=aa; lm=l; Vm=m; suball=ee; EE=em;
         }                                                             
       }                                                               

      if(!mra && lm>=0) {mra=true; le=lb=lm; goto skyloop;}    // get MRA principle components
                                                                                              
      pwc->sCuts[id-1] = -1;                                                                  
      pwc->cData[id-1].likenet = Lm;                                                          
      pwc->cData[id-1].energy = Em;                                                           
      pwc->cData[id-1].theta = net->nLikelihood.getTheta(lm);                                      
      pwc->cData[id-1].phi = net->nLikelihood.getPhi(lm);                                          
      pwc->cData[id-1].skyIndex = lm;                                                         

      rHo = 0.;
      if(mra) {
         submra = Ls*Eo/(Eo-Ls);                                     // MRA subnet statistic
         submra/= fabs(submra)+fabs(Eo-Lo)+2*(m+6);                  // MRA subnet coefficient 
         To = 0;                                                                               
         pwc->p_Ind[id-1].push_back(lm);                                                       
         for(j=0; j<vint->size(); j++) {                                                       
            pix = pwc->getPixel(id,j);                                                         
            pix->theta = net->nLikelihood.getTheta(lm);                                             
            pix->phi   = net->nLikelihood.getPhi(lm);                                               
            To += pix->time/pix->rate/pix->layers;                                             
            if(j==0&&mra) pix->ellipticity = submra;                 // subnet MRA propagated to L-stage
            if(j==0&&mra) pix->polarisation = fabs(Eo-Lo)+2*(m+6);   // submra NULL propagated to L-stage
            if(j==1&&mra) pix->ellipticity = suball;                 // subnet all-sky propagated to L-stage
            if(j==1&&mra) pix->polarisation = EE;                    // suball NULL propagated to L-stage   
         }                                                                                                  
         To /= vint->size();                                                                                
         rHo = sqrt(Lo*Lo/(Eo+2*m)/nIFO);                            // estimator of coherent amplitude     
      }                                                                                                     
                                                                                                            
      if(hist && rHo>net->netRHO)                                                                          
         for(j=0;j<vint->size();j++) hist->Fill(suball,submra);                                             

      if(fmin(suball,submra)>TH && rHo>net->netRHO) {
         count += vint->size();                       
         if(hist) {                                   
            printf("lag|id %3d|%3d rho=%5.2f To=%5.1f stat: %5.3f|%5.3f|%5.3f ",
                   int(lag),int(id),rHo,To,suball,submra,stat);                 
            printf("E: %6.1f|%6.1f L: %6.1f|%6.1f|%6.1f pix: %4d|%4d|%3d|%2d \n",
                   Em,Eo,Lm,Lo,Ls,int(vint->size()),int(V),Vm,int(m));           
         }                                                                       
      }
      else pwc->sCuts[id-1]=1;

// clean time delay data

      V = vint->size();
      for(j=0; j<V; j++) {                           // loop over pixels
         pix = pwc->getPixel(id,j);
         pix->core = true;
         if(pix->tdAmp.size()) pix->clean();
      }
   }                                                 // end of loop over clusters
   return count;
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
/*
   cout<<"EE="<<EE<<endl;
   EE = 0;
   for(j=0; j<V; ++j) {
      if(pp[j]>=0) EE += ee[j];
      if(pp[j]>=0.) cout<<j<<"|"<<pp[j]<<"|"<<ee[j]<<" ";               // find max pixel
   }
   cout<<"EE="<<EE<<endl;
*/
   return k;
}

