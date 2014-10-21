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
long gpu_subNetCut(network* net, int lag, float snc, TH2F* hist, double *d);

float Lo;
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
	  clock_t start, finish;
      while(1) {
        count = pwc->loadTDampSSE(*net, 'a', cfg->BATCH, cfg->LOUD);
        bench.Continue();	
// add
		start = clock();
#ifdef USE_LOCAL_SUBNETCUT
        psel += subNetCut(net,(int)j,cfg->subnet,NULL);
#else
        psel += net->subNetCut((int)j,cfg->subnet,NULL);
#endif
		finish = clock();
		printf("This Time = %f\n", (double)(finish-start)/CLOCKS_PER_SEC);
// add
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
	double d[10];
	for(int i=0; i<10; i++)
		d[i] = 0;
	count = gpu_subNetCut(net, lag, snc, hist, d);
	
	cout<<"Callback pre cal Time d[0] = "<<d[0]<<endl;
	cout<<"Callback loop Time d[1] = "<<d[1]<<endl;
	cout<<"Callback After Loop Time d[2] = "<<d[2]<<endl;
	cout<<"Allocation Time d[3] = "<<d[3]<<endl;
	cout<<"Loop Cluster Time d[4] = "<<d[4]<<endl;
	cout<<"PRE_GPU Time d[5] = "<<d[5]<<endl;
	cout<<"GPU Time d[6] = "<<d[6]<<endl;
	return count;
}

long Callback(void* post_gpu_data, network *gpu_net, TH2F *gpu_hist, netcluster *pwc, double **FP, double **FX, float **pa, float **pA, size_t *streamCount, double *d)
{
	bool mra = false;
	float vvv[NIFO];
	float *v00[NIFO];
	float *v90[NIFO];
	float *rE, *pE;				//pointers of rNRG.data and pNRG.data
	float Ln = 0;
	float Eo = 0;
	float Ls = 0;
	float aa, AA, En, Es, ee, em, stat, Lm, Em, Am, EE, rHo, To, TH;
	int m = 0;
	int lm, Vm; 
	size_t id, nIFO, V, V4, tsize, count;
	size_t k = 0;
	int f_ =NIFO/4;
	int lb=0;
	int l, le, lag, stream;
	short *ml[NIFO]; 
	short *mm;
	float *eTD[NIFO];
	double suball=0;
	double submra=0;
	double xx[NIFO];
	int Lsky;

	stat=Lm=Em=Am=EE=0.;	lm=Vm= -1;
	count = 0;
	V4 = ((post_data*)post_gpu_data)->other_data.V4;
	le = ((post_data*)post_gpu_data)->other_data.le;
	k = ((post_data*)post_gpu_data)->other_data.k;
	En = ((post_data*)post_gpu_data)->other_data.T_En;
	Es = ((post_data*)post_gpu_data)->other_data.T_Es;
	TH = ((post_data*)post_gpu_data)->other_data.TH;
	lag = ((post_data*)post_gpu_data)->other_data.lag;
	stream = ((post_data*)post_gpu_data)->other_data.stream;
	id = ((post_data*)post_gpu_data)->other_data.id;
	nIFO = ((post_data*)post_gpu_data)->other_data.nIFO;
	V = ((post_data*)post_gpu_data)->other_data.V;
	tsize = ((post_data*)post_gpu_data)->other_data.tsize;
	
	Lsky = le + 1;
	for(int i=0; i<NIFO; i++)
		ml[i] = ((post_data*)post_gpu_data)->other_data.ml_mm + i*Lsky;
	mm = ((post_data*)post_gpu_data)->other_data.ml_mm + 4*Lsky;
	rE = ((post_data*)post_gpu_data)->output.output;
	
	std::vector<int> pI;				// buffer for pixel TDs
	wavearray<float> tmp(tsize*V4); tmp=0;
	wavearray<float>  fp(NIFO*V4);  fp=0;     // aligned array for + antenna pattern 
	wavearray<float>  fx(NIFO*V4);  fx=0;     // aligned array for x antenna pattern 
	wavearray<float>  nr(NIFO*V4);  nr=0;            // aligned array for inverse rms 
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
	__m128* _bb = (__m128*) bb.data; 
	__m128* _BB = (__m128*) BB.data; 
	__m128* _nr = (__m128*) nr.data;

	__m128 _E_n = _mm_setzero_ps();		// network energy above the threshold
    	__m128 _E_s = _mm_setzero_ps();		// subnet energy above the threshold 
	netpixel* pix;	
	std::vector<int> *vint;

	gpu_net->a_00.resize(NIFO*V4);	gpu_net->a_00=0.;
	gpu_net->a_90.resize(NIFO*V4);	gpu_net->a_90=0.;
	__m128* _aa = (__m128*) gpu_net->a_00.data;         // set pointer to 00 array
    	__m128* _AA = (__m128*) gpu_net->a_90.data;         // set pointer to 90 array

	gpu_net->rNRG.resize(V4);	gpu_net->rNRG=0.;
	gpu_net->pNRG.resize(V4);	gpu_net->pNRG=0.;
	
	pI = gpu_net->wdmMRA.getXTalk(pwc, id);
	gpu_net->pList.clear();
	for(int j=0; j<V; j++)			// loop over selected pixels
	{
		pix = pwc->getPixel(id, pI[j]);	// get pixel pointer
		gpu_net->pList.push_back(pix);
		double rms = 0.;
		for(int i=0; i<nIFO; i++)
		{
			xx[i] = 1./pix->data[i].noiserms;
			rms += xx[i]*xx[i];	// total inverse variance
		}
		
		for(int i=0; i<nIFO; i++)
			nr.data[j*NIFO+i]=(float)xx[i]/sqrt(rms);	// normalized 1/rms
	}
	int rEDim = Lsky * V4;
	
skyloop:
	// after skyloop
	for(l=lb; l<=le; l++)
	{
		if(!mm[l] || l<0) continue; 
		
		aa = ((post_data*)post_gpu_data)->output.output[rEDim+l];
		if(aa == -1)	continue;
		for(int j=0; j<V; j++)
		{
			gpu_net->rNRG.data[j] = rE[l*V4+j];
		}
		gpu_net->pnt_(v00, pa, ml, (int)l, (int)V4);	// pointers to first pixel 00 data
		gpu_net->pnt_(v90, pA, ml, (int)l, (int)V4);	// pointers to first pixel 90 data
		
		float *pfp = fp.data;
		float *pfx = fx.data;
		float *p00 =gpu_net->a_00.data;
		float *p90 =gpu_net->a_90.data;
		
		m = 0;
		for(int j=0; j<V; j++)
		{
			int jf= j*f_;
			gpu_net->cpp_(p00,v00);	gpu_net->cpp_(p90,v90);			// copy amplitudes with target increment
			gpu_net->cpf_(pfp,FP,l);gpu_net->cpf_(pfx,FX,l);		// copy antenna with target increment
			_sse_zero_ps(_xi+jf);                      // zero MRA amplitudes
	        	_sse_zero_ps(_XI+jf);                      // zero MRA amplitudes
	           	_sse_cpf_ps(_am+jf,_aa+jf);                // duplicate 00
        	   	_sse_cpf_ps(_AM+jf,_AA+jf);                // duplicate 90 
           		if(gpu_net->rNRG.data[j]>En) m++;              // count superthreshold pixels
		}
		
	        __m128* _pp = (__m128*) am.data;              // point to multi-res amplitudes
        	__m128* _PP = (__m128*) AM.data;              // point to multi-res amplitudes
		
		if(mra)										// do MRA
		{
			_sse_MRA_ps(gpu_net, xi.data, XI.data, En, m);	// get principal components
			_pp = (__m128*) xi.data;						// point to PC amplitudes
			_PP = (__m128*) XI.data;						// point to Pc amplitudes
		}		
		
		m = 0; Ls=Ln=Eo=0;
		for(int j=0; j<V; j++)
		{
			int jf = j*f_;	// source sse pointer increment 
	            	int mf = m*f_;  // target sse pointer increment 
        	    	_sse_zero_ps(_bb+jf);	// reset array for MRA amplitudes
	            	_sse_zero_ps(_BB+jf);       // reset array for MRA amplitudes
        	    	ee = _sse_abs_ps(_pp+jf,_PP+jf);	// total pixel energy
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
	    _E_n = _mm_setzero_ps();                     // + likelihood

		for(int j=0; j<m4; j+=4) 
		{                                   
	       	    int jf = j*f_;                                        
	            _sse_dpf4_ps(_Fp+jf,_Fx+jf,_fp+jf,_fx+jf);	// go to DPF
        	    _E_s = _sse_like4_ps(_fp+jf,_fx+jf,_bb+jf,_BB+jf);	// std likelihood
	            _E_n = _mm_add_ps(_E_n,_E_s);            	      	// total likelihood
        	}
	        _mm_storeu_ps(vvv,_E_n);                                                        
		
		Lo = vvv[0]+vvv[1]+vvv[2]+vvv[3];
        	AA = aa/(fabs(aa)+fabs(Eo-Lo)+2*m*(Eo-Ln)/Eo);        //  subnet stat with threshold
        	ee = Ls*Eo/(Eo-Ls); 
        	em = fabs(Eo-Lo)+2*m;	//  suball NULL
        	ee = ee/(ee+em);       	//  subnet stat without threshold
       
		aa = (aa-m)/(aa+m);                                                                    
			
		if(AA>stat && !mra)
		{
			stat=AA; Lm=Lo; Em=Eo; Am=aa; lm=l; Vm=m; suball=ee; EE=em;
		}
		
	}
    if(!mra && lm>=0) {mra=true; le=lb=lm; goto skyloop;}    // get MRA principle components
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
		submra = Ls*Eo/(Eo-Ls);		// MRA subnet statistic
		submra /= fabs(submra)+fabs(Eo-Lo)+2*(m+6);	// MRA subnet coefficient
		To = 0;
        pwc->p_Ind[id-1].push_back(lm); 
		for(int j=0; j<vint->size(); j++)
		{
			pix = pwc->getPixel(id,j);
			pix->theta = gpu_net->nLikelihood.getTheta(lm);
            pix->phi   = gpu_net->nLikelihood.getPhi(lm);
            To += pix->time/pix->rate/pix->layers;
            if(j==0&&mra) pix->ellipticity = submra;	// subnet MRA propagated to L-stage
            if(j==0&&mra) pix->polarisation = fabs(Eo-Lo)+2*(m+6);   // submra NULL propagated to L-stage
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
		pwc->sCuts[id-1]=1;

// clean time delay data
	V = vint->size();
	for(int j=0; j<V; j++)	// loop over pixels
	{
		pix = pwc->getPixel(id,j);
		pix->core = true;
		if(pix->tdAmp.size())
			pix->clean();
	}
	streamCount[stream] += count;
	return 0; 
}
               
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

