//////////////////////////////////////////////////////////
//   class for WaveBurst event 
//   Sergey Klimenko, University of Florida
//////////////////////////////////////////////////////////


#ifndef wbevent_h
#define wbevent_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

/* Structure of WaveBurst triple IFO event */

class wbsingle;

class wbevent {
   public :

  TTree          *fChain;        //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent;      //!current Tree number in a TChain
  Int_t           ndim;          //! event dimension

//Declaration of leaves types
// for arrays: ifo1 - first index, ifo2 - second index, .....
  Int_t           run;            // run ID                                                       
  Int_t           nevent;         // event count                                                  
  Int_t*          ifo;            // ifo ID: 1/2/3/.. - L1/H1/H2/..                            
  Int_t           eventID;        // event ID                                                     
  Int_t           type;           // event type             
  Int_t           stype;          // simulation type                                              
  Int_t*          rate;           // 1/rate - wavelet time resolution
				  
  Int_t*          volume;         // cluster volume                                               
  Int_t*          size;           // cluster size (black pixels only)                             
  Int_t           usize;          // cluster union size                                           
				  
  Float_t*        gap;            // time between consecutive events                              
  Float_t*        shift;          // time delay                                                   
  Double_t        strain;         // strain of injected simulated signals                         
  Float_t         phi;            // phi angle
  Float_t         teta;           // teta angle
  Float_t*        bp;             // beam pattern coefficients for hp
  Float_t*        bx;             // beam pattern coefficients for hx 
				  
  Float_t*        power;          // energy/noise_variance/size                                   
  Float_t*        rSNR;           // rank SNR = rLH*sqrt(3)+1.69*size (10% bpp)                   
  Float_t*        gSNR;           // gaussian snr = gLH*sqrt(3)+1.69*size (10% bpp)               
  Float_t*        rLH;            // rank likelihood                                              
  Float_t*        gLH;            // Gaussian likelihood                                          
  Float_t*        rSF;            // rank significance
  Float_t*        gSF;            // Gaussians significance
				  
  Double_t*       time;           // average center_of_snr time
  Double_t*       itime;          // injection time: 0 - LLO, 1 - LHO                           
  Float_t*        right;          // min cluster time                                   
  Float_t*        left;           // max cluster time                                    
  Float_t*        duration;       // cluster duration = stopW-startW
  Double_t*       start;          // actual start GPS time
  Double_t*       stop;           // actual stop GPS time 
				  
  Float_t*        frequency;      // average center_of_snr frequency
  Float_t*        low;            // min frequency 
  Float_t*        high;           // max frequency 
  Float_t*        bandwidth;      // high-low 
  Double_t*       hrss;           // log10(calibrated hrss)
  Double_t*       noise;          // log10(calibrated noise rms)
  Float_t*        asymmetry;      // sign x-correlation

//List of branches

   TBranch        *b_run;   //!
   TBranch        *b_nevent;   //!
   TBranch        *b_ifo;   //!
   TBranch        *b_eventID;   //!
   TBranch        *b_type;   //!
   TBranch        *b_stype;   //!
   TBranch        *b_rate;   //!

   TBranch        *b_volume;   //!
   TBranch        *b_size;   //!
   TBranch        *b_usize;   //!

   TBranch        *b_gap;   //!
   TBranch        *b_shift;   //!
   TBranch        *b_strain;   //!
   TBranch        *b_phi;   //!
   TBranch        *b_teta;   //!
   TBranch        *b_bp;   //!
   TBranch        *b_bx;   //!

   TBranch        *b_power;   //!
   TBranch        *b_rLH;   //!
   TBranch        *b_gLH;   //!
   TBranch        *b_rSNR;   //!
   TBranch        *b_gSNR;   //!
   TBranch        *b_rSF;   //!
   TBranch        *b_gSF;   //!

   TBranch        *b_time;   //!
   TBranch        *b_itime;   //!
   TBranch        *b_right;   //!
   TBranch        *b_left;   //!
   TBranch        *b_duration;   //!
   TBranch        *b_start;   //!
   TBranch        *b_stop;   //!

   TBranch        *b_frequency;   //!
   TBranch        *b_low;   //!
   TBranch        *b_high;   //!
   TBranch        *b_bandwidth;   //!

   TBranch        *b_hrss;   //!
   TBranch        *b_noise;   //!
   TBranch        *b_asymmetry;   //!

   wbevent() {fChain=NULL;};

   wbevent() :
      run(0),nevent(0),ifo(NULL),eventID(0),type(1),stype(0),rate(NULL),       
      volume(NULL),size(NULL),usize(0),gap(NULL),shift(NULL),strain(0),      
      phi(0),teta(0),bp(NULL),bx(NULL),power(NULL),rSNR(NULL),gSNR(NULL),       
      rLH(NULL),gLH(NULL),rSF(NULL),gSF(NULL),time(NULL),itime(NULL),
      right(NULL),left(NULL),duration(NULL),start(NULL),stop(NULL),
      frequency(NULL),low(NULL),high(NULL),bandwidth(NULL),hrss(NULL),       
      noise(NULL),asymmetry(NULL),  
   { ndim=1; allocate(); return; }

   wbevent(int n) :
      run(0),nevent(0),ifo(NULL),eventID(0),type(1),stype(0),rate(NULL),       
      volume(NULL),size(NULL),usize(0),gap(NULL),shift(NULL),strain(0),      
      phi(0),teta(0),bp(NULL),bx(NULL),power(NULL),rSNR(NULL),gSNR(NULL),       
      rLH(NULL),gLH(NULL),rSF(NULL),gSF(NULL),time(NULL),itime(NULL),
      right(NULL),left(NULL),duration(NULL),start(NULL),stop(NULL),
      frequency(NULL),low(NULL),high(NULL),bandwidth(NULL),hrss(NULL),       
      noise(NULL),asymmetry(NULL),  
   { ndim=n; allocate(); return; }

   wbevent(const wbevent& a) :
      run(0),nevent(0),ifo(NULL),eventID(0),type(1),stype(0),rate(NULL),       
      volume(NULL),size(NULL),usize(0),gap(NULL),shift(NULL),strain(0),      
      phi(0),teta(0),bp(NULL),bx(NULL),power(NULL),rSNR(NULL),gSNR(NULL),       
      rLH(NULL),gLH(NULL),rSF(NULL),gSF(NULL),time(NULL),itime(NULL),
      right(NULL),left(NULL),duration(NULL),start(NULL),stop(NULL),
      frequency(NULL),low(NULL),high(NULL),bandwidth(NULL),hrss(NULL),       
      noise(NULL),asymmetry(NULL),  
   { ndim=a.ndim; allocate(); *this = a; return; }

   wbevent(TTree *tree) {if(tree) Init(tree);};

   virtual ~wbevent() { 
      if (fChain)      free(fChain->GetCurrentFile());
      if (ifo)         free(ifo);         // ifo ID: 1/2/3/.. - L1/H1/H2/..                            
      if (rate)        free(rate);        // 1/rate - wavelet time resolution
      
      if (volume)      free(volume);      // cluster volume                                               
      if (size)        free(size);        // cluster size (black pixels only)                             
      
      if (gap)         free(gap);         // time between consecutive events                              
      if (shift)       free(shift);       // time delay                                                   
      if (bp)          free(bp);          // beam pattern coefficients for hp
      if (bx)          free(bx);          // beam pattern coefficients for hx 
      
      if (power)       free(power);       // energy/noise_variance/size                                   
      if (rSNR)        free(rSNR);        // rank SNR = rLH*sqrt(3)+1.69*size (10% bpp)                   
      if (gSNR)        free(gSNR);        // gaussian snr = gLH*sqrt(3)+1.69*size (10% bpp)               
      if (rLH)         free(rLH);         // rank likelihood                                              
      if (gLH)         free(gLH);         // Gaussian likelihood                                          
      if (rSF)         free(rSF);         // rank significance
      if (gSF)         free(gSF);         // Gaussians significance
      
      if (time)        free(time);        // average center_of_snr time
      if (itime)       free(itime);       // injection time: 0 - LLO, 1 - LHO                           
      if (right)       free(right);       // min cluster time                                   
      if (left)        free(left);        // max cluster time                                    
      if (duration)    free(duration);    // cluster duration = stopW-startW
      if (start)       free(start);       // actual start GPS time
      if (stop)        free(stop);        // actual stop GPS time 
      
      if (frequency)   free(frequency);   // average center_of_snr frequency
      if (low)         free(low);         // min frequency 
      if (high)        free(high);        // max frequency 
      if (bandwidth)   free(bandwidth);   // high-low 
      if (hrss)        free(hrss);        // log10(calibrated hrss)
      if (noise)       free(noise);       // log10(calibrated noise rms)
      if (asymmetry)   free(asymmetry);
   };

   virtual wbevent& operator=(wbevent &);

   Int_t  GetEntry(Int_t);
   void   allocate();
   void   Init(TTree *);
   Bool_t Notify();
   TTree* setTree();
   void setfrom(int n, wbsingle** p);

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);
   void   Show(Int_t entry = -1);

};

#endif

 
 



































