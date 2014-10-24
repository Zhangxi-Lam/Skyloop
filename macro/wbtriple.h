//////////////////////////////////////////////////////////
//   class for triple IFO WaveBurst event 
//   Sergey Klimenko, University of Florida
//////////////////////////////////////////////////////////


#ifndef wbtriple_h
#define wbtriple_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

/* Structure of WaveBurst triple IFO event */

class wbsingle;

class wbtriple {
   public :

  TTree          *fChain;        //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent;      //!current Tree number in a TChain

//Declaration of leaves types
// for arrays: ifo1 - first index, ifo2 - second index

  Int_t           run;               // run ID                                                       
  Int_t           nevent;            // event count                                                  
  Int_t           ifo[3];            // ifo ID: 1/2/3 - L1/H1/H2                                     
  Int_t           eventID;           // event ID                                                     
  Int_t           type;              // event type 0/1/2/3/4 - gw/bkgr/rand1/rand2/rand3             
  Int_t           stype;             // simulation type                                              
  Int_t           rate[3];           // 1/rate - wavelet time resolution
				  
  Int_t           volume[3];         // cluster volume                                               
  Int_t           size[3];           // cluster size (black pixels only)                             
  Int_t           usize;             // cluster union size                                           
				  
  Float_t         gap[3];            // time between consecutive events                              
  Float_t         shift[3];          // time delay                                                   
  Double_t        strain;            // strain of injected simulated signals                         
  Float_t         phi;               // phi angle
  Float_t         teta;              // teta angle
  Float_t         bp[3];             // beam pattern coefficients for hp
  Float_t         bx[3];             // beam pattern coefficients for hx 
				  
  Float_t         power[3];          // energy/noise_variance/size                                   
  Float_t         rSNR[3];           // rank SNR = rLH*sqrt(3)+1.69*size (10% bpp)                   
  Float_t         gSNR[3];           // gaussian snr = gLH*sqrt(3)+1.69*size (10% bpp)               
  Float_t         rLH[3];            // rank likelihood                                              
  Float_t         gLH[3];            // Gaussian likelihood                                          
  Float_t         rSF[3];            // rank significance
  Float_t         gSF[3];            // Gaussians significance
				  
  Double_t        time[3];           // average center_of_snr time
  Double_t        itime[3];          // injection time: 0 - LLO, 1 - LHO                           
  Float_t         right[3];          // min cluster time                                   
  Float_t         left[3];           // max cluster time                                    
  Float_t         duration[3];       // cluster duration = stopW-startW
  Double_t        start[3];          // actual start GPS time
  Double_t        stop[3];           // actual stop GPS time 
				  
  Float_t         frequency[3];      // average center_of_snr frequency
  Float_t         low[3];            // min frequency 
  Float_t         high[3];           // max frequency 
  Float_t         bandwidth[3];      // high-low 
  Double_t        hrss[3];           // log10(calibrated hrss)
  Double_t        noise[3];          // log10(calibrated noise rms)
  Float_t         asymmetry[3];      // sign x-correlation

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

   wbtriple() {fChain=NULL;};
   wbtriple(TTree *tree) {if(tree) Init(tree);};
   virtual ~wbtriple() { if (!fChain) return; delete fChain->GetCurrentFile(); };

   virtual wbtriple& operator=(wbtriple &);

   Int_t  GetEntry(Int_t);
   void   Init(TTree *);
   Bool_t Notify();
   TTree* setTree();
   void setfrom(wbsingle** p);

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);

   void   Show(Int_t entry = -1);

};

#endif




