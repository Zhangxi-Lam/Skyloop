//////////////////////////////////////////////////////////
//   class for single IFO WaveBurst event 
//   Sergey Klimenko, University of Florida
//////////////////////////////////////////////////////////


#ifndef wbsingle_h
#define wbsingle_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

/* Structure of WaveBurst single IFO event */

class wavecluster;

class wbsingle {
   public :

  TTree          *fChain;        //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent;      //!current Tree number in a TChain

//Declaration of leaves types

  Int_t           run;         // run ID                                                             
  Int_t           nevent;      // event count                                                        
  Int_t           ifo;         // ifo ID: 1/2/3 - L1/H1/H2                                           
  Int_t           eventID;     // event ID                                                           
  Int_t           type;        // event type 0/1/2/3/4 - gw/bkgr/rand1/rand2/rand3                   
  Int_t           stype;       // simulation type                                                    
  Int_t           rate;        // 1/rate - wavelet time resolution		
  Int_t           volume;      // cluster volume                                                     
  Int_t           size;        // cluster size (black pixels only)                                   
  Int_t           usize;       // cluster union size                                                 
  Float_t         gap;         // time between consecutive events                                    
  Float_t         shift;       // time delay
			      
  Double_t        strain;      // strain of injected simulated signals                               
  Float_t         phi;         // phi angle
  Float_t         teta;        // teta angle
  Float_t         bp;          // beam pattern coefficient for hp
  Float_t         bx;          // beam pattern coefficient for hx                                    
			      
  Float_t         power;       // energy/noise_variance/size                                         
  Float_t         rSNR;        // rank SNR = rLH*sqrt(3)+1.69*size (10% bpp)                         
  Float_t         gSNR;        // gaussian snr = gLH*sqrt(3)+1.69*size (10% bpp)                     
  Float_t         rLH;         // rank likelihood                                                    
  Float_t         gLH;         // Gaussian likelihood                                                
  Float_t         rSF;         // rank significance                                                    
  Float_t         gSF;         // Gaussians significance                                               
			      
  Double_t        time;        // central GPS time in zero lag frame    
  Double_t        itime;       // GPS injection time: 0 - LLO, 1 - LHO                                 
  Float_t         right;       // segment_stop-trigger_stop time interval  
  Float_t         left;        // trigger_start-segment_start time interval
  Float_t         duration;    // cluster duration = stopW-startW                                    
  Double_t        start;       // actual trigger start GPS time     
  Double_t        stop;        // actual trigger stop GPS time      
			      
  Float_t         frequency;   // average center_of_snr frequency                                    
  Float_t         low;         // min frequency                                                      
  Float_t         high;        // max frequency                                                      
  Float_t         bandwidth;   // high-low                                                           
  Double_t        hrss;        // calibrated hrss                                          
  Double_t        noise;       // calibrated noise hrss averaged over cluster 
  Float_t         asymmetry;   // cluster asymmetry (sign x-correlation) 

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

   wbsingle() {fChain=NULL;};
   wbsingle(TTree *tree) {if(tree) Init(tree);};
   virtual ~wbsingle() { if (!fChain) return; delete fChain->GetCurrentFile(); };

   virtual wbsingle& operator=(wbsingle &);

   Int_t  GetEntry(Int_t);
   void   Init(TTree *);
   Bool_t Notify();
   TTree* setTree();
   void output(TTree* wave_tree, wavecluster** p, int np=1);

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);

   void   Show(Int_t entry = -1);

};

#endif




