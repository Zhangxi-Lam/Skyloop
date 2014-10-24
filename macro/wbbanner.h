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
// for arrays: ifo1 - first index, ifo2 - second index

  Int_t           run;               // run ID                                                       
  Int_t           nevent;            // event count                                                  
  Int_t           ifo;               // ifo ID: 1/2/3 - L1/H1/H2                                     
  Int_t           type;              // run type             
  Int_t           rate;              // data sampling rate
  Int_t           level;             // lowest wavelet level
  Int_t           order;             // wavelet order                                              

  Int_t           volume;            // cluster volume                                               
  Int_t           size;              // cluster size (black pixels only)                             
  Int_t           usize;             // cluster union size                                           

  Float_t         gap;               // time between consecutive events                              
  Float_t         offset;            // wavelet baundary offset                                                   
  Double_t        start;             // gps start time of WB interval                           
  Double_t        stop;              // gps stop time of WB interval                         

  Float_t         power;             // energy/noise_variance/size                                   
  Float_t         rSNR;              // rank SNR = rLH*sqrt(3)+1.69*size (10% bpp)                   
  Float_t         gSNR;              // gaussian snr = gLH*sqrt(3)+1.69*size (10% bpp)               
  Float_t         rLH;               // rank likelihood                                              
  Float_t         gLH;               // Gaussian likelihood                                          
  Float_t         rSF;               // rank significance                                              
  Float_t         gSF;               // Gaussians significance                                         

  Double_t        time;              // average center_of_snr time                                   
  Double_t        itime;             // injection time: 0 - LLO, 1 - LHO                           
  Float_t         start;             // min cluster time                                   
  Float_t         stop;              // max cluster time                                    
  Float_t         duration;          // cluster duration = stopW-startW                              
  Double_t        startGPS;          // actual start GPS time                                        
  Double_t        stopGPS;           // actual stop GPS time                                         

  Float_t         frequency;         // average center_of_snr frequency                              
  Float_t         low;               // min frequency                                                
  Float_t         high;              // max frequency                                                
  Float_t         bandwidth;         // high-low                                                     

  Double_t        noise;             // log10(calibrated noise rms)                                  
  Float_t         xcorrelation;      // sign x-correlation                                           

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

   TBranch        *b_power;   //!
   TBranch        *b_rLH;   //!
   TBranch        *b_gLH;   //!
   TBranch        *b_rSNR;   //!
   TBranch        *b_gSNR;   //!
   TBranch        *b_rSF;   //!
   TBranch        *b_gSF;   //!

   TBranch        *b_time;   //!
   TBranch        *b_itime;   //!
   TBranch        *b_start;   //!
   TBranch        *b_stop;   //!
   TBranch        *b_duration;   //!
   TBranch        *b_startGPS;   //!
   TBranch        *b_stopGPS;   //!

   TBranch        *b_frequency;   //!
   TBranch        *b_low;   //!
   TBranch        *b_high;   //!
   TBranch        *b_bandwidth;   //!

   TBranch        *b_noise;   //!
   TBranch        *b_xcorrelation;   //!

   wbsingle() {fChain=NULL;};
   wbsingle(TTree *tree) {if(tree) Init(tree);};
   virtual ~wbsingle() { if (!fChain) return; delete fChain->GetCurrentFile(); };

   virtual wbsingle& operator<<(wbsingle &);

   Int_t  GetEntry(Int_t);
   void   Init(TTree *);
   Bool_t Notify();
   TTree* setTree();
   void output(TTree* wave_tree, wavecluster** p);

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);

   void   Show(Int_t entry = -1);

};

#endif




