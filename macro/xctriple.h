//////////////////////////////////////////////////////////
//   class for triple IFO WaveBurst x-correlation event 
//   Sergey Klimenko, University of Florida
//////////////////////////////////////////////////////////


#ifndef xctriple_h
#define xctriple_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

/* Structure of WaveBurst triple IFO event for x-correlation */

class xctriple {
   public :

  TTree          *fChain;        //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent;      //!current Tree number in a TChain

//Declaration of leaves types
// for arrays: ifo1 - first index, ifo2 - second index

  Int_t           run;               // run ID                                                       
  Int_t           nevent;            // event count                                                  
  Int_t           ifo[3];            // ifo ID: 1/2/3 - L1H1/H1H2/H2L1
  Int_t           eventID;           // event ID                                                     
  Int_t           type;              // event type 0/1/2/3/4 - gw/bkgr/rand1/rand2/rand3             
  Int_t           stype;             // simulation type                                              

  Int_t           size[3];           // cluster size
  Int_t           usize;             // cluster union size                                           

  Float_t         gap[3];            // time between consecutive events                              
  Float_t         shift[3];          // time delay                                                   
  Double_t        strain;            // strain of injected simulated signals                         

  Float_t         xcor[3];           // cluster mean x-correlation 
  Float_t         XCOR[3];           // maximum x-correlation in the cluster
  Float_t         xlag[3];           // cluster mean x-correlation lag
  Float_t         XLAG[3];           // maximum x-correlation lag in the cluster
  Float_t         aSF[3];            // significance for average x-correlation
  Float_t         mSF[3];            // significance for maximum x-correlation

  Double_t        time[3];           // average center time
  Double_t        itime[3];          // injection time: 0 - LLO, 1 - LHO                           
  Float_t         duration[3];       // cluster duration = stopW-startW
  Double_t        start[3];          // actual start GPS time
  Double_t        stop[3];           // actual stop GPS time 


//List of branches

   TBranch        *b_run;   //!
   TBranch        *b_nevent;   //!
   TBranch        *b_ifo;   //!
   TBranch        *b_eventID;   //!
   TBranch        *b_type;   //!
   TBranch        *b_stype;   //!

   TBranch        *b_size;   //!
   TBranch        *b_usize;   //!

   TBranch        *b_gap;   //!
   TBranch        *b_shift;   //!
   TBranch        *b_strain;   //!

   TBranch        *b_xcor;   //!
   TBranch        *b_XCOR;   //!
   TBranch        *b_xlag;   //!
   TBranch        *b_XLAG;   //!
   TBranch        *b_aSF;   //!
   TBranch        *b_mSF;   //!

   TBranch        *b_time;   //!
   TBranch        *b_itime;   //!
   TBranch        *b_duration;   //!
   TBranch        *b_start;   //!
   TBranch        *b_stop;   //!

   xctriple() {fChain=NULL;}
   xctriple(TTree *tree) {if(tree) Init(tree);}
   virtual ~xctriple() { if(!fChain) return; delete fChain->GetCurrentFile(); }

   virtual xctriple& operator=(xctriple &);

   Int_t  GetEntry(Int_t);
   void   Init(TTree *);
   Bool_t Notify();
   TTree* setTree();

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);

   void   Show(Int_t entry = -1);

};

#endif




