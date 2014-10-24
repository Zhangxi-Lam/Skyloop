//////////////////////////////////////////////////////////
//   noise variability class 
//   Sergey Klimenko, University of Florida
//////////////////////////////////////////////////////////


#ifndef variability_h
#define variability_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

/* Structure of WaveBurst single IFO event */

class variability {
   public :

  TTree          *fChain;        //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent;      //!current Tree number in a TChain

//Declaration of leaves types
// for arrays: ifo1 - first index, ifo2 - second index

  Int_t           nevent;            // event count                                                  
  Int_t           ifo;               // ifo ID: 1/2/3 - L1/H1/H2                                     
  Float_t         value;             // noise variability                                   
  Double_t        time;              // average center_of_snr time                                   
  Double_t        gps;               // segment gps time                                   

//List of branches

   TBranch        *b_nevent;   //!
   TBranch        *b_ifo;   //!
   TBranch        *b_value;   //!
   TBranch        *b_time;   //!
   TBranch        *b_gps;   //!

   variability() {fChain=NULL;};
   variability(TTree *tree) {if(tree) Init(tree);};
   virtual ~variability() { if (!fChain) return; delete fChain->GetCurrentFile(); };

   virtual variability& operator<<(variability &);

   Int_t  GetEntry(Int_t);
   void   Init(TTree *);
   Bool_t Notify();
   TTree* setTree();
   void output(TTree*, wavearray<float>*, int=0, double=0.);

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);

   void   Show(Int_t entry = -1);

};

#endif




