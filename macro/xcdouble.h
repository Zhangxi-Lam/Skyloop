//////////////////////////////////////////////////////////
//   x-correlation class for root
//   Sergey Klimenko, University of Florida
//////////////////////////////////////////////////////////


#ifndef xcdouble_h
#define xcdouble_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

/* Structure of WaveBurst single IFO event */

class xcdouble {
   public :

  TTree          *fChain;        //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent;      //!current Tree number in a TChain

//Declaration of leaves types

  Int_t           nevent;            // event count
  Int_t           ifo;               // ifopair ID: 1/2/3 - L1H1/H1H2/H2L1
  Float_t         xcor;              // x-correlation                                   
  Float_t         xlag;              // x-correlation time lag                                   
  Float_t         shift;             // global time shift between time series
  Double_t        time;              // time stamp                                   

//List of branches

   TBranch        *b_nevent;   //!
   TBranch        *b_ifo;     //!
   TBranch        *b_xcor;   //!
   TBranch        *b_xlag;   //!
   TBranch        *b_shift;   //!
   TBranch        *b_time;   //!

   xcdouble() {fChain=NULL;};
   xcdouble(TTree *tree) {if(tree) Init(tree);};
   virtual ~xcdouble() { if (!fChain) return; delete fChain->GetCurrentFile(); };

   virtual xcdouble& operator=(xcdouble &);

   Int_t  GetEntry(Int_t);
   void   Init(TTree *);
   Bool_t Notify();
   TTree* setTree();
   void output(TTree*, wavecor&);

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);

   void   Show(Int_t entry = -1);

};

#endif




