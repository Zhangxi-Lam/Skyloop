//////////////////////////////////////////////////////////
//   root tree class to store live time   
//   Sergey Klimenko, University of Florida
//////////////////////////////////////////////////////////


#ifndef livetime_h
#define livetime_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "../wat.hh"

/* Structure of WaveBurst single IFO event */

class network;

class livetime {
   public :

  TTree          *fChain;        //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent;      //!current Tree number in a TChain

//Declaration of leaves types
// for arrays: ifo1 - first index, ifo2 - second index

  Int_t           run;            // run                                                  
  Double_t        gps;            // segment gps start time
  Double_t        live;           // live time                                   
  Float_t*        lag;            // time lag [sec]                                                   
  Float_t*        slag;           // time slag [sec]                                                   

//List of branches

   TBranch        *b_run;   //!
   TBranch        *b_gps;   //!
   TBranch        *b_live;  //!
   TBranch        *b_lag;   //!
   TBranch        *b_slag;   //!

   livetime() : lag(NULL), slag(NULL) {allocate(); fChain=NULL;};
   livetime(TTree *tree) : lag(NULL), slag(NULL) {allocate(); if(tree) Init(tree);};
   virtual ~livetime() { 
     if (lag)         free(lag);         // lag array                            
     if (slag)        free(slag);        // slag array                            
     if (fChain)      delete fChain->GetCurrentFile();
     return;
   };

   virtual livetime& operator=(livetime &);

   Int_t  GetEntry(Int_t);
   void   Init(TTree *);
   void   allocate();
   Bool_t Notify();
   TTree* setTree();
   void output(TTree* waveTree, network* net, float* slag=NULL);

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);

   void   Show(Int_t entry = -1);

};

#endif




