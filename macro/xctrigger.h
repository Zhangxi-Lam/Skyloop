//////////////////////////////////////////////////////////
//   class for triple IFO WaveBurst x-correlation event 
//   Sergey Klimenko, University of Florida
//////////////////////////////////////////////////////////


#ifndef xctrigger_h
#define xctrigger_h

#include <stdio.h>
#include <stdlib.h>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

/* Structure of WaveBurst triple IFO event for x-correlation */

class xctrigger {
   public :

  TTree          *fChain;        //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent;      //!current Tree number in a TChain
  Int_t           ndim;          //! trigger dimension

//Declaration of leaves types
// for arrays: ifo1 - first index, ifo2 - second index

  Int_t           run;               // run ID                                                       
  Int_t           nevent;            // event count                                                  
  Int_t*          ifo;               // ifo ID: 1/2/3 - L1H1/H1H2/H2L1
  Int_t           eventID;           // event ID                                                     
  Int_t           type;              // event type 0/1/2/3/4 - gw/bkgr/rand1/rand2/rand3             
  Int_t           stype;             // simulation type                                              

  Int_t*          size;              // cluster size
  Int_t           usize;             // cluster union size                                           

  Float_t*        gap;               // time between consecutive events                              
  Float_t*        shift;             // time delay                                                   
  Float_t*        xcor;              // cluster mean x-correlation 
  Float_t*        XCOR;              // maximum x-correlation in the cluster
  Float_t*        xlag;              // cluster mean x-correlation lag
  Float_t*        XLAG;              // maximum x-correlation lag in the cluster
  Float_t*        aSF;               // significance for average x-correlation
  Float_t*        mSF;               // significance for maximum x-correlation
  Float_t*        duration;          // cluster duration = stopW-startW

  Double_t        TIME;              // trigger time stamp                         
  Double_t*       time;              // average center time
  Double_t*       itime;             // injection time: 0 - LLO, 1 - LHO                           
  Double_t*       start;             // actual start GPS time
  Double_t*       stop;              // actual stop GPS time 
  Double_t        strain;            // strain of injected simulated signals                         


//List of branches

   TBranch        *b_run;   
   TBranch        *b_nevent;   
   TBranch        *b_ifo;   
   TBranch        *b_eventID;   
   TBranch        *b_type;   
   TBranch        *b_stype;   

   TBranch        *b_size;   
   TBranch        *b_usize;   

   TBranch        *b_gap;   
   TBranch        *b_shift;   
   TBranch        *b_strain;   
   TBranch        *b_TIME;   

   TBranch        *b_xcor;   
   TBranch        *b_XCOR;   
   TBranch        *b_xlag;   
   TBranch        *b_XLAG;   
   TBranch        *b_aSF;   
   TBranch        *b_mSF;   

   TBranch        *b_time;   
   TBranch        *b_itime;   
   TBranch        *b_duration;   
   TBranch        *b_start;   
   TBranch        *b_stop;   

   xctrigger() :
      ifo(NULL),size(NULL),gap(NULL),shift(NULL),xcor(NULL),XCOR(NULL),     
      xlag(NULL),XLAG(NULL),aSF(NULL),mSF(NULL),duration(NULL),
      time(NULL),itime(NULL),start(NULL),stop(NULL) 
   { ndim=1; allocate(); }

   xctrigger(int n) : 
      ifo(NULL),size(NULL),gap(NULL),shift(NULL),xcor(NULL),XCOR(NULL),     
      xlag(NULL),XLAG(NULL),aSF(NULL),mSF(NULL),duration(NULL),
      time(NULL),itime(NULL),start(NULL),stop(NULL) 
   { ndim=n; allocate(); }

   xctrigger(const xctrigger& a) :
      ifo(NULL),size(NULL),gap(NULL),shift(NULL),xcor(NULL),XCOR(NULL),     
      xlag(NULL),XLAG(NULL),aSF(NULL),mSF(NULL),duration(NULL),
      time(NULL),itime(NULL),start(NULL),stop(NULL) 
   { ndim=a.ndim; allocate(); *this = a; return; }

   xctrigger(TTree *tree) {if(tree) Init(tree);}

   virtual xctrigger& operator=(const xctrigger &);

   virtual ~xctrigger() { 
      if(fChain)     delete fChain->GetCurrentFile(); 
      if(ifo)        free(ifo);      
      if(size)       free(size);    
      if(shift)      free(shift);   
      if(gap)        free(gap);   
      if(xcor)       free(xcor);    
      if(XCOR)       free(XCOR);    
      if(xlag)       free(xlag);    
      if(XLAG)       free(XLAG);    
      if(aSF)        free(aSF);     
      if(mSF)        free(mSF);     
      if(time)       free(time);    
      if(itime)      free(itime);   
      if(start)      free(start);   
      if(stop)       free(stop);    
      if(duration)   free(duration);
   }

   Int_t  GetEntry(Int_t);
   void   allocate();
   void   Init(TTree *);
   Bool_t Notify();
   TTree* setTree();

//   void   Loop();
//   Int_t  Cut(Int_t entry);
//   Int_t  LoadTree(Int_t entry);

   void   Show(Int_t entry = -1);

};

#endif




