#include "xctrigger.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"


//   Set branch addresses
void xctrigger::Init(TTree *tree)
{
   if (tree == 0) return;
   fChain    = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   allocate();              // allocate pointers

   fChain->SetBranchAddress("run",&run);
   fChain->SetBranchAddress("nevent",&nevent);
   fChain->SetBranchAddress("ifo",ifo);
   fChain->SetBranchAddress("eventID",&eventID);
   fChain->SetBranchAddress("type",&type);
   fChain->SetBranchAddress("stype",&stype);

   fChain->SetBranchAddress("gap",gap);
   fChain->SetBranchAddress("shift",shift);
   fChain->SetBranchAddress("strain",&strain);

   fChain->SetBranchAddress("size",size);
   fChain->SetBranchAddress("usize",&usize);

   fChain->SetBranchAddress("xcor",xcor);
   fChain->SetBranchAddress("XCOR",XCOR);
   fChain->SetBranchAddress("xlag",xlag);
   fChain->SetBranchAddress("XLAG",XLAG);
   fChain->SetBranchAddress("aSF",aSF);
   fChain->SetBranchAddress("mSF",mSF);

   fChain->SetBranchAddress("time",time);
   fChain->SetBranchAddress("itime",itime);
   fChain->SetBranchAddress("duration",duration);
   fChain->SetBranchAddress("start",start);
   fChain->SetBranchAddress("stop",stop);

   Notify();
}

// allocate memory
void xctrigger::allocate()
{
   if(!time)     time=      (Double_t*)malloc(ndim*sizeof(Double_t)); 
   else          time=      (Double_t*)realloc(time,ndim*sizeof(Double_t)); 
   if(!itime)    itime=     (Double_t*)malloc(ndim*sizeof(Double_t)); 
   else          itime=     (Double_t*)realloc(itime,ndim*sizeof(Double_t)); 
   if(!start)    start=     (Double_t*)malloc(ndim*sizeof(Double_t)); 
   else          start=     (Double_t*)realloc(start,ndim*sizeof(Double_t)); 
   if(!stop)     stop=      (Double_t*)malloc(ndim*sizeof(Double_t)); 
   else          stop=      (Double_t*)realloc(stop,ndim*sizeof(Double_t)); 
   		 			   
   if(!ifo)      ifo=       (Int_t*)malloc(ndim*sizeof(Int_t)); 
   else          ifo=       (Int_t*)realloc(ifo,ndim*sizeof(Int_t)); 
   if(!size)     size=      (Int_t*)malloc(ndim*sizeof(Int_t)); 
   else          size=      (Int_t*)realloc(size,ndim*sizeof(Int_t)); 
    		 			   
   if(!shift)    shift=     (Float_t*)malloc(ndim*sizeof(Float_t)); 
   else          shift=     (Float_t*)realloc(shift,ndim*sizeof(Float_t)); 
   if(!xcor)     xcor=      (Float_t*)malloc(ndim*sizeof(Float_t)); 
   else          xcor=      (Float_t*)realloc(xcor,ndim*sizeof(Float_t)); 
   if(!XCOR)     XCOR=      (Float_t*)malloc(ndim*sizeof(Float_t)); 
   else          XCOR=      (Float_t*)realloc(XCOR,ndim*sizeof(Float_t)); 
   if(!xlag)     xlag=      (Float_t*)malloc(ndim*sizeof(Float_t)); 
   else          xlag=      (Float_t*)realloc(xlag,ndim*sizeof(Float_t)); 
   if(!XLAG)     XLAG=      (Float_t*)malloc(ndim*sizeof(Float_t)); 
   else          XLAG=      (Float_t*)realloc(XLAG,ndim*sizeof(Float_t)); 
   if(!aSF)      aSF=       (Float_t*)malloc(ndim*sizeof(Float_t)); 
   else          aSF=       (Float_t*)realloc(aSF,ndim*sizeof(Float_t)); 
   if(!mSF)      mSF=       (Float_t*)malloc(ndim*sizeof(Float_t)); 
   else          mSF=       (Float_t*)realloc(mSF,ndim*sizeof(Float_t)); 
   if(!gap)      gap=       (Float_t*)malloc(ndim*sizeof(Float_t)); 
   else          gap=       (Float_t*)realloc(gap,ndim*sizeof(Float_t)); 
   if(!duration) duration=  (Float_t*)malloc(ndim*sizeof(Float_t)); 
   else          duration=  (Float_t*)realloc(duration,ndim*sizeof(Float_t)); 

   return;
}

Bool_t xctrigger::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   b_run = fChain->GetBranch("run");
   b_nevent = fChain->GetBranch("nevent");
   b_ifo = fChain->GetBranch("ifo");
   b_eventID = fChain->GetBranch("eventID");
   b_type = fChain->GetBranch("type");
   b_stype = fChain->GetBranch("stype");

   b_gap = fChain->GetBranch("gap");
   b_shift = fChain->GetBranch("shift");
   b_strain = fChain->GetBranch("strain");
   b_TIME = fChain->GetBranch("TIME");

   b_size = fChain->GetBranch("size");
   b_usize = fChain->GetBranch("usize");

   b_xcor = fChain->GetBranch("xcor");
   b_XCOR = fChain->GetBranch("XCOR");
   b_xlag = fChain->GetBranch("xlag");
   b_XLAG = fChain->GetBranch("XLAG");
   b_aSF = fChain->GetBranch("aSF");
   b_mSF = fChain->GetBranch("mSF");

   b_time = fChain->GetBranch("time");
   b_itime = fChain->GetBranch("itime");
   b_start = fChain->GetBranch("start");
   b_stop = fChain->GetBranch("stop");
   b_duration = fChain->GetBranch("duration");

   return kTRUE;
}

Int_t xctrigger::GetEntry(Int_t entry) 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntry(entry); 
};

void xctrigger::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}


//++++++++++++++++++++++++++++++++++++++++++++++
// set triple event tree
//++++++++++++++++++++++++++++++++++++++++++++++
TTree* xctrigger::setTree()
{
   TTree* waveTree = new TTree("xctrigger","xctrigger");

 //==================================
 // Define trigger tree
 //==================================

   char ctime[16];    
   char citime[16];   
   char cstart[16];   
   char cstop[16];    
   
   char cifo[16];     	   
   char csize[16];    
   
   char cshift[16];   	   
   char cxcor[16];    
   char cXCOR[16];    
   char cxlag[16];    
   char cXLAG[16];    
   char caSF[16];     
   char cmSF[16];     
   char cgap[16];     
   char cduration[16];
   
   sprintf(ctime,"time[%1d]/D",ndim);    
   sprintf(citime,"itime[%1d]/D",ndim);   
   sprintf(cstart,"start[%1d]/D",ndim);   
   sprintf(cstop,"stop[%1d]/D",ndim);    
   
   sprintf(cifo,"ifo[%1d]/I",ndim);     
   sprintf(csize,"size[%1d]/I",ndim);    
   
   sprintf(cshift,"shift[%1d]/F",ndim);   
   sprintf(cxcor,"xcor[%1d]/F",ndim);    
   sprintf(cXCOR,"XCOR[%1d]/F",ndim);    
   sprintf(cxlag,"xlag[%1d]/F",ndim);    
   sprintf(cXLAG,"XLAG[%1d]/F",ndim);    
   sprintf(caSF,"aSF[%1d]/F",ndim);     
   sprintf(cmSF,"mSF[%1d]/F",ndim);     
   sprintf(cgap,"gap[%1d]/F",ndim);     
   sprintf(cduration,"duration[%1d]/F",ndim);
     
   waveTree->Branch("run",         &run,         "run/I");
   waveTree->Branch("nevent",      &nevent,      "nevent/I");
   waveTree->Branch("eventID",     &eventID,     "eventID/I");
   waveTree->Branch("type",        &type,        "type/I");
   waveTree->Branch("stype",       &stype,       "stype/I");
   waveTree->Branch("usize",       &usize,       "usize/I");
   waveTree->Branch("strain",      &strain,      "strain/D");
   waveTree->Branch("TIME",        &TIME,        "TIME/D");
   
   waveTree->Branch("ifo",         ifo,      cifo);     
   waveTree->Branch("size",        size,     csize);    
				   	     	    	     
   waveTree->Branch("gap",         gap,      cgap);     
   waveTree->Branch("shift",       shift,    cshift);   
   waveTree->Branch("xcor",        xcor,     cxcor);    
   waveTree->Branch("XCOR",        XCOR,     cXCOR);    
   waveTree->Branch("xlag",        xlag,     cxlag);    
   waveTree->Branch("XLAG",        XLAG,     cXLAG);    
   waveTree->Branch("aSF",         aSF,      caSF);     
   waveTree->Branch("mSF",         mSF,      cmSF);     
   waveTree->Branch("duration",    duration, cduration);
   				   	     	    	     
   waveTree->Branch("time",        time,     ctime);    
   waveTree->Branch("itime",       itime,    citime);   
   waveTree->Branch("start",       start,    cstart);   
   waveTree->Branch("stop",        stop,     cstop);    

   return waveTree;
}


xctrigger& xctrigger::operator=(const xctrigger& a)
{
   int i;

   fChain=       NULL;             
   fCurrent=     a.fCurrent;                                                        
   run=          a.run;
   nevent=       a.nevent;
   eventID=      a.eventID;
   type=         a.type;
   stype=        a.stype;
   strain=       a.strain;
   usize=        a.usize;
   TIME=         a.TIME;

   if(ndim!=a.ndim) { ndim=a.ndim; allocate(); }

   for(i=0; i<ndim; i++){
      
      ifo[i]=          a.ifo[i];
      
      gap[i]=          a.gap[i];
      shift[i]=        a.shift[i];
      size[i]=         a.size[i];
      
      xcor[i]=         a.xcor[i];
      XCOR[i]=         a.XCOR[i];
      xlag[i]=         a.xlag[i];
      XLAG[i]=         a.XLAG[i];
      aSF[i]=          a.aSF[i];
      mSF[i]=          a.mSF[i];
      
      time[i]=         a.time[i];
      itime[i]=        a.itime[i];
      duration[i]=     a.duration[i];
      start[i]=        a.start[i];
      stop[i]=         a.stop[i];
      
   }
   return *this;
}

/*
Int_t xctrigger::LoadTree(Int_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Int_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->IsA() != TChain::Class()) return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
      Notify();
   }
   return centry;
}

Int_t xctrigger::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}

void xctrigger::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L xctrigger.C
//      Root > xctrigger t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   Int_t nentries = Int_t(fChain->GetEntriesFast());

   Int_t nbytes = 0, nb = 0;
   for (Int_t jentry=0; jentry<nentries;jentry++) {
      Int_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
   }
}
*/








