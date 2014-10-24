
#include "xctriple.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"


//   Set branch addresses
void xctriple::Init(TTree *tree)
{
   if (tree == 0) return;
   fChain    = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

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

Bool_t xctriple::Notify()
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

Int_t xctriple::GetEntry(Int_t entry) 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntry(entry); 
};

void xctriple::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}


//++++++++++++++++++++++++++++++++++++++++++++++
// set triple event tree
//++++++++++++++++++++++++++++++++++++++++++++++
TTree* xctriple::setTree()
{
   TTree* waveTree = new TTree("xctriple","xctriple");

 //==================================
 // Define trigger tree
 //==================================

   waveTree->Branch("run",         &run,         "run/I");
   waveTree->Branch("nevent",      &nevent,      "nevent/I");
   waveTree->Branch("eventID",     &eventID,     "eventID/I");
   waveTree->Branch("ifo",         ifo,          "ifo[3]/I");
   waveTree->Branch("type",        &type,        "type/I");
   waveTree->Branch("stype",       &stype,       "stype/I");
   
   waveTree->Branch("gap",         gap,          "gap[3]/F");
   waveTree->Branch("shift",       shift,        "shift[3]/F");
   waveTree->Branch("strain",      &strain,      "strain/D");
   
   waveTree->Branch("usize",       &usize,       "usize/I");
   waveTree->Branch("size",        size,         "size[3]/I");
   
   waveTree->Branch("xcor",        xcor,         "xcor[3]/F");
   waveTree->Branch("XCOR",        XCOR,         "XCOR[3]/F");
   waveTree->Branch("xlag",        xlag,         "xlag[3]/F");
   waveTree->Branch("XLAG",        XLAG,         "XLAG[3]/F");
   waveTree->Branch("aSF",         aSF,          "aSF[3]/F");
   waveTree->Branch("mSF",         mSF,          "mSF[3]/F");
   
   waveTree->Branch("time",        time,         "time[3]/D");
   waveTree->Branch("itime",       itime,        "itime[3]/D");
   waveTree->Branch("start",       start,        "start[3]/D");
   waveTree->Branch("stop",        stop,         "stop[3]/D");
   waveTree->Branch("duration",    duration,     "duration[3]/F");
   
   return waveTree;
}


xctriple& xctriple::operator=(xctriple& a)
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

   for(i=0; i<3; i++){
      
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
Int_t xctriple::LoadTree(Int_t entry)
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

Int_t xctriple::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}

void xctriple::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L xctriple.C
//      Root > xctriple t
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








