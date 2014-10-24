#include "macro/xcdouble.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

xcdouble& xcdouble::operator=(xcdouble& a)
{
  fChain=                NULL;             
  fCurrent=              a.fCurrent;                                                        
  nevent=                a.nevent;             
  ifo=                   a.ifo;             
  xcor=                  a.xcor;              
  xlag=                  a.xlag;              
  shift=                 a.shift;              
  time=                  a.time;               
  return *this;
}

//   Set branch addresses
void xcdouble::Init(TTree *tree)
{
   if (tree == 0) return;
   fChain    = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("nevent",&nevent);
   fChain->SetBranchAddress("ifo",&ifo);
   fChain->SetBranchAddress("xcor",&xcor);
   fChain->SetBranchAddress("xlag",&xlag);
   fChain->SetBranchAddress("shift",&shift);
   fChain->SetBranchAddress("time",&time);

   Notify();
}


//++++++++++++++++++++++++++++++++++++++++++++++
// set noise xcdouble tree
//++++++++++++++++++++++++++++++++++++++++++++++
TTree* xcdouble::setTree()
{
   TTree* waveTree = new TTree("xcdouble","xcdouble");

 //==================================
 // Define trigger tree
 //==================================

   waveTree->Branch("nevent",      &nevent,       "nevent/I");
   waveTree->Branch("ifo",         &ifo,          "ifo/I");
   waveTree->Branch("xcor",        &xcor,         "xcor/F");
   waveTree->Branch("xlag",        &xlag,         "xlag/F");
   waveTree->Branch("shift",       &shift,        "shift/F");
   waveTree->Branch("time",        &time,         "time/D");
   
   return waveTree;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++
// dump noise xcdouble into tree
//++++++++++++++++++++++++++++++++++++++++++++++++++++
void xcdouble::output(TTree* waveTree, wavecor& c)
{
   size_t i;
   size_t n = c.xcor.size();
   double start_var = c.xcor.start();
   double rate_var  = c.xcor.rate();

//Fill tree

   for(i=0; i<n; i++){
      xcor=         c.xcor.data[i];
      if(xcor==0.) continue;
      nevent=       i;
      ifo=          c.ifo;
      shift=        c.shift;
      xlag=         c.xlag.data[i];
      time=         start_var+i/rate_var;
      waveTree->Fill();
   }   
}



Bool_t xcdouble::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   b_nevent = fChain->GetBranch("nevent");
   b_ifo = fChain->GetBranch("ifo");
   b_xcor = fChain->GetBranch("xcor");
   b_xlag = fChain->GetBranch("xlag");
   b_shift = fChain->GetBranch("shift");
   b_time = fChain->GetBranch("time");
   return kTRUE;
}

Int_t xcdouble::GetEntry(Int_t entry) 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntry(entry); 
};

void xcdouble::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}

/*
Int_t xcdouble::LoadTree(Int_t entry)
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

Int_t xcdouble::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}

void xcdouble::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L xcdouble.C
//      Root > xcdouble t
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








