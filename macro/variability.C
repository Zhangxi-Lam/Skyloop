
#include "variability.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

variability& variability::operator<<(variability& a)
{
  fChain=                NULL;             
  fCurrent=              a.fCurrent;                                                        
  nevent=                a.nevent;             
  ifo=                   a.ifo;             
  value=                 a.value;              
  time=                  a.time;               
  gps=                   a.gps;               
  return *this;
}

//   Set branch addresses
void variability::Init(TTree *tree)
{
   if (tree == 0) return;
   fChain    = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("nevent",&nevent);
   fChain->SetBranchAddress("ifo",&ifo);
   fChain->SetBranchAddress("value",&value);
   fChain->SetBranchAddress("time",&time);
   fChain->SetBranchAddress("gps",&gps);

   Notify();
}


//++++++++++++++++++++++++++++++++++++++++++++++
// set noise variability tree
//++++++++++++++++++++++++++++++++++++++++++++++
TTree* variability::setTree()
{
   TTree* waveTree = new TTree("variability","variability");

 //==================================
 // Define trigger tree
 //==================================

   waveTree->Branch("nevent",      &nevent,       "nevent/I");
   waveTree->Branch("ifo",         &ifo,          "ifo/I");
   waveTree->Branch("value",       &value,        "value/F");
   waveTree->Branch("time",        &time,         "time/D");
   waveTree->Branch("gps",         &gps,          "gps/D");
   
   return waveTree;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++
// dump noise variability into tree
//++++++++++++++++++++++++++++++++++++++++++++++++++++
void variability::output(TTree* waveTree, wavearray<float>* p, int ifo_var, double t_var)
{
   size_t i;
   size_t n = p->size();
   double rate_var  = p->rate();
   size_t m = size_t(t_var*rate_var+0.5); // time offset

   this->gps = p->start();

   if(m>n || !n) return;

//Fill tree

   for(i=m; i<n-m; i++){
      this->nevent=       i;
      this->ifo=          ifo_var;
      this->value=        p->data[i];
      this->time=         this->gps + i/rate_var;
      waveTree->Fill();
   }   
}



Bool_t variability::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   b_nevent = fChain->GetBranch("nevent");
   b_ifo = fChain->GetBranch("ifo");
   b_value = fChain->GetBranch("value");
   b_time = fChain->GetBranch("time");
   return kTRUE;
}

Int_t variability::GetEntry(Int_t entry) 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntry(entry); 
};

void variability::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}

/*
Int_t variability::LoadTree(Int_t entry)
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

Int_t variability::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}

void variability::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L variability.C
//      Root > variability t
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








