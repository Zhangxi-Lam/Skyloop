
#include "wavenoise.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

wavenoise& wavenoise::operator<<(wavenoise& a)
{
  fChain=                NULL;             
  fCurrent=              a.fCurrent;                                                        
  layer=                 a.layer;             
  ifo=                   a.ifo;             
  rms=                   a.rms;              
  frequency=             a.frequency;               
  time=                  a.time;               
  gps=                   a.gps;               
  return *this;
}

//   Set branch addresses
void wavenoise::Init(TTree *tree)
{
   if (tree == 0) return;
   fChain    = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("layer",&layer);
   fChain->SetBranchAddress("ifo",&ifo);
   fChain->SetBranchAddress("rms",&rms);
   fChain->SetBranchAddress("frequency",&frequency);
   fChain->SetBranchAddress("time",&time);
   fChain->SetBranchAddress("gps",&gps);

   Notify();
}


//++++++++++++++++++++++++++++++++++++++++++++++
// set noise wavenoise tree
//++++++++++++++++++++++++++++++++++++++++++++++
TTree* wavenoise::setTree()
{
   TTree* waveTree = new TTree("noise","noise");

 //==================================
 // Define trigger tree
 //==================================

   waveTree->Branch("layer",       &layer,        "layer/I");
   waveTree->Branch("ifo",         &ifo,          "ifo/I");
   waveTree->Branch("rms",         &rms,          "rms/D");
   waveTree->Branch("frequency",   &frequency,    "frequency/F");
   waveTree->Branch("time",        &time,         "time/D");
   waveTree->Branch("gps",         &gps,          "gps/D");
   
   return waveTree;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++
// dump noise wavenoise into tree
//++++++++++++++++++++++++++++++++++++++++++++++++++++
void wavenoise::output(TTree* waveTree, WSeries<double>* p, int ifo_rms, double band_rms)
{
   size_t i,j;
   size_t n = p->maxLayer()+1;          // number of layers
   double rate_rms  = p->wrate();
   wavearray<double> a;

//Fill tree

   this->gps = p->start();
	 
   for(i=0; i<n; i++){
      p->getLayer(a,i);
      for(j=0; j<a.size(); j++){	 
	 this->layer=       i;
	 this->ifo=         ifo_rms;
	 this->rms=         a.data[j];
	 this->time=        this->gps+j/rate_rms;
	 this->frequency=   (i+0.5)*band_rms/n;
	 waveTree->Fill();
      }
   }   
}



Bool_t wavenoise::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   b_layer = fChain->GetBranch("layer");
   b_ifo = fChain->GetBranch("ifo");
   b_rms = fChain->GetBranch("rms");
   b_frequency = fChain->GetBranch("frequency");
   b_time = fChain->GetBranch("time");
   b_gps = fChain->GetBranch("gps");
   return kTRUE;
}

Int_t wavenoise::GetEntry(Int_t entry) 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntry(entry); 
};

void wavenoise::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}

/*
Int_t wavenoise::LoadTree(Int_t entry)
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

Int_t wavenoise::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}

void wavenoise::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L wavenoise.C
//      Root > wavenoise t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show rmss of entry 12
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








