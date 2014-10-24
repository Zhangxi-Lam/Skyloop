#include "wbsingle.h"
#include "wbtriple.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"


//   Set branch addresses
void wbtriple::Init(TTree *tree)
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
   fChain->SetBranchAddress("rate",rate);

   fChain->SetBranchAddress("gap",gap);
   fChain->SetBranchAddress("shift",shift);
   fChain->SetBranchAddress("strain",&strain);
   fChain->SetBranchAddress("phi",&phi);
   fChain->SetBranchAddress("teta",&teta);
   fChain->SetBranchAddress("bp",bp);
   fChain->SetBranchAddress("bx",bx);

   fChain->SetBranchAddress("volume",volume);
   fChain->SetBranchAddress("size",size);
   fChain->SetBranchAddress("usize",&usize);

   fChain->SetBranchAddress("power",power);
   fChain->SetBranchAddress("rLH",rLH);
   fChain->SetBranchAddress("gLH",gLH);
   fChain->SetBranchAddress("rSNR",rSNR);
   fChain->SetBranchAddress("gSNR",gSNR);
   fChain->SetBranchAddress("rSF",rSF);
   fChain->SetBranchAddress("gSF",gSF);

   fChain->SetBranchAddress("time",time);
   fChain->SetBranchAddress("itime",itime);
   fChain->SetBranchAddress("right",right);
   fChain->SetBranchAddress("left",left);
   fChain->SetBranchAddress("duration",duration);
   fChain->SetBranchAddress("start",start);
   fChain->SetBranchAddress("stop",stop);

   fChain->SetBranchAddress("frequency",frequency);
   fChain->SetBranchAddress("low",low);
   fChain->SetBranchAddress("high",high);
   fChain->SetBranchAddress("bandwidth",bandwidth);

   fChain->SetBranchAddress("hrss",hrss);
   fChain->SetBranchAddress("noise",noise);
   fChain->SetBranchAddress("asymmetry",asymmetry);

   Notify();
}

Bool_t wbtriple::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   b_run = fChain->GetBranch("run");
   b_nevent = fChain->GetBranch("nevent");
   b_ifo = fChain->GetBranch("ifo");
   b_eventID = fChain->GetBranch("eventID");
   b_type = fChain->GetBranch("type");
   b_stype = fChain->GetBranch("stype");
   b_rate = fChain->GetBranch("rate");

   b_gap = fChain->GetBranch("gap");
   b_shift = fChain->GetBranch("shift");
   b_strain = fChain->GetBranch("strain");
   b_phi = fChain->GetBranch("phi");
   b_teta = fChain->GetBranch("teta");
   b_bp= fChain->GetBranch("bx");
   b_bx = fChain->GetBranch("bp");

   b_volume = fChain->GetBranch("volume");
   b_size = fChain->GetBranch("size");
   b_usize = fChain->GetBranch("usize");

   b_power = fChain->GetBranch("power");
   b_rLH = fChain->GetBranch("rLH");
   b_gLH = fChain->GetBranch("gLH");
   b_rSNR = fChain->GetBranch("rSNR");
   b_gSNR = fChain->GetBranch("gSNR");
   b_rSF = fChain->GetBranch("rSF");
   b_gSF = fChain->GetBranch("gSF");

   b_time = fChain->GetBranch("time");
   b_itime = fChain->GetBranch("itime");
   b_right = fChain->GetBranch("right");
   b_left = fChain->GetBranch("left");
   b_start = fChain->GetBranch("start");
   b_stop = fChain->GetBranch("stop");
   b_duration = fChain->GetBranch("duration");

   b_frequency = fChain->GetBranch("frequency");
   b_low = fChain->GetBranch("low");
   b_high = fChain->GetBranch("high");
   b_bandwidth = fChain->GetBranch("bandwidth");

   b_hrss = fChain->GetBranch("hrss");
   b_noise = fChain->GetBranch("noise");
   b_asymmetry = fChain->GetBranch("asymmetry");

   return kTRUE;
}

Int_t wbtriple::GetEntry(Int_t entry) 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntry(entry); 
};

void wbtriple::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}


//++++++++++++++++++++++++++++++++++++++++++++++
// set single event tree
//++++++++++++++++++++++++++++++++++++++++++++++
TTree* wbtriple::setTree()
{
   TTree* waveTree = new TTree("waveburst","waveburst");

 //==================================
 // Define trigger tree
 //==================================

   waveTree->Branch("run",         &run,         "run/I");
   waveTree->Branch("nevent",      &nevent,      "nevent/I");
   waveTree->Branch("eventID",     &eventID,     "eventID/I");
   waveTree->Branch("ifo",         ifo,          "ifo[3]/I");
   waveTree->Branch("type",        &type,        "type/I");
   waveTree->Branch("stype",       &stype,       "stype/I");
   waveTree->Branch("rate",        rate,         "rate[3]/I");
   
   waveTree->Branch("gap",         gap,          "gap[3]/F");
   waveTree->Branch("shift",       shift,        "shift[3]/F");
   waveTree->Branch("strain",      &strain,      "strain/D");
   waveTree->Branch("phi",         &phi,          "phi/F");
   waveTree->Branch("teta",        &teta,         "teta/F");
   waveTree->Branch("bp",          bp,            "bp[3]/F");
   waveTree->Branch("bx",          bx,            "bx[3]/F");
   
   waveTree->Branch("usize",       &usize,       "usize/I");
   waveTree->Branch("volume",      volume,       "volume[3]/I");
   waveTree->Branch("size",        size,         "size[3]/I");
   
   waveTree->Branch("power",       power,        "power[3]/F");
   waveTree->Branch("rLH",         rLH,          "rLH[3]/F");
   waveTree->Branch("gLH",         gLH,          "gLH[3]/F");
   waveTree->Branch("rSNR",        rSNR,         "rSNR[3]/F");
   waveTree->Branch("gSNR",        gSNR,         "gSNR[3]/F");
   waveTree->Branch("rSF",         rSF,          "rSF[3]/F");
   waveTree->Branch("gSF",         gSF,          "gSF[3]/F");
   
   waveTree->Branch("time",        time,         "time[3]/D");
   waveTree->Branch("itime",       itime,        "itime[3]/D");
   waveTree->Branch("left",        left,         "left[3]/F");
   waveTree->Branch("right",       right,        "right[3]/F");
   waveTree->Branch("start",       start,        "start[3]/D");
   waveTree->Branch("stop",        stop,         "stop[3]/D");
   waveTree->Branch("duration",    duration,     "duration[3]/F");
   
   waveTree->Branch("frequency",   frequency,    "frequency[3]/F");
   waveTree->Branch("low",         low,          "low[3]/F" );
   waveTree->Branch("high",        high,         "high[3]/F");
   waveTree->Branch("bandwidth",   bandwidth,    "bandwidth[3]/F");
   
   waveTree->Branch("hrss",        hrss,         "hrss[3]/D");
   waveTree->Branch("noise",       noise,        "noise[3]/D");
   waveTree->Branch("asymmetry",asymmetry, "asymmetry[3]/F");
   
   return waveTree;
}

wbtriple& wbtriple::operator=(wbtriple& a)
{
   int i;

   run=          a.run;
   nevent=       a.nevent;
   eventID=      a.eventID;
   type=         a.type;
   stype=        a.stype;
   strain=       a.strain;
   usize=        a.usize;
   phi=          a.phi;             
   teta=         a.teta;             

   for(i=0; i<3; i++){
      
      ifo[i]=          a.ifo[i];
      rate[i]=         a.rate[i];
      
      gap[i]=          a.gap[i];
      shift[i]=        a.shift[i];
      volume[i]=       a.volume[i];
      size[i]=         a.size[i];
      bp[i]=           a.bp[i];             
      bx[i]=           a.bx[i];             
      
      power[i]=        a.power[i];
      rLH[i]=          a.rLH[i];
      gLH[i]=          a.gLH[i];
      rSNR[i]=         a.rSNR[i];
      gSNR[i]=         a.gSNR[i];
      rSF[i]=          a.rSF[i];
      gSF[i]=          a.gSF[i];
      
      time[i]=         a.time[i];
      itime[i]=        a.itime[i];
      right[i]=        a.right[i];
      left[i]=         a.left[i];
      duration[i]=     a.duration[i];
      start[i]=        a.start[i];
      stop[i]=         a.stop[i];
      
      frequency[i]=    a.frequency[i];
      low[i]=          a.low[i];
      high[i]=         a.high[i];
      bandwidth[i]=    a.bandwidth[i];
      
      hrss[i]=         a.hrss[i];
      noise[i]=        a.noise[i];
      asymmetry[i]= a.asymmetry[i];
   }
   return *this;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++
// output wavecluster into tree
//++++++++++++++++++++++++++++++++++++++++++++++++++++
void wbtriple::setfrom(wbsingle** p)
{
   int i;

   run=          p[0]->run;
   nevent=       p[0]->nevent;
   eventID=      0;
   type=         p[0]->type;
   stype=        p[0]->stype;
   strain=       p[0]->strain;
   usize=        0;
   phi=          p[0]->phi;             
   teta=         p[0]->teta;             

   for(i=0; i<3; i++){
      
      ifo[i]=          p[i]->ifo;
      rate[i]=         p[i]->rate;
      
      gap[i]=          p[i]->gap;
      shift[i]=        p[i]->shift;
      volume[i]=       p[i]->volume;
      size[i]=         p[i]->size;
      bp[i]=           p[i]->bp;             
      bx[i]=           p[i]->bx;             
      
      power[i]=        p[i]->power;
      rLH[i]=          p[i]->rLH;
      gLH[i]=          p[i]->gLH;
      rSNR[i]=         p[i]->rSNR;
      gSNR[i]=         p[i]->gSNR;
      rSF[i]=          p[i]->rSF;
      gSF[i]=          p[i]->gSF;
      
      time[i]=         p[i]->time;
      itime[i]=        p[i]->itime;
      right[i]=        p[i]->right;
      left[i]=         p[i]->left;
      duration[i]=     p[i]->duration;
      start[i]=        p[i]->start;
      stop[i]=         p[i]->stop;
      
      frequency[i]=    p[i]->frequency;
      low[i]=          p[i]->low;
      high[i]=         p[i]->high;
      bandwidth[i]=    p[i]->bandwidth;
      
      hrss[i]=         p[i]->hrss;
      noise[i]=        p[i]->noise;
      asymmetry[i]=    p[i]->asymmetry;

   }
   
}


/*
Int_t wbtriple::LoadTree(Int_t entry)
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

Int_t wbtriple::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}

void wbtriple::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L wbtriple.C
//      Root > wbtriple t
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








