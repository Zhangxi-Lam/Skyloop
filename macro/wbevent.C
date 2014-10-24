#include "wbsingle.h"
#include "wbevent.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"


//   Set branch addresses
void wbevent::Init(TTree *tree)
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

// allocate memory
void wbevent::allocate()
{
   if (!ifo)         ifo=      (Int_t*)malloc(ndim*sizeof(Int_t));
   else              ifo=      (Int_t*)realloc(ifo,ndim*sizeof(Int_t));
   if (!rate)        rate=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              rate=     (Float_t*)realloc(rate,ndim*sizeof(Float_t));
   
   if (!volume)      volume=   (Int_t*)malloc(ndim*sizeof(Int_t));
   else              volume=   (Int_t*)realloc(volume,ndim*sizeof(Int_t));
   if (!size)        size=     (Int_t*)malloc(ndim*sizeof(Int_t));
   else              size=     (Int_t*)realloc(size,ndim*sizeof(Int_t));
   
   if (!gap)         gap=      (Float_t*)malloc(ndim*sizeof(Float_t));
   else              gap=      (Float_t*)realloc(gap,ndim*sizeof(Float_t));
   if (!shift)       shift=    (Float_t*)malloc(ndim*sizeof(Float_t));
   else              shift=    (Float_t*)realloc(shift,ndim*sizeof(Float_t));
   if (!bp)          bp=       (Float_t*)malloc(ndim*sizeof(Float_t));
   else              bp=       (Float_t*)realloc(bp,ndim*sizeof(Float_t));
   if (!bx)          bx=       (Float_t*)malloc(ndim*sizeof(Float_t));
   else              bx=       (Float_t*)realloc(bx,ndim*sizeof(Float_t));
   
   if (!power)       power=    (Float_t*)malloc(ndim*sizeof(Float_t));
   else              power=    (Float_t*)realloc(powerndim*sizeof(Float_t));
   if (!rSNR)        rSNR=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              rSNR=     (Float_t*)realloc(rSNR,ndim*sizeof(Float_t));
   if (!gSNR)        gSNR=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              gSNR=     (Float_t*)realloc(gSNR,ndim*sizeof(Float_t));
   if (!rLH)         rLH=      (Float_t*)malloc(ndim*sizeof(Float_t));
   else              rLH=      (Float_t*)realloc(rLHndim*sizeof(Float_t));
   if (!gLH)         gLH=      (Float_t*)malloc(ndim*sizeof(Float_t));
   else              gLH=      (Float_t*)realloc(gLH,ndim*sizeof(Float_t));
   if (!rSF)         rSF=      (Float_t*)malloc(ndim*sizeof(Float_t));
   else              rSF=      (Float_t*)realloc(rSF,ndim*sizeof(Float_t));
   if (!gSF)         gSF=      (Float_t*)malloc(ndim*sizeof(Float_t));
   else              gSF=      (Float_t*)realloc(gSF,ndim*sizeof(Float_t));
   
   if (!time)        time=     (Double_t*)malloc(ndim*sizeof(Double_t));
   else              time=     (Double_t*)realloc(time,ndim*sizeof(Double_t));
   if (!itime)       itime=    (Double_t*)malloc(ndim*sizeof(Double_t));
   else              itime=    (Double_t*)realloc(itime,ndim*sizeof(Double_t));
   if (!right)       right=    (Float_t*)malloc(ndim*sizeof(Float_t));
   else              right=    (Float_t*)realloc(right,ndim*sizeof(Float_t));
   if (!left)        left=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              left=     (Float_t*)realloc(left,ndim*sizeof(Float_t));
   if (!duration)    duration= (Float_t*)malloc(ndim*sizeof(Float_t));
   else              duration= (Float_t*)realloc(duration,ndim*sizeof(Float_t));
   if (!start)       start=    (Double_t*)malloc(ndim*sizeof(Double_t));
   else              start=    (Double_t*)realloc(start,ndim*sizeof(Double_t));
   if (!stop)        stop=     (Double_t*)malloc(ndim*sizeof(Double_t));
   else              stop=     (Double_t*)realloc(stop,ndim*sizeof(Double_t));
   
   if (!frequency)   frequency=(Float_t*)malloc(ndim*sizeof(Float_t));
   else              frequency=(Float_t*)realloc(frequency,ndim*sizeof(Float_t));
   if (!low)         low=      (Float_t*)malloc(ndim*sizeof(Float_t));
   else              low=      (Float_t*)realloc(low,ndim*sizeof(Float_t));
   if (!high)        high=     (Float_t*)malloc(ndim*sizeof(Float_t));
   else              high=     (Float_t*)realloc(high,ndim*sizeof(Float_t));
   if (!bandwidth)   bandwidth=(Float_t*)malloc(ndim*sizeof(Float_t));
   else              bandwidth=(Float_t*)realloc(bandwidth,ndim*sizeof(Float_t));
   if (!hrss)        hrss=     (Double_t*)malloc(ndim*sizeof(Double_t));
   else              hrss=     (Double_t*)realloc(hrss,ndim*sizeof(Double_t));
   if (!noise)       noise=    (Double_t*)malloc(ndim*sizeof(Double_t));
   else              noise=    (Double_t*)realloc(noise,ndim*sizeof(Double_t));
   if (!asymmetry)   asymmetry=(Float_t*)malloc(ndim*sizeof(Float_t));
   else              asymmetry=(Float_t*)realloc(asymmetry,ndim*sizeof(Float_t));
   

   return;
}


Bool_t wbevent::Notify()
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

Int_t wbevent::GetEntry(Int_t entry) 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntry(entry); 
};

void wbevent::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}


//++++++++++++++++++++++++++++++++++++++++++++++
// set single event tree
//++++++++++++++++++++++++++++++++++++++++++++++
TTree* wbevent::setTree()
{
   TTree* waveTree = new TTree("waveburst","waveburst");

char cifo[16];      
char crate[16];     
	    
char cvolume[16];   
char csize[16];     
	    
char cgap[16];      
char cshift[16];    
char cphi[16];      
char cteta[16];     
char cbp[16];       
char cbx[16];       
	    
char cpower[16];    
char crSNR[16];     
char cgSNR[16];     
char crLH[16];      
char cgLH[16];      
char crSF[16];      
char cgSF[16];      
	    
char ctime[16];     
char citime[16];    
char cright[16];    
char cleft[16];     
char cduration[16]; 
char cstart[16];    
char cstop[16];     
	    
char cfrequency[16];
char clow[16];      
char chigh[16];     
char cbandwidth[16];
char chrss[16];     
char cnoise[16];    
char casymmetry[16];


 sprintf(cifo,       "ifo[%1d]/I",ndim);      
 sprintf(crate,      "rate[%1d]/F",ndim);     
 
 sprintf(cvolume,    "volume[%1d]/F",ndim);   
 sprintf(csize,      "size[%1d]/F",ndim);     
 
 sprintf(cgap,       "gap[%1d]/F",ndim);      
 sprintf(cshift,     "shift[%1d]/F",ndim);    
 sprintf(cphi,       "phi[%1d]/F",ndim);      
 sprintf(cteta,      "teta[%1d]/F",ndim);     
 sprintf(cbp,        "bp[%1d]/F",ndim);       
 sprintf(cbx,        "bx[%1d]/F",ndim);       
 
 sprintf(cpower,     "power[%1d]/F",ndim);    
 sprintf(crSNR,      "rSNR[%1d]/F",ndim);     
 sprintf(cgSNR,      "gSNR[%1d]/F",ndim);     
 sprintf(crLH,       "rLH[%1d]/F",ndim);      
 sprintf(cgLH,       "gLH[%1d]/F",ndim);      
 sprintf(crSF,       "rSF[%1d]/F",ndim);      
 sprintf(cgSF,       "gSF[%1d]/F",ndim);      
 
 sprintf(ctime,      "time[%1d]/D",ndim);     
 sprintf(citime,     "itime[%1d]/D",ndim);    
 sprintf(cright,     "right[%1d]/F",ndim);    
 sprintf(cleft,      "left[%1d]/F",ndim);     
 sprintf(cduration,  "duration[%1d]/F",ndim); 
 sprintf(cstart,     "start[%1d]/D",ndim);    
 sprintf(cstop,      "stop[%1d]/D",ndim);     
	    	   	    	     
 sprintf(cfrequency, "frequency[%1d]/F",ndim);
 sprintf(clow,       "low[%1d]/F",ndim);      
 sprintf(chigh,      "high[%1d]/F",ndim);     
 sprintf(cbandwidth, "bandwidth[%1d]/F",ndim);
 sprintf(chrss,      "hrss[%1d]/D",ndim);     
 sprintf(cnoise,     "noise[%1d]/D",ndim);    
 sprintf(casymmetry, "asymmetry[%1d]/F",ndim); 
 

 //==================================
 // Define trigger tree
 //==================================

   waveTree->Branch("run",         &run,         "run/I");
   waveTree->Branch("nevent",      &nevent,      "nevent/I");
   waveTree->Branch("eventID",     &eventID,     "eventID/I");
   waveTree->Branch("ifo",         ifo,           cifo);
   waveTree->Branch("type",        &type,        "type/I");
   waveTree->Branch("stype",       &stype,       "stype/I");
   waveTree->Branch("rate",        rate,          crate);
   
   waveTree->Branch("gap",         gap,           cgap);
   waveTree->Branch("shift",       shift,         cshift);
   waveTree->Branch("strain",      &strain,      "strain/D");
   waveTree->Branch("phi",         &phi,         "phi/F");
   waveTree->Branch("teta",        &teta,        "teta/F");
   waveTree->Branch("bp",          bp,            cbp);
   waveTree->Branch("bx",          bx,            cbx);
   
   waveTree->Branch("usize",       &usize,       "usize/I");
   waveTree->Branch("volume",      volume,        cvolume);
   waveTree->Branch("size",        size,          csize);
   
   waveTree->Branch("power",       power,         cpower);
   waveTree->Branch("rLH",         rLH,           crLH);
   waveTree->Branch("gLH",         gLH,           cgLH);
   waveTree->Branch("rSNR",        rSNR,          crSNR);
   waveTree->Branch("gSNR",        gSNR,          cgSNR);
   waveTree->Branch("rSF",         rSF,           crSF);
   waveTree->Branch("gSF",         gSF,           cgSF);
 
   waveTree->Branch("time",        time,          ctime");
   waveTree->Branch("itime",       itime,         citime);
   waveTree->Branch("left",        left,          cleft);
   waveTree->Branch("right",       right,         ccright);
   waveTree->Branch("start",       start,         cstart);
   waveTree->Branch("stop",        stop,          cstop);
   waveTree->Branch("duration",    duration,      cduration);
   
   waveTree->Branch("frequency",   frequency,     cfrequency);
   waveTree->Branch("low",         low,           clow);
   waveTree->Branch("high",        high,          chigh);
   waveTree->Branch("bandwidth",   bandwidth,     cbandwidth);
   
   waveTree->Branch("hrss",        hrss,          chrss);
   waveTree->Branch("noise",       noise,         cnoise);
   waveTree->Branch("asymmetry",   asymmetry,     casymmetry);
   
   return waveTree;
}

wbevent& wbevent::operator=(wbevent& a)
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

   for(i=0; i<ndim; i++){
      
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
      asymmetry[i]=    a.asymmetry[i];
   }
   return *this;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++
// output wavecluster into tree
//++++++++++++++++++++++++++++++++++++++++++++++++++++
void wbevent::setfrom(int n, wbsingle** p)
{
   int i;

   if(ndim != n) {
      cout<<"wbevent::error - wrong number of detectors! "<<ndim<<" != "<<n<<endl; 
      exit 0;
   }

   run=          p[0]->run;
   nevent=       p[0]->nevent;
   eventID=      0;
   type=         p[0]->type;
   stype=        p[0]->stype;
   strain=       p[0]->strain;
   usize=        0;
   phi=          p[0]->phi;             
   teta=         p[0]->teta;             

   for(i=0; i<ndim; i++){
      
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
Int_t wbevent::LoadTree(Int_t entry)
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

Int_t wbevent::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}

void wbevent::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L wbevent.C
//      Root > wbevent t
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








