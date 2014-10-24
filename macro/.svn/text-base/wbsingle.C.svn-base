
#include "wbsingle.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

wbsingle& wbsingle::operator=(wbsingle& a)
{
  fChain=                NULL;             
  fCurrent=              a.fCurrent;                                                        
  run=                   a.run;                
  nevent=                a.nevent;             
  ifo=                   a.ifo;             
  eventID=               a.eventID;            
  type=                  a.type;               
  stype=                 a.stype;              
  rate=                  a.rate;              

  gap=                   a.gap;                
  shift=                 a.shift;              
  strain=                a.strain;             
  phi=                   a.phi;             
  teta=                  a.teta;             
  bp=                    a.bp;             
  bx=                    a.bx;             

  volume=                a.volume;          
  size=                  a.size;            
  usize=                 a.usize;              

  power=                 a.power;           
  rLH=                   a.rLH;      
  gLH=                   a.gLH;      
  rSNR=                  a.rSNR;             
  gSNR=                  a.gSNR;             
  rSF=                   a.rSF;             
  gSF=                   a.gSF;             

  time=                  a.time;               
  itime=                 a.itime;               
  right=                 a.right;              
  left=                  a.left;               
  duration=              a.duration;        
  start=                 a.start;        
  stop=                  a.stop;         

  frequency=             a.frequency;          
  low=                   a.low;             
  high=                  a.high;            
  bandwidth=             a.bandwidth;       

  hrss=                  a.hrss;           
  noise=                 a.noise;           
  asymmetry=             a.asymmetry;    

  return *this;
}

//   Set branch addresses
void wbsingle::Init(TTree *tree)
{
   if (tree == 0) return;
   fChain    = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("run",&run);
   fChain->SetBranchAddress("nevent",&nevent);
   fChain->SetBranchAddress("ifo",&ifo);
   fChain->SetBranchAddress("eventID",&eventID);
   fChain->SetBranchAddress("type",&type);
   fChain->SetBranchAddress("stype",&stype);
   fChain->SetBranchAddress("rate",&rate);

   fChain->SetBranchAddress("gap",&gap);
   fChain->SetBranchAddress("shift",&shift);
   fChain->SetBranchAddress("strain",&strain);
   fChain->SetBranchAddress("phi",&phi);
   fChain->SetBranchAddress("teta",&teta);
   fChain->SetBranchAddress("bp",&bp);
   fChain->SetBranchAddress("bx",&bx);

   fChain->SetBranchAddress("volume",&volume);
   fChain->SetBranchAddress("size",&size);
   fChain->SetBranchAddress("usize",&usize);

   fChain->SetBranchAddress("power",&power);
   fChain->SetBranchAddress("rLH",&rLH);
   fChain->SetBranchAddress("gLH",&gLH);
   fChain->SetBranchAddress("rSNR",&rSNR);
   fChain->SetBranchAddress("gSNR",&gSNR);
   fChain->SetBranchAddress("rSF",&rSF);
   fChain->SetBranchAddress("gSF",&gSF);

   fChain->SetBranchAddress("time",&time);
   fChain->SetBranchAddress("itime",&itime);
   fChain->SetBranchAddress("right",&right);
   fChain->SetBranchAddress("left",&left);
   fChain->SetBranchAddress("duration",&duration);
   fChain->SetBranchAddress("start",&start);
   fChain->SetBranchAddress("stop",&stop);

   fChain->SetBranchAddress("frequency",&frequency);
   fChain->SetBranchAddress("low",&low);
   fChain->SetBranchAddress("high",&high);
   fChain->SetBranchAddress("bandwidth",&bandwidth);

   fChain->SetBranchAddress("hrss",&hrss);
   fChain->SetBranchAddress("noise",&noise);
   fChain->SetBranchAddress("asymmetry",&asymmetry);

   Notify();
}

Bool_t wbsingle::Notify()
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

Int_t wbsingle::GetEntry(Int_t entry) 
{ 
  if (!fChain) return 0; 
  return fChain->GetEntry(entry); 
};

void wbsingle::Show(Int_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}


//++++++++++++++++++++++++++++++++++++++++++++++
// set single event tree
//++++++++++++++++++++++++++++++++++++++++++++++
TTree* wbsingle::setTree()
{
   TTree* waveTree = new TTree("waveburst","waveburst");

 //==================================
 // Define trigger tree
 //==================================

   waveTree->Branch("run",         &run,          "run/I");
   waveTree->Branch("nevent",      &nevent,       "nevent/I");
   waveTree->Branch("eventID",     &eventID,      "eventID/I");
   waveTree->Branch("ifo",         &ifo,          "ifo/I");
   waveTree->Branch("type",        &type,         "type/I");
   waveTree->Branch("stype",       &stype,        "stype/I");
   waveTree->Branch("rate",        &rate,         "rate/I");
   
   waveTree->Branch("gap",         &gap,          "gap/F");
   waveTree->Branch("shift",       &shift,        "shift/F");
   waveTree->Branch("strain",      &strain,       "strain/D");
   waveTree->Branch("phi",         &phi,          "phi/F");
   waveTree->Branch("teta",        &teta,         "teta/F");
   waveTree->Branch("bp",          &bp,           "bp/F");
   waveTree->Branch("bx",          &bx,           "bx/F");
   
   waveTree->Branch("usize",       &usize,        "usize/I");
   waveTree->Branch("volume",      &volume,       "volume/I");
   waveTree->Branch("size",        &size,         "size/I");
   
   waveTree->Branch("power",       &power,        "power/F");
   waveTree->Branch("rLH",         &rLH,          "rLH/F");
   waveTree->Branch("gLH",         &gLH,          "gLH/F");
   waveTree->Branch("rSNR",        &rSNR,         "rSNR/F");
   waveTree->Branch("gSNR",        &gSNR,         "gSNR/F");
   waveTree->Branch("rSF",         &rSF,          "rSF/F");
   waveTree->Branch("gSF",         &gSF,          "gSF/F");
   
   waveTree->Branch("time",        &time,         "time/D");
   waveTree->Branch("itime",       &itime,        "itime/D");
   waveTree->Branch("right",       &right,        "right/F");
   waveTree->Branch("left",        &left,         "left/F");
   waveTree->Branch("start",       &start,        "start/D");
   waveTree->Branch("stop",        &stop,         "stop/D");
   waveTree->Branch("duration",    &duration,     "duration/F");
   
   waveTree->Branch("frequency",   &frequency,    "frequency/F");
   waveTree->Branch("low",         &low,          "low/F" );
   waveTree->Branch("high",        &high,         "high/F");
   waveTree->Branch("bandwidth",   &bandwidth,    "bandwidth/F");
   
   waveTree->Branch("hrss",        &hrss,         "hrss/D");
   waveTree->Branch("noise",       &noise,        "noise/D");
   waveTree->Branch("asymmetry",   &asymmetry,    "asymmetry/F");
   
   return waveTree;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++
// output wavecluster into tree
//++++++++++++++++++++++++++++++++++++++++++++++++++++
void wbsingle::output(TTree* waveTree, wavecluster** p, int np)
{ 
   int i,j,k,n,m;

//   if(p[0]->shift != p[1]->shift) cout<<"illegal shift parameter\n";
//   if(p[0]->shift != p[2]->shift) cout<<"illegal shift parameter\n";
//   if(p[1]->shift != p[2]->shift) cout<<"illegal shift parameter\n";
      
// arrays for cluster parameters

   wavearray<float> clusterID_wb1;
   wavearray<float> volume_wb1;
   wavearray<float> size_wb1;
   wavearray<float> time_wb1;
   wavearray<float> start_wb1;
   wavearray<float> stop_wb1;
   wavearray<float> frequency_wb1;
   wavearray<float> low_wb1;
   wavearray<float> high_wb1;
   wavearray<float> rLH_wb1;
   wavearray<float> gLH_wb1;
   wavearray<float> rSF_wb1;
   wavearray<float> gSF_wb1;
   wavearray<float> rSNR_wb1;
   wavearray<float> gSNR_wb1;
   wavearray<float> rate_wb1;
   wavearray<float> hrss_wb1;
   wavearray<float> noise_wb1;
   wavearray<float> asym_wb1;

   wavearray<int>   type_wb1;
   wavearray<int>   ifo_wb1;
   wavearray<double> gpsStart_wb1;
   wavearray<double> gpsStop_wb1;
   
// read cluster parameters

   n = 0;

   for(i=0; i<np; i++){
      clusterID_wb1.append(p[i]->get("ID"));
      volume_wb1.append(p[i]->get("volume"));
      size_wb1.append(p[i]->get("size"));
      time_wb1.append(p[i]->get("time",0,'S'));
      start_wb1.append(p[i]->get("start"));
      stop_wb1.append(p[i]->get("stop"));
      frequency_wb1.append(p[i]->get("frequency",0,'S'));
      low_wb1.append(p[i]->get("low"));
      high_wb1.append(p[i]->get("high"));
      rLH_wb1.append(p[i]->get("confidence",0,'R'));
      gLH_wb1.append(p[i]->get("confidence",0,'S'));
      rSF_wb1.append(p[i]->get("significance",0,'R'));
      gSF_wb1.append(p[i]->get("significance",0,'S'));
      rSNR_wb1.append(p[i]->get("SNR",0,'R'));
      gSNR_wb1.append(p[i]->get("SNR",0,'S'));
      rate_wb1.append(p[i]->get("rate"));
      hrss_wb1.append(p[i]->get("hrss"));
      noise_wb1.append(p[i]->get("noise"));
      asym_wb1.append(p[i]->get("asymmetry",0,'S'));
      
      m = clusterID_wb1.size();
      
      if(m != volume_wb1.size()) cout<<"output: illegal volume_wb1.size()\n";
      if(m != size_wb1.size()) cout<<"output: illegal size_wb1.size()\n";
      if(m != time_wb1.size()) cout<<"output: illegal time_wb1.size()\n";
      if(m != start_wb1.size()) cout<<"output: illegal start_wb1.size()\n";
      if(m != stop_wb1.size()) cout<<"output: illegal stop_wb1.size()\n";
      if(m != frequency_wb1.size()) cout<<"output: illegal frequency_wb1.size()\n";
      if(m != low_wb1.size()) cout<<"output: illegal low_wb1.size()\n";
      if(m != high_wb1.size()) cout<<"output: illegal high_wb1.size()\n";
      if(m != rLH_wb1.size()) cout<<"output: illegal rLH_wb1.size()\n";
      if(m != gLH_wb1.size()) cout<<"output: illegal gLH_wb1.size()\n";
      if(m != rSF_wb1.size()) cout<<"output: illegal rSF_wb1.size()\n";
      if(m != gSF_wb1.size()) cout<<"output: illegal gSF_wb1.size()\n";
      if(m != rSNR_wb1.size()) cout<<"output: illegal rSNR_wb1.size()\n";
      if(m != gSNR_wb1.size()) cout<<"output: illegal gSNR_wb1.size()\n";
      if(m != rate_wb1.size()) cout<<"output: illegal rate_wb1.size()\n";
      if(m != hrss_wb1.size()) cout<<"output: illegal hrss_wb1.size()\n";
      if(m != noise_wb1.size()) cout<<"output: illegal noise_wb1.size()\n";
      if(m != asym_wb1.size()) cout<<"output: illegal asym_wb1.size()\n";
      
      type_wb1.resize(m);
      ifo_wb1.resize(m);
      gpsStart_wb1.resize(m);
      gpsStop_wb1.resize(m);
      for(k=n; k<m; k++) {
	 type_wb1.data[k] = j;
	 ifo_wb1.data[k] = p[i]->ifo;
	 gpsStart_wb1.data[k] = p[i]->start;
	 gpsStop_wb1.data[k] = p[i]->stop;
      }
      n = m;
   }
   
// sort time
   
   wavearray<int> index_wb1(m);
   for(i=0; i<m; i++){ index_wb1.data[i]=i; }
   TMath::Sort(m,time_wb1.data,index_wb1.data,false);
   
   wavearray<double> STOP(np);
   for(i=0; i<np; i++) STOP.data[i] = p[i]->stop; 
   
//+++++++++++ lag definitions +++++++++++++++++++++++++
// o - start of zero lag interval
// lag>0:   o----------          L1
//              o-----------     H1,H2
//
// lag<0:       o----------      L1
//          o-----------         H1,H2
//+++++++++++++++++++++++++++++++++++++++++++++++++++++

   double START = p[0]->start;
   for(i=1; i<np; i++) if(START>p[i]->start) START=p[i]->start;
   
//   cout<<START<<" "<<SHIFT<<endl;

//Fill tree

   for(j=0; j<m; j++){
      k = index_wb1.data[j];

      if(!int(size_wb1.data[k]+0.5)) continue;
      
      run=          p[0]->run;
      nevent=       j;
      eventID=      clusterID_wb1.data[k];
      ifo=          ifo_wb1.data[k];
      type=         type_wb1.data[k];
      stype=        0.;
      rate=         rate_wb1.data[k];
      
      gap=          start_wb1.data[k] - STOP.data[ifo_wb1.data[k]-1];
      shift=        p[0]->shift;
      strain=       0.;
      
      usize=        0.;
      volume=       volume_wb1.data[k];
      size=         int(size_wb1.data[k]+0.5);
      
      power=        gSNR_wb1.data[k]/size_wb1.data[k];
      rLH=          rLH_wb1.data[k];
      gLH=          gLH_wb1.data[k];
      rSNR=         rSNR_wb1.data[k];
      gSNR=         gSNR_wb1.data[k];
      rSF=          rSF_wb1.data[k];
      gSF=          gSF_wb1.data[k];
      
      time=         time_wb1.data[k] + START;
      itime=        START;
      left=         start_wb1.data[k];
      right=        gpsStop_wb1.data[k]-gpsStart_wb1.data[k]-stop_wb1.data[k];
      duration=     stop_wb1.data[k] - start_wb1.data[k];
      start=        start_wb1.data[k] + gpsStart_wb1.data[k];
      stop=         stop_wb1.data[k] + gpsStart_wb1.data[k];
      
      frequency=    frequency_wb1.data[k];
      low=          low_wb1.data[k];
      high=         high_wb1.data[k];
      bandwidth=    high_wb1.data[k] - low_wb1.data[k];
      
      hrss=         pow(10.,hrss_wb1.data[k])/sqrt(16384.);
      noise=        pow(10.,noise_wb1.data[k])/sqrt(16384.);
      asymmetry=    asym_wb1.data[k];

      waveTree->Fill();
      STOP.data[ifo_wb1.data[k]-1] = stop_wb1.data[k];
      
   }
}


/*
Int_t wbsingle::LoadTree(Int_t entry)
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

Int_t wbsingle::Cut(Int_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}

void wbsingle::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L wbsingle.C
//      Root > wbsingle t
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
