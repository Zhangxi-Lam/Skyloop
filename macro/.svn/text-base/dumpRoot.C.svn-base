void output(TTree* wave_tree, wavecluster** p)
{
   int i,k,n,m;
   waveevent waveburst;

/*
   cout<<"output file name: "<<name<<endl;
   TFile *f  = new TFile(name, "RECREATE");
   TTree* wave_tree;
   wave_tree = new TTree("waveburst","waveburst");
*/

 //==================================
 // Define Ntuple
 //==================================

   wave_tree->Branch("run",         &waveburst.run,          "run/I");
   wave_tree->Branch("nevent",      &waveburst.nevent,       "nevent/I");
   wave_tree->Branch("eventID",     &waveburst.eventID,      "eventID/I");
   wave_tree->Branch("ifo",         &waveburst.ifo,          "ifo/I");
   wave_tree->Branch("type",        &waveburst.type,         "type/I");
   wave_tree->Branch("stype",       &waveburst.stype,        "stype/I");
   wave_tree->Branch("rate",        &waveburst.rate,         "rate/I");
   
   wave_tree->Branch("gap",         &waveburst.gap,          "gap/F");
   wave_tree->Branch("shift",       &waveburst.shift,        "shift/F");
   wave_tree->Branch("strain",      &waveburst.strain,       "strain/D");
   
   wave_tree->Branch("usize",       &waveburst.usize,        "usize/I");
   wave_tree->Branch("volume",      &waveburst.volume,       "volume/I");
   wave_tree->Branch("size",        &waveburst.size,         "size/I");
   
   wave_tree->Branch("power",       &waveburst.power,        "power/F");
   wave_tree->Branch("rLH",         &waveburst.rLH,          "rLH/F");
   wave_tree->Branch("gLH",         &waveburst.gLH,          "gLH/F");
   wave_tree->Branch("rSNR",        &waveburst.rSNR,         "rSNR/F");
   wave_tree->Branch("gSNR",        &waveburst.gSNR,         "gSNR/F");
   wave_tree->Branch("rSF",         &waveburst.rSF,          "rSF/F");
   wave_tree->Branch("gSF",         &waveburst.gSF,          "gSF/F");
   
   wave_tree->Branch("time",        &waveburst.time,         "time/D");
   wave_tree->Branch("itime",       &waveburst.itime,        "itime/D");
   wave_tree->Branch("start",       &waveburst.start,        "start/F");
   wave_tree->Branch("stop",        &waveburst.stop,         "stop/F");
   wave_tree->Branch("startGPS",    &waveburst.startGPS,     "startGPS/D");
   wave_tree->Branch("stopGPS",     &waveburst.stopGPS,      "stopGPS/D");
   wave_tree->Branch("duration",    &waveburst.duration,     "duration/F");
   
   wave_tree->Branch("frequency",   &waveburst.frequency,    "frequency/F");
   wave_tree->Branch("low",         &waveburst.low,          "low/F" );
   wave_tree->Branch("high",        &waveburst.high,         "high/F");
   wave_tree->Branch("bandwidth",   &waveburst.bandwidth,    "bandwidth/F");
   
   wave_tree->Branch("noise",       &waveburst.noise,        "noise/D");
   wave_tree->Branch("xcorrelation",&waveburst.xcorrelation, "xcorrelation/F");
   
// arrays for cluster parameters

   wavearray<float> clusterID[3];
   wavearray<float> volume[3];
   wavearray<float> size[3];
   wavearray<float> time[3];
   wavearray<float> Start[3];
   wavearray<float> Stop[3];
   wavearray<float> frequency[3];
   wavearray<float> low[3];
   wavearray<float> high[3];
   wavearray<float> rLH[3];
   wavearray<float> gLH[3];
   wavearray<float> rSF[3];
   wavearray<float> gSF[3];
   wavearray<float> rSNR[3];
   wavearray<float> gSNR[3];
   wavearray<float> rate[3];
   
   double shift[3];
   int    ifo[3];
   

// read cluster parameters

   for(int i=0; i<3; i++){
      clusterID[i] = p[i]->get("ID");
      volume[i]    = p[i]->get("volume");
      size[i]      = p[i]->get("size",0);
      time[i]      = p[i]->get("time",2);
      Start[i]     = p[i]->get("start",0);
      Stop[i]      = p[i]->get("stop",0);
      frequency[i] = p[i]->get("frequency",2);
      low[i]       = p[i]->get("low",0);
      high[i]      = p[i]->get("high",0);
      rLH[i]       = p[i]->get("likelihood",1);
      gLH[i]       = p[i]->get("likelihood",2);
      rSF[i]       = p[i]->get("significance",1);
      gSF[i]       = p[i]->get("significance",2);
      rSNR[i]      = p[i]->get("energy",1);
      gSNR[i]      = p[i]->get("energy",2);
      rate[i]      = p[i]->get("rate",0);
      ifo[i]       = p[i]->ifo;
      shift[i]     = p[i]->shift;
   }
   
// sort time

   int N[3];
   N[0] = time[0].size();
   N[1] = time[1].size();
   N[2] = time[2].size();
   
   wavearray<int> index(N[0]+N[1]+N[2]);
   wavearray<float> T(N[0]+N[1]+N[2]);
   T.cpf(time[0],N[0]);
   T.cpf(time[1],N[1],0,N[0]);
   T.cpf(time[2],N[2],0,N[0]+N[1]);
   for(i=0; i<index.size(); i++){ index.data[i]=i; }
   TMath::Sort(index.size(),T.data,index.data,false);

   double stop[3];
   stop[0] = p[0]->stop; 
   stop[1] = p[1]->stop; 
   stop[2] = p[2]->stop; 

//Fill tree

   for(int i=0; i<index.size(); i++){
      k = index.data[i];
      n = 0;
      if(k>=N[0]) { n=1; k-=N[0]; }
      if(k>=N[1]) { n=2; k-=N[1]; }
      m=k;

      if(!int(size[n].data[m]+0.5)) continue;
      
      waveburst.run=          0.;
      waveburst.nevent=       i;
      waveburst.eventID=      clusterID[n].data[m];
      waveburst.ifo=          p[n]->ifo;
      waveburst.type=         0.;
      waveburst.stype=        0.;
      waveburst.rate=         rate[n].data[m];
      
      waveburst.gap=          Start[n].data[m] - stop[n];
      waveburst.shift=        p[n]->shift;
      waveburst.strain=       0.;
      
      waveburst.usize=        0.;
      waveburst.volume=       volume[n].data[m];
      waveburst.size=         int(size[n].data[m]+0.5);
      
      waveburst.power=        gSNR[n].data[m]/size[n].data[m];
      waveburst.rLH=          rLH[n].data[m];
      waveburst.gLH=          gLH[n].data[m];
      waveburst.rSNR=         rSNR[n].data[m];
      waveburst.gSNR=         gSNR[n].data[m];
      waveburst.rSF=          rSF[n].data[m];
      waveburst.gSF=          gSF[n].data[m];
      
      waveburst.time=         time[n].data[m] + p[n]->start;
      waveburst.itime=        0.;
      waveburst.start=        Start[n].data[m];
      waveburst.stop=         Stop[n].data[m];
      waveburst.duration=     Stop[n].data[m] - Start[n].data[m];
      waveburst.startGPS=     Start[n].data[m] + p[n]->start;
      waveburst.stopGPS=      Stop[n].data[m] + p[n]->start;
      
      waveburst.frequency=    frequency[n].data[m];
      waveburst.low=          low[n].data[m];
      waveburst.high=         high[n].data[m];
      waveburst.bandwidth=    high[n].data[m] - low[n].data[m];
      
      waveburst.noise=        0.;
      waveburst.xcorrelation= 0.;
      
      wave_tree->Fill();
      stop[n] = Stop[n].data[m];
      
   }
   
//   f->Write();
//   f->Close();
   
}
