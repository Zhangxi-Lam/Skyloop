
#define USE_GAUS_NOISE

// frame files list
#define FRLIST_NAME  "S6A_LDR_L1_LDAS_C02_L2.frames"
// channel name
#define CHANNEL_NAME "L1:LDAS-STRAIN"

watplot* plot;

void 
Test2G_Whitening() {
// this tutorial shows how to whiten colored gaussian data  

  //Draw plot of strain & whitened PSD data
  plot = new watplot("plot");
  TCanvas *c1 = plot->canvas;
  c1->SetCanvasSize(800,1200);
  c1->Divide(2,4);

  // whitening parameters
  double whiteWindow = 60.;   // [sec] time window dT. if = 0 - dT=T, where T is segment duration
  double whiteStride = 20.;   // [sec] noise sampling time stride
  // segment parameters
  double segEdge     = 8.;    // wavelet boundary offset [sec]
  double segLen      = 616.;  // Segment length [sec]
  // frequency range
  int RATE           = 16384; // data sample rate
#ifdef USE_GAUS_NOISE
  double fLow        = 16.;   // low frequency of the search
#else
  double fLow        = 48.;   // low frequency of the search
#endif
  double fHigh       = 2048.; // high frequency of the search

  wavearray<double> x;

#ifdef USE_GAUS_NOISE
  // generate colored gaussian noise
  TString fName = TString(gSystem->ExpandPathName("$HOME_WAT"))+
                  "/tools/cwb/plugins/strains/advLIGO_NSNS_Opt_8khz_one_side.txt";
  
  CWB::Toolbox::getSimNoise(x, fName, 1000, 1);
  x.resize(segLen*x.rate());
#else
  // real data noise 
  double start = 931081124;
  double stop = start+segLen;
  // read frame lists
  CWB::frame fr(FRLIST_NAME);
  int nfiles=fr.getNfiles();
  cout << "nfiles : " << nfiles << endl;
  // get frame list into frfile structure
  frfile FRF = fr.getFrList(start, stop, 0);
  // read data into x from frame files
  fr.readFrames(FRF,CHANNEL_NAME,x);
  x.start(0);
#endif

  // ----------------------------------------------------------
  // resample data from 16384 Hz -> 4096 Hz
  // ----------------------------------------------------------
  Meyer<double> B(1024);           // set wavelet for resampling
  WSeries<double> W;               // WSeries used for resampling

  W.Forward(x,B,2);
  W.getLayer(x,0);		   // extract resampled data

  RATE = RATE>>2;		   // 16384 -> 4096

  // ----------------------------------------------------------
  // plot WDM strain data distribution
  // ----------------------------------------------------------
  double rms=x.rms();
  TH1F* histS = new TH1F("histS","histS",100,-4*rms,+4*rms);
  for(int i=0;i<x.size();i++) histS->Fill(x[i]); 
  c1->cd(1);
  gStyle->SetLineColor(kBlack);
  histS->Draw();
  histS->SetTitle("Strain data samples distribution");

  // ----------------------------------------------------------
  // plot strain PSD data
  // ----------------------------------------------------------
  c1->cd(2);
  gPad->SetLogx();
  gPad->SetLogy();
  gPad->SetGridx();
  gPad->SetGridy();
  plot->plot(x, "alp", 1, segEdge, segLen-segEdge, true, fLow, fHigh, true, 8);
  plot->graph[0]->SetTitle("Strain data PSD");;

  // ----------------------------------------------------------
  // initialize WDM used for whitening
  // ----------------------------------------------------------
  int level=10;			// decomposition level
  int rate  = RATE>>level;
  double dt = 1000./rate;
  double df = RATE/2./double(1<<level);
  cout << "level : " << level << "\t rate : " << rate
       << "\t df(hz) : " << df << "\t dt(ms) : " << dt << endl;
  int layers = 1<<level;
  WDM<double> WDMlpr(layers,layers,6,10);

  // ----------------------------------------------------------
  // Time Frequency x data transform
  // ----------------------------------------------------------
  WSeries<double> w;            // mdc WSeries
  w.Forward(x,WDMlpr);          // x data TF tranform 

  int wlayers = w.maxLayer()+1;  // numbers of frequency bins (first & last bins have df/2)
  int wslices = w.sizeZero();    // number of time bins

  float wdf = w.resolution();    // frequency bin resolution (hz)
  float wdt = 1./(2*wdf);        // time bin resolution (sec)

  int wedge = segEdge/wdt;       // scratch data

  // ----------------------------------------------------------
  // plot strain TF data coefficients
  // ----------------------------------------------------------
  c1->cd(3);
  gPad->SetLogz();
  plot->plot(w, 2, segEdge, segLen-segEdge,const_cast<char*>("COLZ"));
  plot->hist2D->GetYaxis()->SetRangeUser(fLow, fHigh);
  plot->hist2D->GetZaxis()->SetLabelSize(0.04);
  char title[64];
  sprintf(title,"Strain data TF sqrt((E00+E90)/2) - level:%d - dt:%g (ms) - df=%g (Hz)", level,dt,df);
  plot->hist2D->SetTitle(title);
  plot->hist2D=NULL;

  // ----------------------------------------------------------
  // whitening x data
  // ----------------------------------------------------------
  WSeries<double> nRMS;         // noise RMS
  w.setlow(fLow);
  w.sethigh(fHigh);
  nRMS = w.white(whiteWindow,0,segEdge,whiteStride);  // calculate noise rms
  w.white(nRMS,1);              // whiten  0 phase WSeries
  w.white(nRMS,-1);             // whiten 90 phase WSeries

  // ----------------------------------------------------------
  // plot nRMS data
  // ----------------------------------------------------------
  // nRMS do not come from a TF transform
  // nRMS params must be fixed before to pass to watplot
  int levels = nRMS.getLevel()+1;               // number of levels
  int slices = nRMS.size()/levels;              // number of nRMS samples
  double length = slices*whiteStride;           // nRMS len in sec
  double fNinquist = nRMS.rate()/2.;
  nRMS.rate(nRMS.size()/length);
  WDM<double>* wdm = (WDM<double>*) nRMS.pWavelet;
  wdm->nSTS=nRMS.size();

  c1->cd(4);
  gPad->SetLogy();
  gPad->SetLogz();
  gPad->SetGridx();
  gPad->SetGridy();
  double start = nRMS.start();
  double stop  = nRMS.start()+nRMS.size()/nRMS.rate();
  plot->plot(nRMS, 3, start, stop,const_cast<char*>("COLZ"));
  plot->hist2D->GetYaxis()->Set(plot->hist2D->GetNbinsY(),0, fNinquist); // correct freq range
  plot->hist2D->GetYaxis()->SetRangeUser(fLow, fHigh);
  plot->hist2D->GetZaxis()->SetLabelSize(0.04);
  char title[64];
  sprintf(title,"nRMS TF data");
  plot->hist2D->SetTitle(title);
  plot->hist2D=NULL;

  // ----------------------------------------------------------
  // plot whitened TF data coefficients
  // ----------------------------------------------------------
  c1->cd(5);
  plot->plot(w, 2, segEdge, segLen-segEdge,const_cast<char*>("COLZ"));
  plot->hist2D->GetYaxis()->SetRangeUser(fLow, fHigh);
  char title[64];
  sprintf(title,"Whiten data TF sqrt((E00+E90)/2) - level:%d - dt:%g (ms) - df=%g (Hz)", level,dt,df);
  plot->hist2D->SetTitle(title);
  plot->hist2D->GetZaxis()->SetLabelSize(0.04);

  // ----------------------------------------------------------
  // plot WDM whitened TF data coefficients distribution
  // ----------------------------------------------------------

  TH1F* histW = new TH1F("histW","histW",100,-4,+4);
  for(int i=wedge;i<=wslices-wedge;i++) { // exclude scratch data 
    for(int j=1;j<=wlayers;j++) {
      int I = i-1;
      double J = (j-1)+0.01;
      float aa = w.getSample(I,J);	  // phase 00 coefficients
      float AA = w.getSample(I,-J);	  // phase 90 coefficients
      histW->Fill(aa); 
      //histW->Fill(AA); 
    }
  }
  c1->cd(6);
  gStyle->SetOptFit();
  gStyle->SetLineColor(kBlack);
  gPad->SetLogx(false);
  histW->Draw();
  histW->Fit("gaus"); 		  	  // Fit Histogram
  histW->SetTitle("Whitened WDM coefficients distribution");

  // ----------------------------------------------------------
  // plot white PSD data
  // ----------------------------------------------------------
  w.Inverse();
  c1->cd(7);
  gPad->SetLogx();
  gPad->SetGridx();
  gPad->SetGridy();
  plot->plot(w, "alp", 1, segEdge, segLen-segEdge, true, fLow, fHigh, true, 8);
  plot->graph[1]->SetTitle("Whitened data PSD");

  // ----------------------------------------------------------
  // plot average & rms of the whitened data vs time
  // ----------------------------------------------------------
  int M = w.rate()/16;
  int N = w.size()/M;
  double R = w.rate()/M;
  wavearray<double> TIM(N); TIM.start(0); TIM.rate(R);
  wavearray<double> AVR(N); AVR.start(0); AVR.rate(R);
  wavearray<double> RMS(N); RMS.start(0); RMS.rate(R);
  for(int n=0;n<N;n++) {
    double avr=0;
    double rms=0;  
    for(int m=0;m<M;m++) {
      int p=n*M+m;
      avr+=w.data[p];
      rms+=w.data[p]*w.data[p];
    }
    avr /= M;
    rms = sqrt(rms/M-avr*avr);
    AVR[n]=avr;
    RMS[n]=rms;
    TIM[n]=n*(1./double(M));
  }
  c1->cd(8);
  gPad->SetGridx();
  gPad->SetGridy();
  plot->plot(AVR, "alp", 1, segEdge, segLen-segEdge);
  plot->graph[2]->SetTitle("AVR (black )RMS (red) of the whitened data");
  plot->graph[2]->GetHistogram()->GetYaxis()->SetRangeUser(-1,2);
  plot->plot(RMS, "same", 2, segEdge, segLen-segEdge);

  return;
}
