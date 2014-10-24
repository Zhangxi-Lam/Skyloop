// Coherent WaveBurst production script for 5 detector network
{
  int  i,j,n,m;

  cout<<"network of "<<ifo[0]<<"x"<<ifo[1]<<"x"<<ifo[2]<<"x"<<ifo[3]<<"x"<<ifo[4]<<" detectors\n\n";

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  Meyer<double> B(1024);       // set wavelet for resampling
  Meyer<double> S(1024,2);     // set wavelet for production

  netcluster WC[lags];         // array of cluster structures
  netcluster wc;           
  netcluster* pwc;

  wavearray<double> x,y;       // temporary time series
  WSeries<double> wB(B);       // original WSeries
  WSeries<float>   v[5];       // noise variability

  detector        D0(ifo[0]);  // detector
  detector        D1(ifo[1]);  // detector
  detector        D2(ifo[2]);  // detector
  detector        D3(ifo[3]);  // detector
  detector        D4(ifo[4]);  // detector
  network         NET;         // network

  NET.add(&D0); 
  NET.add(&D1); 
  NET.add(&D2);
  NET.add(&D3);
  NET.add(&D4);
  NET.setSkyMaps(angle,Theta1,Theta2,Phi1,Phi2);
  NET.setIndex(&D1);
  NET.setAntenna(&D0); 
  NET.setAntenna(&D1); 
  NET.setAntenna(&D2);
  NET.setAntenna(&D3);
  NET.setAntenna(&D4);
  NET.setRunID(runID);
  NET.constraint(delta,gamma);
  NET.Edge = waveoffset;

  double mTau=NET.getDelay("MAX");  // maximum time delay 
  double dTau=NET.getDelay();       // time delay difference 
  cout<<"maximum time delay between detectors: "<<mTau<<endl;
  cout<<"       maximum time delay difference: "<<dTau<<endl;
  cout<<"           skymap angular resolution: "<<angle<<endl;
  cout<<"          skymap size in polar angle: "<<Theta1<<", "<<Theta2<<endl;
  cout<<"      skymap size in azimuthal angle: "<<Phi1<<", "<<Phi2<<endl;
  cout<<"                          constraint: "<<delta<<" "<<gamma<<endl<<endl;

  injection mdc(5);
  livetime live;
  netevent netburst(5);
  variability wavevar;
  wavenoise noiserms;

  gSystem->Exec("date");
  gSystem->Exec("hostname");

  // parse input

  if(!runID) cin>>runID>>input_dir>>input_label>>output_dir>>output_label;
  cout<<"job ID: "<<runID<<endl;
  cout<<"Input : "<<input_dir<<",  label: "<<input_label<<endl;
  cout<<"Output: "<<output_dir<<",  label: "<<output_label<<endl;
  NET.setRunID(runID);

  // read input framelist file

  char file[512], buFFer[1024];

  sprintf(file,"%s/%s%d",input_dir,fileNamesRaw[0],runID);
  FILE *pFile = fopen(file,"r");
  fgets(buFFer,512,pFile);
  fgets(buFFer,512,pFile);
  fgets(buFFer,512,pFile);
  double Tb = atof(buFFer)+waveoffset; // WB segment start
  fgets(buFFer,512,pFile);
  double Te = atof(buFFer)-waveoffset; // WB segment stop
  double dT = Te-Tb;                   // WB segment duration
  fclose(pFile);

  if(simulation) {         // reag MDC log file
    i=NET.readMDClog(injectionList,double(long(Tb)));
    printf("GPS: %16.6f saved,  injections: %d\n",double(long(Tb)),i);
  } 
  else {                   // setup constant shifts
    D0.shift(shift[0]);
    D1.shift(shift[1]);
    D2.shift(shift[2]);
    D3.shift(shift[3]);
    D4.shift(shift[4]);
  }

  NET.setTimeShifts(lags,step);
  if(strlen(finalSegmentList)>1) NET.readSEGlist(finalSegmentList,2);

// read delay filters

  detector Do[9];          // dummy detectors
  for(i=3; i<9; i++) {
    sprintf(file,"%s/Meyer1024_L%1d.dat",data_dir,i);
    cout<<file<<endl;
    Do[i].readFilter(file);
  }  

  char outFile[1024];
  char tmpFile[1024];
  char endFile[1024];
  FileStat_t fstemp;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// if simulation==1, loop on the injection strain factors
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  if(!simulation) nfactor = 1;

  for(int iii=0; iii<nfactor; iii++) {

    if(simulation) {
      factor=factors[iii];
      cout<<"factor="<<factor<<endl;
    
      sprintf(outFile,"%s/wave_%d_%d_%s_%g_id%d.root",
	      nodedir,int(Tb),int(dT),output_label,factor,runID);
      sprintf(endFile,"%s/wave_%d_%d_%s_%g_id%d.root",
	      output_dir,int(Tb),int(dT),output_label,factor,runID);
      sprintf(tmpFile,"%s/wave_%d_%d_%s_%g_id%d.root.tmp",
	      nodedir,int(Tb),int(dT),output_label,factor,runID);

      if(!gSystem->GetPathInfo(endFile,fstemp)) {
	printf("The file %s already exists - skip\n",endFile);
	fflush(stdout);
	continue;
      }

    }
    else {
      sprintf(outFile,"%s/wave_%d_%d_%s_id%d.root",
	      nodedir,int(Tb),int(dT),output_label,runID);
      sprintf(endFile,"%s/wave_%d_%d_%s_id%d.root",
	      output_dir,int(Tb),int(dT),output_label,runID);
      sprintf(tmpFile,"%s/wave_%d_%d_%s_id%d.root.tmp",
	      nodedir,int(Tb),int(dT),output_label,runID);
    }

    cout<<"output file on the node: "<<outFile<<endl;
    cout<<"final output file name : "<<endFile<<endl;
    cout<<"temporary output file  : "<<tmpFile<<endl;

    for(int j=0; j<NET.nLag; j++) WC[j].clear();

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input data
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    double start, end, rate, duration, Ao;

// D0

    sprintf(file,"%s/%s%d",input_dir,fileNamesRaw[0],runID);
    readframes1(file,channelNamesRaw[0],x);
    printf("file=%s\n",file);

    start    = x.start(); 
    rate     = x.rate();
    end      = start+x.size()/rate;
    duration = end-start-2*waveoffset;

    fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",
	    start,end,duration,rate);

    if(simulation) {
      sprintf(file,"%s/%s.lst.%d",input_dir,input_label,runID);
      readframes1(file,channelNamesMDC[0],y);
      y*=factor;
      x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
      y.resize(1);
    }

    wB.Forward(x,levelR); 
    wB.getLayer(x,0); 
    D0.getTFmap()->Forward(x,S,levelD);
    D0.getTFmap()->lprFilter(2,0,120.,4.);
    D0.getTFmap()->setlow(fLow);
    D0.white(60.,1,8.,30.);
    D0.getTFmap()->Inverse(3);
    D0.getTFmap()->lprFilter(2,0,120.,4.);
    D0.getTFmap()->Forward(3);
    v[0] = D0.getTFmap()->variability();

    cout<<"After "<<ifo[0]<<" data conditioning"<<endl; 
    gSystem->Exec("date");

// D1

    sprintf(file,"%s/%s%d",input_dir,fileNamesRaw[1],runID);
    readframes1(file,channelNamesRaw[1],x);
    
    if(start!=x.start() || rate!=x.rate()) {
      fprintf(stderr,"%s/%s mismatch: %f, %f, %f, %f\n",
	      ifo[1],ifo[0],start,x.start(),rate,x.rate()); exit(1);
    }

    if(simulation) {
      sprintf(file,"%s/%s.lst.%d",input_dir,input_label,runID);
      readframes1(file,channelNamesMDC[1],y);
      y*=factor;
      x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
      y.resize(1);
    }

    wB.Forward(x,levelR);
    wB.getLayer(x,0);
    D1.getTFmap()->Forward(x,S,levelD);
    D1.getTFmap()->lprFilter(2,0,120.,4.);
    D1.getTFmap()->setlow(fLow);
    D1.white(60.,1,8.,30.);
    D1.getTFmap()->Inverse(3);
    D1.getTFmap()->lprFilter(2,0,120.,4.);
    D1.getTFmap()->Forward(3);
    v[1] = D1.getTFmap()->variability();

    cout<<"After "<<ifo[1]<<" data conditioning"<<endl; 
    gSystem->Exec("date");

// D2

    sprintf(file,"%s/%s%d",input_dir,fileNamesRaw[2],runID);
    readframes1(file,channelNamesRaw[2],x);

    if(start!=x.start() || rate!=x.rate()) {
      fprintf(stderr,"%s/%s mismatch: %f, %f, %f, %f\n",
	      ifo[2],ifo[0],start,x.start(),rate,x.rate()); exit(1);
    }

    if(simulation) {
      sprintf(file,"%s/%s.lst.%d",input_dir,input_label,runID);
      readframes1(file,channelNamesMDC[2],y);
      y*=factor;
      x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
      y.resize(1);
    }

    wB.Forward(x,levelR);
    wB.getLayer(x,0);
    D2.getTFmap()->Forward(x,S,levelD);
    D2.getTFmap()->lprFilter(2,0,120.,4.);
    D2.getTFmap()->setlow(fLow);
    D2.white(60.,1,8.,30.);
    D2.getTFmap()->Inverse(3);
    D2.getTFmap()->lprFilter(2,0,120.,4.);
    D2.getTFmap()->Forward(3);
    v[2] = D2.getTFmap()->variability();
      
    cout<<"After "<<ifo[2]<<" data conditioning"<<endl; 
    gSystem->Exec("date");
      
// D3

    sprintf(file,"%s/%s%d",input_dir,fileNamesRaw[3],runID);
    readframes1(file,channelNamesRaw[3],x);

    if(start!=x.start() || rate!=x.rate()) {
      fprintf(stderr,"%s/%s mismatch: %f, %f, %f, %f\n",
	      ifo[3],ifo[0],start,x.start(),rate,x.rate()); exit(1);
    }

    if(simulation) {
      sprintf(file,"%s/%s.lst.%d",input_dir,input_label,runID);
      readframes1(file,channelNamesMDC[3],y);
      y*=factor;
      x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
      y.resize(1);
    }

    wB.Forward(x,levelR);
    wB.getLayer(x,0);
    D3.getTFmap()->Forward(x,S,levelD);
    D3.getTFmap()->lprFilter(2,0,120.,4.);
    D3.getTFmap()->setlow(fLow);
    D3.white(60.,1,8.,30.);
    D3.getTFmap()->Inverse(3);
    D3.getTFmap()->lprFilter(2,0,120.,4.);
    D3.getTFmap()->Forward(3);
    v[3] = D3.getTFmap()->variability();
      
    cout<<"After "<<ifo[3]<<" data conditioning"<<endl; 
    gSystem->Exec("date");
      
// D4

    sprintf(file,"%s/%s%d",input_dir,fileNamesRaw[4],runID);
    readframes1(file,channelNamesRaw[4],x);

    if(start!=x.start() || rate!=x.rate()) {
      fprintf(stderr,"%s/%s mismatch: %f, %f, %f, %f\n",
	      ifo[4],ifo[0],start,x.start(),rate,x.rate()); exit(1);
    }

    if(simulation) {
      sprintf(file,"%s/%s.lst.%d",input_dir,input_label,runID);
      readframes1(file,channelNamesMDC[4],y);
      y*=factor;
      x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
      y.resize(1);
    }

    wB.Forward(x,levelR);
    wB.getLayer(x,0);
    D4.getTFmap()->Forward(x,S,levelD);
    D4.getTFmap()->lprFilter(2,0,120.,4.);
    D4.white(60.,1,8.,30.);
    D4.getTFmap()->Inverse(3);
    D4.getTFmap()->lprFilter(2,0,120.,4.);
    D4.getTFmap()->Forward(3);
    v[4] = D4.getTFmap()->variability();
      
    cout<<"After "<<ifo[4]<<" data conditioning"<<endl; 
    gSystem->Exec("date");
      
    double R = x.rate();          // original data rate

    wB.resize(1);
    x.resize(1);
    
    printf("size=%d, segment start=%16.6f \n",
	   D0.getTFmap()->size(), D0.getTFmap()->start());

    cout<<"live time: "<<NET.setVeto(4.)<<endl;  // set veto array 

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// initialization of the output root file 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    TFile *froot = new TFile(tmpFile, "RECREATE");

    TTree* net_tree = netburst.setTree();
    TTree* live_tree= live.setTree();

    if(simulation) {
      TTree* mdc_tree = mdc.setTree();
    }
    else {
      TTree* var_tree = wavevar.setTree();
      TTree* noise_tree = noiserms.setTree();
    }

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// band pass filtering
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    D0.bandPass();
    D1.bandPass();
    D2.bandPass();
    D3.bandPass();
    D4.bandPass();

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// start of the coherent search
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    double Ao;
    
    // loop over TF resolutions
    
    for(i=levelD; i>=l_low; i--) {
	  
      if(i<=l_high) {
	    	    	    
	NET.setDelayFilters(&Do[i],mTau+0.002);
	    
	Ao = NET.threshold(bpp,dTau);
	NET.set2or(Ao*Ao*x2or);
	cout<<"pixel threshold in units of noise rms: "<<Ao<<endl;
	cout<<"2 OR  threshold in units of noise var: "<<Ao*Ao*x2or<<endl;
	
	cout<<"total    pixels: "<<NET.coherence5(Ao)<<"  ";
	    
	n = size_t(2.*Tgap*D0.getTFmap()->resolution(0)+0.1);
	m = size_t(Fgap/D0.getTFmap()->resolution(0)+0.1);
	    
	cout<<"clusters: "<<NET.cluster(n,m)<<"  "<<endl;
	cout<<"selected pixels: "<<NET.likelihood5('E',false,Acore)<<"\n";

	for(j=0; j<NET.nLag; j++) {
	  wc = *(NET.getwc(j));
	  cout<<wc.csize()<<"|"<<wc.size()<<"|"<<WC[j].append(wc)<<" "; 
	}
	cout<<endl;
      }
	  
      if(i>l_low) NET.Inverse(1); 
      gSystem->Exec("date");
	  
    }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// supercluster analysis
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for(j=0; j<lags; j++){
      pwc = NET.getwc(j); *pwc = WC[j];
      m = pwc->supercluster('L',NET.e2or,true);
      cout<<m<<"|"<<pwc->size()<<" ";
    }
    cout<<endl;
    cout<<"reconstructed events: "<<NET.events()<<"\n";
    gSystem->Exec("date");

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// likelihood
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for(i=l_low; i<=l_high; i++) {
      NET.setDelayFilters(&Do[i],mTau+0.002);
      cout<<"correlated energy for level "<<i<<": "<<NET.likelihood5('c',false,Acore)<<"\n";
      if(i<l_high) NET.Forward(1);
    }

    cout<<"rejected weak pixels: "<<NET.netcut(eCOR,'x',0,1)<<"\n";    // remove weak glitches
    cout<<"rejected loud pixels: "<<NET.netcut(netCOR-0.1,'c',0,1)<<"\n";  // remove loud glitches
    cout<<"reconstructed events: "<<NET.events()<<"\n";
    gSystem->Exec("date");

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// final likelihood
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    NET.optim = false;
    for(i=l_high; i>=l_low; i--) {
      NET.setDelayFilters(&Do[i],mTau+0.002);
      cout<<"final likelihood  for level "<<i<<": "<<NET.likelihood5('h',false,Acore)<<"\n";
      NET.setRank(8.);  
      if(i>l_low) NET.Inverse(1);         
    }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// optimal resolution
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for(j=0; j<lags; j++){
      pwc = NET.getwc(j); WC[j]=*pwc; *pwc = WC[j];
      m = pwc->supercluster('L',NET.e2or,true);
      cout<<m<<"|"<<pwc->size()<<" ";
    }
    cout<<endl;

    cout<<"\nSearch done\n";
    cout<<"rejected      pixels: "<<NET.netcut(eCOR,'x',0,1)<<"\n";    // remove weak glitches
    cout<<"rejected loud pixels: "<<NET.netcut(netCOR,'c',0,1)<<"\n";  // remove loud glitches
    cout<<"reconstructed events: "<<NET.events()<<"\n";

    if(simulation) NET.printwc(0);

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// save data in root file
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    live.output(live_tree,&NET);

    if(simulation) {
      netburst.output(net_tree,&NET,factor);
      mdc.output(mdc_tree,&NET,factor);
    }
    else {
      netburst.output(net_tree,&NET);
      wavevar.output(var_tree,&v[0],1,waveoffset);
      wavevar.output(var_tree,&v[1],2,waveoffset);
      wavevar.output(var_tree,&v[2],3,waveoffset);
      wavevar.output(var_tree,&v[3],4,waveoffset);
      wavevar.output(var_tree,&v[4],5,waveoffset);
      noiserms.output(noise_tree,&D0.nRMS,1,R/2);
      noiserms.output(noise_tree,&D1.nRMS,2,R/2);
      noiserms.output(noise_tree,&D2.nRMS,3,R/2);
      noiserms.output(noise_tree,&D3.nRMS,4,R/2);
      noiserms.output(noise_tree,&D4.nRMS,5,R/2);
    }

    froot->Write();
    froot->Close();

    char command[512];
    sprintf(command,"mv %s %s", tmpFile, outFile);
    gSystem->Exec(command);
    sprintf(command,"cp %s %s",outFile,output_dir);
    gSystem->Exec(command);
  }

  cout<<"Stopping the job "<<runID<<endl;
  gSystem->Exec("date");
  
  return 0;

}







