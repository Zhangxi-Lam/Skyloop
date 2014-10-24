// Coherent WaveBurst production script for 3 detector network
{
  int  i,j,n,m;

  cout<<"\n network of ";
  for(i=0; i<nIFO; i++) cout<<ifo[i]<<" ";
  cout<<" detectors\n\n";

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  Meyer<double> B(1024);       // set wavelet for resampling
  Meyer<double> S(1024,2);     // set wavelet for production

  netcluster WC[lags];         // array of cluster structures
  netcluster wc;           
  netcluster* pwc;

  wavearray<double> x;         // temporary time series
  wavearray<double> y[6];         // temporary time series
  WSeries<double> wM;          // mdc WSeries
  WSeries<float>   v[nIFO];    // noise variability
  detector*       pD[nIFO];    // poiners to detectors
  WSeries<double>* pTF[nIFO];  // pointer to WSeries

  for(i=0; i<nIFO; i++) pD[i] = new detector(ifo[i]);

  network         NET;         // network

  for(i=0; i<nIFO; i++) NET.add(pD[i]); 
  NET.setSkyMaps(angle,Theta1,Theta2,Phi1,Phi2);
  NET.setIndex(pD[1]);
  for(i=0; i<nIFO; i++) NET.setAntenna(pD[i]); 
  NET.constraint(delta,gamma);
  NET.Edge = waveoffset;

  double mTau=NET.getDelay("MAX");  // maximum time delay 
  double dTau=NET.getDelay();       // time delay difference 
  cout<<"maximum time delay between detectors: "<<mTau<<endl;
  cout<<"       maximum time delay difference: "<<dTau<<endl;
  cout<<"           skymap angular resolution: "<<angle<<endl;
  cout<<"          skymap size in polar angle: "<<Theta1<<", "<<Theta2<<endl;
  cout<<"      skymap size in azimuthal angle: "<<Phi1<<", "<<Phi2<<endl;
  cout<<"                          constraint: "<<delta<<endl<<endl;

  injection mdc(nIFO);
  livetime live;
  netevent netburst(nIFO);
  variability wavevar;
  wavenoise noiserms;

  gSystem->Exec("date");
  gSystem->Exec("hostname");

  // parse input

  if(!runID) cin>>runID>>input_dir>>input_label>>output_dir>>output_label;
  cout<<"job ID: "<<runID<<endl;
  cout<<"Input : "<<input_dir<<"\n  label: "<<input_label<<endl;
  cout<<"Output: "<<output_dir<<"\n  label: "<<output_label<<endl;
  NET.setRunID(runID);

  // read input framelist file

  char file[512], buFFer[1024];

  sprintf(file,"%s/%s",input_dir,fileNamesRaw[0]);
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
    for(i=0; i<nIFO; i++) pD[i].shift(shift[i]);
  }

  NET.setTimeShifts(lags,step);
  if(strlen(finalSegmentList)>1) NET.readSEGlist(finalSegmentList,2);

// read delay filters

  detector Do[levelD+1];          // dummy detectors
  for(i=l_low; i<=levelD; i++) {
    sprintf(file,"%s/Meyer1024_L%1d.dat",data_dir,i);
    cout<<file<<endl;
    Do[i].readFilter(file);
  }  

  char outFile[1024];
  char tmpFile[1024];
  char endFile[1024];
  char command[1024];
  FileStat_t fstemp;

// read and dump data on local disk (nodedir)

  for(i=0; i<nIFO; i++) {
    sprintf(file,"%s/%s",input_dir,fileNamesRaw[i]);
    readframes1(file,channelNamesRaw[i],x);
    sprintf(file,"%s/%s_%d.dat",nodedir,ifo[i],runID);
    y[i]=x; y[i+3]=x;
    pTF[i] = pD[i]->getTFmap();
    pTF[i]->Forward(x,B,levelR);
    pTF[i]->getLayer(x,0);
    pTF[i]->Forward(x,S,levelF);
    pTF[i]->DumpBinary(file);

    fprintf(stdout,"start=%f duration=%f rate=%f\n",
	    x.start(),x.size()/x.rate(),x.rate());
    if(i>0 && pTF[0]->start() != x.start()) exit(1);
    if(i>0 && pTF[0]->rate()  != x.rate())  exit(1);
 
    if(simulation) {
      sprintf(file,"%s/%s%d",mdc_dir,input_label,runID);
      readframes1(file,channelNamesMDC[i],x);
      //      pD[i]->setsim(x,NET.getmdcTime());
      sprintf(file,"%s/mdc%s_%d.dat",nodedir,ifo[i],runID);
      wM.Forward(x,B,levelR); 
      wM.getLayer(x,0); 
      wM.Forward(x,S,levelF);
      wM.DumpBinary(file);

      fprintf(stdout,"start=%f duration=%f rate=%f\n",
	      x.start(),x.size()/x.rate(),x.rate());
      if(i>0 && wM.start() != x.start()) exit(1);
      if(i>0 && wM.rate() != x.rate())   exit(1);
    }
  }

  double R = x.rate();          // data rate
  x.resize(0);

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

    for(j=0; j<NET.nLag; j++) WC[j].clear();
    gSystem->Exec("date");

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// data conditioning
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    int nx, ny, nn;

    for(i=0; i<nIFO; i++) {
      sprintf(file,"%s/%s_%d.dat",nodedir,ifo[i],runID);
      pTF[i]->setLevel(levelF);
      pTF[i]->ReadBinary(file);
      n = pTF[i]->size();
      
      if(simulation) {
	sprintf(file,"%s/mdc%s_%d.dat",nodedir,ifo[i],runID);
	wM.ReadBinary(file); wM*=factor;
	m  = wM.size();
	nx = (wM.start()-pTF[i]->start())*wM.rate();
	nn = m-n + int((wM.start()-pTF[i]->start())*wM.rate());
	ny = nx<0 ? -nx :  0;
	nx = nx<0 ?   0 : nx;
	nn = nn<0 ? m-ny : n-nx;
	pTF[i]->add(wM,nn,ny,nx);
      }

      pTF[i]->lprFilter(2,0,120.,4.);
      pTF[i]->Forward(levelD-levelF);
      pTF[i]->setlow(fLow);
      pD[i]->white(60.,1,8.,20.);
      pTF[i]->lprFilter(2,0,120.,4.);
      pTF[i]->sethigh(fHigh);
      v[i] = pTF[i]->variability();
      pD[i]->bandPass();                   // band pass filtering
            
      cout<<"After "<<ifo[i]<<" data conditioning"<<endl; 
      gSystem->Exec("date");
    }

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

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// start of the coherent search
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    double Ao;

    for(i=levelD; i>=l_low; i--) {  // loop over TF resolutions
	  
      if(i<=l_high) {
	    	    	    
        NET.setDelayFilters(&Do[i],mTau+0.002);
	    
	Ao = NET.threshold(bpp,dTau);
	NET.set2or(x2or*Ao*Ao);
	cout<<"pixel threshold in units of noise rms: "<<Ao<<endl;
	cout<<"2 OR  threshold in units of noise var: "<<x2or*Ao*Ao<<endl;
	
	if(nIFO==2) m = NET.coherence2(Ao);
	if(nIFO==3) m = NET.coherence3(Ao);
	if(nIFO==4) m = NET.coherence4(Ao);
	if(nIFO==5) m = NET.coherence5(Ao);
	cout<<"total    pixels: "<<m<<"  ";
	    
	n = size_t(2.*Tgap*pD[0]->getTFmap()->resolution(0)+0.1);
	m = size_t(Fgap/pD[0]->getTFmap()->resolution(0)+0.0001);
	    
	cout<<"clusters: "<<NET.cluster(n,m)<<"  ";
	if(nIFO==2) m = NET.likelihood2('E',false,Acore);
	if(nIFO==3) m = NET.likelihood3('E',false,Acore);
	if(nIFO==4) m = NET.likelihood4('E',false,Acore);
	if(nIFO==5) m = NET.likelihood5('E',false,Acore);
	cout<<"selected pixels: "<<m<<"\n";

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

    NET.optim = false;
    for(i=l_low; i<=l_high; i++) {
      NET.setDelayFilters(&Do[i],mTau+0.002);
      if(nIFO==2) m = NET.likelihood2('h',false,Acore);
      if(nIFO==3) m = NET.likelihood3('h',false,Acore);
      if(nIFO==4) m = NET.likelihood4('h',false,Acore);
      if(nIFO==5) m = NET.likelihood5('h',false,Acore);
      cout<<"correlated energy for level "<<i<<": "<<m<<"\n";
      if(i<l_high) NET.Forward(1);
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
    gSystem->Exec("date");

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
      for(i=0; i<nIFO; i++) {
	wavevar.output(var_tree,&v[i],i+1,waveoffset);
	noiserms.output(noise_tree,&pD[i].nRMS,i+1,R/2);
      }
    }

    froot->Write();
    froot->Close();

    sprintf(command,"mv %s %s", tmpFile, outFile);
    gSystem->Exec(command);
    sprintf(command,"cp %s %s",outFile,output_dir);
    gSystem->Exec(command);


  }

// clean-up temporary data files

  for(j=0; j<3; j++) {
    sprintf(file,"%s/%s_%d.dat",nodedir,ifo[j],runID);
    sprintf(command,"rm %s",file);
    gSystem->Exec(command);
    if(simulation) {
      sprintf(file,"%s/mdc%s_%d.dat",nodedir,ifo[j],runID);
      sprintf(command,"rm %s",file);
      gSystem->Exec(command);
    }
  }
  cout<<"Stopping the job "<<runID<<endl;
  gSystem->Exec("date");
  
  return 0;

}







