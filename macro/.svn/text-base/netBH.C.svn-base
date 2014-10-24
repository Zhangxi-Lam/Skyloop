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

  netcluster wc;           
  netcluster* pwc;

  wavearray<double> x;         // temporary time series
  WSeries<double> wM;          // mdc WSeries
  WSeries<float>   v[nIFO];    // noise variability
  detector*       pD[nIFO];    // poiners to detectors
  WSeries<double>* pTF[nIFO];  // pointer to WSeries

  for(i=0; i<nIFO; i++) pD[i] = new detector(ifo[i]);

  network         NET;         // network

  for(i=0; i<nIFO; i++) NET.add(pD[i]); 
  NET.setSkyMaps(angle,Theta1,Theta2,Phi1,Phi2);
  NET.setIndex(pD[1]);
  NET.setAntenna(); 
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

  char file[512], tdf00[512], tdf90[512], buFFer[1024];

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

  n = int(dT/step);
  if(n<lags) lags = n;     // correct number of lags 

  netcluster WC[lags];     // array of cluster structures
  NET.setTimeShifts(lags,step);
  cout<<"                            lag step: "<<step<<endl;
  cout<<"                 number of time lags: "<<delta<<endl<<endl;

  if(simulation) {         // reag MDC log file
    i=NET.readMDClog(injectionList,double(long(Tb)));
    printf("GPS: %16.6f saved,  injections: %d\n",double(long(Tb)),i);
  } 
  else {                   // setup constant shifts
    for(i=0; i<nIFO; i++) pD[i].shift(shift[i]);
  }

  if(strlen(finalSegmentList)>1) NET.readSEGlist(finalSegmentList,2);

// read and dump data on local disk (nodedir)

  for(i=0; i<nIFO; i++) {
    sprintf(file,"%s/%s%d",input_dir,fileNamesRaw[i],runID);
    readframes1(file,channelNamesRaw[i],x);
    sprintf(file,"%s/%s_%d_%s_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID);
    pTF[i] = pD[i]->getTFmap();
    pTF[i]->Forward(x,B,levelR);
    pTF[i]->getLayer(x,0);
    pTF[i]->Forward(x,S,levelD);
    pTF[i]->DumpBinary(file);

    fprintf(stdout,"start=%f duration=%f rate=%f\n",
	    x.start(),x.size()/x.rate(),x.rate());
    if(i>0 && pTF[0]->start() != x.start()) exit(1);
    if(i>0 && pTF[0]->rate()  != x.rate())  exit(1);
 
    if(simulation) {
      sprintf(file,"%s/%s%d",mdc_dir,input_label,runID);
      readframes1(file,channelNamesMDC[i],x);
      sprintf(file,"%s/mdc%s_%d_%s_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID);
      wM.Forward(x,B,levelR); 
      wM.getLayer(x,0); 
      wM.Forward(x,S,levelD);
      wM.DumpBinary(file);

      fprintf(stdout,"start=%f duration=%f rate=%f\n",
	      x.start(),x.size()/x.rate(),x.rate());
      if(i>0 && wM.start() != x.start()) exit(1);
      if(i>0 && wM.rate() != x.rate())   exit(1);
    }
  }

  double R = x.rate();          // data rate
  x.resize(0);

  char outFile[1024];
  char tmpFile[1024];
  char endFile[1024];
  char command[1024];
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

    for(j=0; j<NET.nLag; j++) WC[j].clear();
    gSystem->Exec("date"); GetProcInfo();

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// data conditioning
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    int nx, ny, nn;

    for(i=0; i<nIFO; i++) {
      sprintf(file,"%s/%s_%d_%s_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID);
      pTF[i]->setLevel(levelD);
      pTF[i]->ReadBinary(file);
      n = pTF[i]->size();
      
      if(simulation) {
	sprintf(file,"%s/mdc%s_%d_%s_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID);
	wM.setLevel(levelD);
	wM.ReadBinary(file); wM*=factor;
	m  = wM.size();
	nx = (wM.start()-pTF[i]->start())*wM.rate();
	nn = m-n + int((wM.start()-pTF[i]->start())*wM.rate());
	ny = nx<0 ? -nx :  0;
	nx = nx<0 ?   0 : nx;
	nn = nn<0 ? m-ny : n-nx;
	pTF[i]->add(wM,nn,ny,nx);
	wM*=1./factor;
      }

      pTF[i]->lprFilter(2,0,120.,4.);
      pTF[i]->setlow(fLow);
      pD[i]->white(60.,1,8.,20.);
//      pD[i]->setsim(wM,NET.getmdcTime());   // use for BH-BH search
      pTF[i]->Inverse(levelD-levelF);
      pTF[i]->lprFilter(2,0,120.,4.);
      pTF[i]->Forward(levelD-levelF);
      pTF[i]->sethigh(fHigh);
      v[i] = pTF[i]->variability();
      pD[i]->bandPass();                // band pass filtering
            
      cout<<"After "<<ifo[i]<<" data conditioning"<<endl; 
      gSystem->Exec("date"); GetProcInfo();
    }

    cout<<"live time: "<<NET.setVeto(gap)<<endl;  // set veto array 

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
	    	    	    

	sprintf(tdf00,"%s/Meyer1024_L%1d.dat",data_dir,i);
	sprintf(tdf90,"/archive/home/klimenko/wat/wat-4.8.0/plots/Meyer1024-90_L%1d.dat",i);
	NET.setDelayFilters(tdf00,mTau+0.002,tdf90);

	Ao = NET.threshold(bpp,dTau);
	NET.set2or(x2or*Ao*Ao);
	cout<<"pixel threshold in units of noise rms: "<<Ao<<endl;
	cout<<"2 OR  threshold in units of noise var: "<<x2or*Ao*Ao<<endl;
	
	cout<<"total    pixels: "<<NET.coherence(Ao)<<"  ";
	    
	n = size_t(2.*Tgap*pD[0]->getTFmap()->resolution(0)+0.1);
	m = size_t(Fgap/pD[0]->getTFmap()->resolution(0)+0.0001);
	    
	cout<<"clusters: "<<NET.cluster(n,m)<<"  ";
	cout<<"selected pixels: "<<NET.likelihood('E',false,Acore)<<"\n";

	for(j=0; j<NET.nLag; j++) {
	  wc = *(NET.getwc(j));
	  cout<<wc.csize()<<"|"<<wc.size()<<"|"<<WC[j].append(wc)<<" "; 
	}
	cout<<endl;
      }
	  
      if(i>l_low) NET.Inverse(1); 
      gSystem->Exec("date"); GetProcInfo();
	  
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
    gSystem->Exec("date"); GetProcInfo();


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// likelihood
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for(i=l_low; i<=l_high; i++) {
      sprintf(tdf00,"%s/Meyer1024_L%1d.dat",data_dir,i);
      sprintf(tdf90,"/archive/home/klimenko/wat/wat-4.8.0/plots/Meyer1024-90_L%1d.dat",i);
      NET.setDelayFilters(tdf00,mTau+0.002,tdf90);
      cout<<"correlated energy for level "<<i<<": "<<NET.likelihoodE('i',false,Acore)<<"\n";
      cout<<"rejected weak pixels: "<<NET.netcut(eCOR,'x',0,1)<<"\n";    // remove weak glitches
      cout<<"rejected loud pixels: "<<NET.netcut(netCOR,'c',0,1)<<"\n";  // remove loud glitches
      cout<<"reconstructed events: "<<NET.events()<<"\n";
      if(i<l_high) NET.Forward(1);
    }

    gSystem->Exec("date");
    gSystem->Exec("date"); GetProcInfo();
    cout<<"\nSearch done\n";

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
//	wavevar.output(var_tree,&v[i],i+1,waveoffset);
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

  for(i=0; i<nIFO; i++) {
    sprintf(file,"%s/%s_%d_%s_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID);
    sprintf(command,"rm %s",file);
    gSystem->Exec(command);

    if(simulation) {
      sprintf(file,"%s/mdc%s_%d_%s_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID);
      sprintf(command,"rm %s",file);
      gSystem->Exec(command);
    }
  }
  cout<<"Stopping the job "<<runID<<endl;
  gSystem->Exec("date");
  
  return 0;

}







