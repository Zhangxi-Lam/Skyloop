// Coherent WaveBurst production script for 3 detector network
{
  int  i,j,n,m;

  if(search=='E' || search=='E') cout<<"\n un-modeled search (Energy): "<<search<<endl;
  if(search=='b' || search=='B') cout<<"\n un-modeled search (single stream): "<<search<<endl;
  if(search=='r' || search=='R') cout<<"\n un-modeled search (dual stream): "<<search<<endl;
  if(search=='i' || search=='I') cout<<"\n elliptical polarisation: "<<search<<endl;
  if(search=='g' || search=='G') cout<<"\n circular polarisation: "<<search<<endl;
  if(search=='s' || search=='S') cout<<"\n linear polarisation: "<<search<<endl;

  cout<<" network of ";
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
  NET.setAntenna(); 
  NET.constraint(delta,gamma);
  NET.setDelay(refIFO);
  NET.Edge = waveoffset;
  NET.netCC = netCC;
  NET.netRHO = netRHO;
  NET.EFEC = EFEC;
  NET.precision = precision;

  double mTau=NET.getDelay("MAX");  // maximum time delay 
  double dTau=NET.getDelay();       // time delay difference 
  cout<<"maximum time delay between detectors: "<<mTau<<endl;
  cout<<"       maximum time delay difference: "<<dTau<<endl;
  cout<<"                  skymap search mode: "<<mode<<endl;
  cout<<"           skymap angular resolution: "<<angle<<endl;
  cout<<"          skymap size in polar angle: "<<Theta1<<", "<<Theta2<<endl;
  cout<<"      skymap size in azimuthal angle: "<<Phi1<<", "<<Phi2<<endl;
  cout<<"                    netRHO and netCC: "<<netRHO<<", "<<netCC<<endl;
  cout<<"              regulator delta, local: "<<delta<<" "<<NET.local<<endl<<endl;

  return;

  injection mdc(nIFO);
  livetime live;
  netevent netburst(nIFO);
  variability wavevar;
  wavenoise noiserms;

  gSystem->Exec("date");
  gSystem->Exec("hostname");
  gRandom->SetSeed(0);

  // parse input

  if(!runID) cin>>runID>>input_dir>>input_label>>output_dir>>output_label;
  cout<<"job ID: "<<runID<<endl;
  cout<<"Input : "<<input_dir<<"\n  label: "<<input_label<<endl;
  cout<<"Output: "<<output_dir<<"\n  label: "<<output_label<<endl;
  NET.setRunID(runID);

  // read input framelist file

  char file[512], tdf00[512], tdf90[512], buFFer[1024];
  int rnID = int(gRandom->Rndm(13)*1.e9);   // random name ID

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

  if(simulation) {                     // reag MDC log file
    i=NET.readMDClog(injectionList,double(long(Tb)));
    printf("GPS: %16.6f saved,  injections: %d\n",double(long(Tb)),i);
  } 

  if(mask>0.) NET.setSkyMask(mask,skyMaskFile);
  if(strlen(finalSegmentList)>1) NET.readSEGlist(finalSegmentList,2);

// read and dump data on local disk (nodedir)

  for(i=0; i<nIFO; i++) {
    sprintf(file,"%s/%s%d",input_dir,fileNamesRaw[i],runID);
    readframes1(file,channelNamesRaw[i],x);
    sprintf(file,"%s/%s_%d_%s_%d_%d.dat",
	    nodedir,ifo[i],int(Tb),output_label,runID,rnID);
    if(dcCal[i]>0.) x*=dcCal[i];               // DC correction
    pTF[i] = pD[i]->getTFmap();
    pTF[i]->Forward(x,B,levelR);
    pTF[i]->getLayer(x,0);
    pTF[i]->Forward(x,S,levelD);
    pTF[i]->DumpBinary(file);
    n = pTF[i]->size();

    fprintf(stdout,"start=%f duration=%f rate=%f\n",
	    x.start(),x.size()/x.rate(),x.rate());
    if(i>0 && pTF[0]->start() != x.start()) exit(1);
    if(i>0 && pTF[0]->rate()  != x.rate())  exit(1);
 
    if(simulation) {
      sprintf(file,"%s/%s%d",input_dir,input_label,runID);
      readframes1(file,channelNamesMDC[i],x);
      sprintf(file,"%s/mdc%s_%d_%s_%d_%d.dat",
	      nodedir,ifo[i],int(Tb),output_label,runID,rnID);
      cout<<file<<endl;
      wM.Forward(x,B,levelR); 
      wM.getLayer(x,0); 
      wM.Forward(x,S,levelD);
      wM.DumpBinary(file);
      m = wM.size();

      fprintf(stdout,"start=%f duration=%f rate=%f\n",
	      x.start(),x.size()/x.rate(),x.rate());
      if(i>0 && wM.start() != x.start()) exit(1);
      if(i>0 && wM.rate() != x.rate())   exit(1);
      if(n != m)   exit(1);
    }
  }

  size_t lags = 0;
  double factor = 1.0;          // strain factor
  double R = x.rate();          // data rate
  x.resize(0);

  char tmpFile[1024];
  char outFile[1024];
  char endFile[1024];
  char outDump[1024];
  char endDump[1024];
  char out_CED[1024];
  char end_CED[1024];
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
      sprintf(outDump,"%s/wave_%d_%d_%s_%g_id%d.txt",
	      nodedir,int(Tb),int(dT),output_label,factor,runID);
      sprintf(endDump,"%s/wave_%d_%d_%s_%g_id%d.txt",
	      output_dir,int(Tb),int(dT),output_label,factor,runID);
      sprintf(out_CED,"%s/ced_%d_%d_%s_%g_id%d",
	      nodedir,int(Tb),int(dT),output_label,factor,runID);
      sprintf(end_CED,"%s/ced_%d_%d_%s_%g_id%d",
	      output_dir,int(Tb),int(dT),output_label,factor,runID);

      if(!gSystem->GetPathInfo(endFile,fstemp)) {
	printf("The file %s already exists - skip\n",endFile);
	fflush(stdout);
	TFile rf(endFile); 
	if(!rf.IsZombie()) continue;
      }

    }
    else {
      sprintf(outFile,"%s/wave_%d_%d_%s_%d_id%d.root",
	      nodedir,int(Tb),int(dT),output_label,lagOff,runID);
      sprintf(endFile,"%s/wave_%d_%d_%s_%d_id%d.root",
	      output_dir,int(Tb),int(dT),output_label,lagOff,runID);
      sprintf(tmpFile,"%s/wave_%d_%d_%s_%d_id%d.root.tmp",
	      nodedir,int(Tb),int(dT),output_label,lagOff,runID);
      sprintf(outDump,"%s/wave_%d_%d_%s_%d_id%d.txt",
	      nodedir,int(Tb),int(dT),output_label,lagOff,runID);
      sprintf(endDump,"%s/wave_%d_%d_%s_%d_id%d.txt",
	      output_dir,int(Tb),int(dT),output_label,lagOff,runID);
      sprintf(out_CED,"%s/ced_%d_%d_%s_%d_id%d",
	      nodedir,int(Tb),int(dT),output_label,lagOff,runID);
      sprintf(end_CED,"%s/ced_%d_%d_%s_%d_id%d",
	      output_dir,int(Tb),int(dT),output_label,lagOff,runID);
    }

    cout<<"output file on the node: "<<outFile<<endl;
    cout<<"final output file name : "<<endFile<<endl;
    cout<<"temporary output file  : "<<tmpFile<<endl;

    gSystem->Exec("date"); GetProcInfo();

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// data conditioning
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for(i=0; i<nIFO; i++) {
      sprintf(file,"%s/%s_%d_%s_%d_%d.dat",
	      nodedir,ifo[i],int(Tb),output_label,runID,rnID);
      pTF[i]->setLevel(levelD);
      pTF[i]->ReadBinary(file);
      
      if(simulation) {
	sprintf(file,"%s/mdc%s_%d_%s_%d_%d.dat",
		nodedir,ifo[i],int(Tb),output_label,runID,rnID);
	wM.setLevel(levelD);
	wM.ReadBinary(file); wM*=factor;
	pTF[i]->add(wM);
	wM*=1./factor;
      }

      pTF[i]->lprFilter(2,0,Tlpr,4.);
      pTF[i]->setlow(fLow);
      pD[i]->white(60.,1,8.,20.);
//      pD[i]->setsim(wM,NET.getmdcTime(),10.,8.);   // use for BH-BH search
      pTF[i]->Inverse(levelD-levelF);
      pTF[i]->lprFilter(2,0,Tlpr,4.);
      pTF[i]->Forward(levelD-levelF);
      pTF[i]->sethigh(fHigh);
      v[i] = pTF[i]->variability();
      pD[i]->bandPass();                // band pass filtering
            
      cout<<"After "<<ifo[i]<<" data conditioning"<<endl; 
      gSystem->Exec("date"); GetProcInfo();
    }

    if(!simulation) {                     // setup lags
      lags = NET.setTimeShifts(lagSize,lagStep,lagOff,lagMax,lagFile,lagMode,lagSite);
      cout<<"lag step: "<<lagStep<<endl;
      cout<<"number of time lags: "<<lags<<endl;
    }
    else if(!lags) lags = NET.setTimeShifts();

    double TL = NET.setVeto(gap);
    cout<<"live time in zero lag: "<<TL<<endl<<endl;  // set veto array 
    if(TL <= 0.) exit(1);                             // exit if live time is zero

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// initialization of the output files 
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

    //if(dump) netburst.dopen(outDump,"w");  //GV

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// start of the coherent search
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    double Ao;
    wavearray<int> np(NET.nLag);      // pixel counter
    bool append = false;              // file append for savemode
    int ceddir = 0;                   // flag if ced directory exists

    cout<<"Start coherent search: "; gSystem->Exec("date");

    for(i=levelD; i>=l_low; i--) {  // loop over TF resolutions
	  
      if(i<=l_high) {
	    	    	    
	sprintf(tdf00,"%s/Meyer1024wat482_00_L%1d.dat",data_dir,i);
	NET.setDelayFilters(tdf00);
	if(i==l_high) {
	  NET.setDelayIndex();
	  NET.setIndexMode(1);
	}

	Ao = NET.threshold(bpp,dTau);
	NET.set2or(x2or*Ao*Ao);
	cout<<"pixel threshold in units of noise rms: "<<Ao<<endl;
	cout<<"2 OR  threshold in units of noise var: "<<x2or*Ao*Ao<<endl;
	
	cout<<"total    pixels: "<<NET.coherence(Ao)<<"  ";
	    
	n = size_t(2.*Tgap*pD[0]->getTFmap()->resolution(0)+0.1);
	m = size_t(Fgap/pD[0]->getTFmap()->resolution(0)+0.0001);
	    
	cout<<"clusters: "<<NET.cluster(n,m)<<"  ";
	cout<<"selected pixels: "<<NET.likelihood('E',Acore)<<"\n";

	for(j=0; j<NET.nLag; j++) {
	  sprintf(file,"%s/pix_%d_%s_%d_%d_%d.lag",
		  nodedir,int(Tb),output_label,int(j),iii,rnID);
	  wc = *(NET.getwc(j));
	  if(!append) np.data[j]  = wc.write(file,0);
	  else        np.data[j] += wc.write(file,1);
	  cout<<wc.csize()<<"|"<<wc.size()<<"|"<<np.data[j]<<" "; 
	}
	cout<<endl;
	append = true;                
      }
	  
      if(i>l_low) NET.Inverse(1); 
      gSystem->Exec("date"); GetProcInfo();
	  
    }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// supercluster analysis
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for(j=0; j<lags; j++){
      sprintf(file,"%s/pix_%d_%s_%d_%d_%d.lag",
	      nodedir,int(Tb),output_label,int(j),iii,rnID);
      wc.read(file);
      m = wc.supercluster('L',NET.e2or,true);
      pwc = NET.getwc(j); pwc->cpf(wc,true);
      cout<<m<<"|"<<pwc->size()<<" ";
      sprintf(command,"rm %s",file);
      gSystem->Exec(command);
    }
    cout<<endl;
    cout<<"events in the buffer: "<<NET.events()<<"\n";
    gSystem->Exec("date"); GetProcInfo();


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// likelihood
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for(i=l_low; i<=l_high; i++) {
      sprintf(tdf00,"%s/Meyer1024wat482_00%s_L%1d.dat",data_dir,filter,i);
      sprintf(tdf90,"%s/Meyer1024wat482_90%s_L%1d.dat",data_dir,filter,i);
      NET.setDelayFilters(tdf00,tdf90);

      if(i==l_low) {
	NET.setDelayIndex();
	NET.setIndexMode(mode);
      }

      cout<<"selected core pixels: "<<NET.likelihood(search,Acore)<<" for level "<<i<<"\n";
      cout<<"rejected weak pixels: "<<NET.netcut(netRHO,'r',0,1)<<"\n";  // remove weak glitches
      cout<<"rejected loud pixels: "<<NET.netcut(netCC,'c',0,1)<<"\n";   // remove loud glitches
      cout<<"events in the buffer: "<<NET.events()<<"\n";

      if(cedDump) {
	cout<<"dump ced into "<<out_CED<<"\n";
	//if(CED(&NET,out_CED,cedRHO)) ceddir = 1; 
        if(netburst.ced(&NET,out_CED,cedRHO,factor)) ceddir = 1; 
      }

      if(i<l_high) NET.Forward(1);
    }

    gSystem->Exec("date");
    gSystem->Exec("date"); GetProcInfo();
    cout<<"\nSearch done\n";
    cout<<"reconstructed events: "<<NET.events()<<"\n";

    if(simulation) NET.printwc(0);

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// save data in root file
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    if(dump) netburst.dopen(outDump,"w");

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
    if(dump) netburst.dclose();

    sprintf(command,"mv %s %s", tmpFile, outFile);
    gSystem->Exec(command);
    sprintf(command,"cp %s %s",outFile,endFile);
    gSystem->Exec(command);
    sprintf(command,"cp %s %s",outDump,endDump);
    if(dump) gSystem->Exec(command);
    sprintf(command,"mv %s %s",out_CED,end_CED);
    if(cedDump && ceddir) gSystem->Exec(command);

  }

// clean-up temporary data files

  for(i=0; i<nIFO; i++) {
    sprintf(file,"%s/%s_%d_%s_%d_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID,rnID);
    sprintf(command,"rm %s",file);
    gSystem->Exec(command);

    if(simulation) {
      sprintf(file,"%s/mdc%s_%d_%s_%d_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID,rnID);
      sprintf(command,"rm %s",file);
      gSystem->Exec(command);
    }
  }
  cout<<"Stopping the job "<<runID<<endl;
  gSystem->Exec("date");
  
  return 0;

}







