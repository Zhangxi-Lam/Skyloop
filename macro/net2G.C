// 2G coherent WaveBurst production script
// S.Klimenko, University of Florida, June 15, 2012
{
   int  i,j,n,m,k;

  gSystem->Exec("date");

  if(search=='E' || search=='E') cout<<"\n un-modeled search (Energy): "<<search<<endl;
  if(search=='b' || search=='B') cout<<"\n un-modeled search (single stream): "<<search<<endl;
  if(search=='r' || search=='R') cout<<"\n un-modeled search (dual stream): "<<search<<endl;
  if(search=='i' || search=='I') cout<<"\n elliptical polarisation: "<<search<<endl;
  if(search=='g' || search=='G') cout<<"\n circular polarisation: "<<search<<endl;
  if(search=='s' || search=='S') cout<<"\n linear polarisation: "<<search<<endl;

  cout<<" network of ";
  for(i=0; i<nIFO; i++) cout<<ifo[i]<<" ";
  cout<<" detectors\n\n";

  // network initialization

  detector*           pD[nIFO];  // poiners to detectors
  WSeries<double>*   pTF[nIFO];  // pointer to WSeries
  wavearray<double>* hot[nIFO];  // temporary time series
  network            NET;        // network

  for(i=0; i<nIFO; i++)  pD[i] = new detector(ifo[i]);
  for(i=0; i<nIFO; i++) NET.add(pD[i]); 

  if(healpix) NET.setSkyMaps(4);
  else        NET.setSkyMaps(angle,Theta1,Theta2,Phi1,Phi2);

  NET.setAntenna(); 
  NET.constraint(delta,gamma);
  NET.setDelay(refIFO);
  NET.Edge = waveoffset;
  NET.netCC = netCC;
  NET.netRHO = netRHO;
  NET.EFEC = EFEC;
  NET.precision = precision;
  NET.optim = optim;
  if(l_high==9)     NET.setMRAcatalog("OverlapCatalog5.bin");
  else if(l_low==3) NET.setMRAcatalog("OverlapCatalog8-1024.bin");
  else              NET.setMRAcatalog("OverlapCatalog16-1024.bin");
  NET.setAcore(Acore);         

  double mTau=NET.getDelay("MAX");  // maximum time delay 
  double dTau=NET.getDelay();       // time delay difference 
  cout<<"maximum time delay between detectors: "<<mTau<<endl;
  cout<<"       maximum time delay difference: "<<dTau<<endl;
  cout<<"                  skymap search mode: "<<mode<<endl;
  cout<<"           skymap angular resolution: "<<angle<<endl;
  cout<<"          skymap size in polar angle: "<<Theta1<<", "<<Theta2<<endl;
  cout<<"      skymap size in azimuthal angle: "<<Phi1<<", "<<Phi2<<endl;
  cout<<"                    netRHO and netCC: "<<netRHO<<", "<<netCC<<endl;
  cout<<"                       acor and e2or: "<<NET.acor<<", "<<NET.e2or<<endl;
  cout<<"              regulator delta, gamma: "<<delta<<" "<<gamma<<endl<<endl;

  injection mdc(nIFO);
  livetime live;
  netevent netburst(nIFO);
  variability wavevar;
  wavenoise noiserms;

  // time_frequency initialization

  int nRES = l_high-l_low+1;     // number of frequency resolution levels
  Meyer<double> B(1024);         // set wavelet for resampling
  WDM<double>* pwdm[nRES];       // wavelet pointers: wdm[0] - l_high, wdm[nRES-1] l_low 
  WDM<double>* pWDM[nRES];       // wavelet pointers: WDM[0] - l_high, WDM[nRES-1] l_low 
  WDM<double> WDH(1024*2,1024*2,6,10);
  WDM<double> WD2(1024,1024*2,6,10);
  WDM<double> WD32(64,64,6,10);
  WDM<double> WD4(1024/2,1024,6,10);

  for(i=l_high; i>=l_low; i--) {
     pwdm[l_high-i] = new WDM<double>(1<<i,1<<i,6,10);
     NET.add(pwdm[l_high-i]);
  }

  gSystem->Exec("date");
  gSystem->Exec("hostname");
  gRandom->SetSeed(0);

  // parse standard input

  runID = 1;
  if(!runID) cin>>runID>>input_dir>>input_label>>output_dir>>output_label;
  cout<<"job ID: "<<runID<<endl;
  cout<<"Input : "<<input_dir<<"\n label: "<<input_label<<endl;
  cout<<"Output: "<<output_dir<<"\n label: "<<output_label<<endl;
  NET.setRunID(runID);

  // read input segment and injection lists

  char file[512], tdf00[512], tdf90[512], buFFer[1024];
  int rnID = int(gRandom->Rndm(13)*1.e9);   // random name ID
  rnID = 3;

  double Tb=942450900.;                     // hardcoded test setup
  double dT=616.;                           // hardcoded test setup
  double Te=Tb+dT;

  if(simulation) {                          // reag MDC log file
     //    i=NET.readMDClog(injectionList,Tb);
     //    printf("GPS: %16.6f saved,  injections: %d\n",Tb),i);
  } 

  //  if(mask>0.) NET.setSkyMask(mask,skyMaskFile);
  //  if(strlen(finalSegmentList)>1) NET.readSEGlist(finalSegmentList,2);

// read and dump data on local disk (nodedir)

  wavearray<double> x;           // temporary time series
  wavearray<double> y;           // temporary time series

  x.start(Tb);
  x.stop(Te);
  x.edge(waveoffset);
  x.resize(1024*16*616);
  x.rate(1024*16);

  for(i=0; i<nIFO; i++) {

     sprintf(file,"%s/%s",input_dir,fileNamesRaw[i]);

     CWB::frame FRAME(file,channelNamesRaw[i]);
     FRAME >> x;

     //     x=0.; addGauss(x,1);

     sprintf(file,"%s/%s_%d_%s_%d_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID,rnID);

     pTF[i] = pD[i]->getTFmap();
     hot[i] = pD[i]->getHoT();
     pTF[i]->Forward(x,B,levelR);
     pTF[i]->getLayer(y,0);
     //y=x;
     y->DumpBinary(file);
     n = pTF[i]->size();

     fprintf(stdout,"start=%f duration=%f rate=%f\n",
             x.start(),x.size()/x.rate(),y.rate());

     if(simulation) {
        sprintf(file,"%s/%s%d",input_dir,input_label,runID);
        CWB::frame FRAME(file,channelNamesMDC[i]);
        FRAME >> x;
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

    gSystem->Exec("date"); // GetProcInfo();

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// data conditioning
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    wavearray<double> htmp;
    WSeries<double> wtmp;
    WSeries<double> utmp;
    WSeries<double> ntmp;

    for(i=0; i<nIFO; i++) {
      sprintf(file,"%s/%s_%d_%s_%d_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID,rnID);
      hot[i]->ReadBinary(file);
      //hot[i]->rate(R);
      hot[i]->rate(R/4);
      // cout<<hot[i]->size()<<" "<<hot[i]->rate()<<endl;

      if(simulation) {
	sprintf(file,"%s/mdc%s_%d_%s_%d_%d.dat",nodedir,ifo[i],int(Tb),output_label,runID,rnID);
	y.ReadBinary(file); y*=factor;
	hot[i]->add(y);
	*hot[i] *= 1./factor;
      }

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++
// REGRESSION stage should be included here
//
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++

      //pTF[i]->Forward(*hot[i],WD2);
      //pD[i]->white(60.,0,NET.Edge,5.);
      //pD[i]->nRMS.bandpass(8.1,0.,1.);              // high pass filtering     
     
      pTF[i]->Forward(*hot[i],WD2);
      regression rr(*pTF[i],"target",32,fHigh);
      rr.add(*hot[i],"witness");   
      rr.setFilter(8);
      rr.setMatrix(NET.Edge,0.95);
      rr.solve(0.,4,'h');
      rr.apply(0.7,'m');
      //htmp = rr.getClean();
      htmp = rr.getNoise();
      //return;
      rr.add(htmp,"witness",0.,10.);
      rr.add(0,2,"bi-witness");
      rr.mask(1);
      rr.setFilter(8);
      rr.setMatrix(NET.Edge,0.95);
      rr.solve(0.,8,'h');
      rr.apply(0.3,'a');
      return;
      continue;
      *hot[i] = rr.getClean();
      
      /*     
      pTF[i]->Forward(*hot[i],WD2);
      pD[i]->white(60.,0,NET.Edge,5.);
      pTF[i]->white(pD[i]->nRMS,1);
      pTF[i]->white(pD[i]->nRMS,-1);
      wtmp = *pTF[i]; utmp = *pTF[i];
      wtmp.Inverse(); utmp.Inverse(-2);
      wtmp += utmp; wtmp *= 0.5;
      ntmp = pD[i]->nRMS;
      pTF[i]->Forward(wtmp,WD32);
      pD[i]->white(10.,0,NET.Edge,5.);
      ntmp.mul(pD[i]->nRMS);
      pD[i]->nRMS = ntmp;
      pTF[i]->Forward(*hot[i],WD2);
      pTF[i]->setlow(fLow);
      pTF[i]->white(pD[i]->nRMS,1);
      pTF[i]->white(pD[i]->nRMS,-1);
      pTF[i]->sethigh(fHigh);
      pD[i]->bandPass(16.,fHigh);              // band pass filtering     
      wtmp = *pTF[i];
      pTF[i]->Inverse();
      wtmp.Inverse(-2);
      *hot[i] = *pTF[i];
      *hot[i] += wtmp;
      *hot[i] *= 0.5;
      */     
     //return;      
     
      pTF[i]->Forward(*hot[i],WD2);
      //pD[i]->white(60.,0,NET.Edge,5.);
      //pD[i]->nRMS.bandpass(8.1,0.,1.);              // high pass filtering     
      pTF[i]->white(pD[i]->nRMS,1);
      pTF[i]->white(pD[i]->nRMS,-1);
      pTF[i]->setlow(fLow);
      pTF[i]->sethigh(fHigh);
      wtmp = *pTF[i];
      pTF[i]->Inverse();
      wtmp.Inverse(-2);
      *hot[i] = *pTF[i];
      *hot[i] += wtmp;
      *hot[i] *= 0.5;
      //return;
      
//      pD[i]->setsim(wM,NET.getmdcTime(),10.,8.);   // use for BH-BH search
            
//      return;

      cout<<"After "<<ifo[i]<<" data conditioning"<<endl; 
      gSystem->Exec("date"); // GetProcInfo();
    }

    return;

    if(!simulation) {                     // setup lags
      lags = NET.setTimeShifts(lagSize,lagStep,lagOff,lagMax,lagFile,lagMode,lagSite);
      cout<<"lag step: "<<lagStep<<endl;
      cout<<"number of time lags: "<<lags<<endl;
    }
    else if(!lags) lags = NET.setTimeShifts();

    double TL = NET.setVeto(gap);
    cout<<"live time in zero lag: "<<TL<<endl<<endl;  // set veto array 
    if(TL <= 0.) exit(1);                             // exit if live time is zero

    //    return;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// initialization of the output files 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    TFile *froot = new TFile(tmpFile, "RECREATE");
    TTree* net_tree = netburst.setTree();

//    TTree* live_tree= live.setTree();

    if(simulation) {
      TTree* mdc_tree = mdc.setTree();
    }
    else {
//      TTree* var_tree = wavevar.setTree();
      TTree* noise_tree = noiserms.setTree();
    }

    //if(dump) netburst.dopen(outDump,"w");  //GV


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// NETPIXEL stage: start of the coherent search
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    netcluster* pwc;
    netcluster wc;

    int npix;
    double Eo, E2;
    bool append = false;              // file append for savemode
    wavearray<int> nP(NET.nLag);      // pixel counter
    wavearray<int> nC(NET.nLag);      // cluster counter
    TH1F* hic[nRES];
    for(j=0; j<nRES; j++) hic[j] = NULL;
    hic[0] = new TH1F(" c "," c ",1000,0.,50.);

    cout<<"Start coherent search: \n"; gSystem->Exec("date");
    //lagSize = 1;

    for(j=0; j<nRES; j++) {           // loop over TF resolutions
	  
       for(i=0; i<nIFO; i++) {        // produce TF maps with max over the sky energy
          NET.getifo(i)->getTFmap()->maxEnergy(*hot[i],*pwdm[j],mTau,4);
       }
       
       gSystem->Exec("date");         // GetProcInfo();
       Eo = NET.THRESHOLD(bpp);       // threshold on pixel energy
       cout<<"thresholds in units of noise variance: Eo="<<Eo<<" Em="<<Eo*2<<endl;
       
       for(k=0; k<lagSize; k++) {
       //      for(k=0; k<1; k++) {

          npix = NET.getNetworkPixels(k,Eo,Eo*2,hic[j]);
          cout<<"lag="<<k<<" pixels: "<<npix<<"  ";

          //n = size_t(2.*Tgap*pD[0]->getTFmap()->resolution(0)+0.1);
          //m = size_t(Fgap/pD[0]->getTFmap()->resolution(0)+0.0001);
          n=1; m=1;
	    
          cout<<"clusters: "<<NET.cluster(n,m)<<" ";
	  //cout<<NET.getwc(k)->select(2,Eo*2)<<" ";
	  wc.cpf(*NET.getwc(k));
          NET.getwc(k)->clear();
          //return;

	  sprintf(file,"%s/pix_%d_%s_%d_%d_%d.lag",
		  nodedir,int(Tb),output_label,int(k),iii,rnID);

	  if(!append) nP.data[k]  = wc.write(file,0);
	  else        nP.data[k] += wc.write(file,1);
          nC.data[k]+=wc.csize();
	  cout<<"stored: "<<nC.data[k]<<"|"<<nP.data[k]<<endl;
          
       }
       append = true;                
       gSystem->Exec("date"); // GetProcInfo();
     
    }	  	  
    //return;
    
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// SUPERCLUSTER stage
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    gSystem->Exec("date"); // GetProcInfo();

    FILE* fp;
    size_t count; 
    
    int nnn = 0;
    int mmm = 0;
    
    TH2F* hist[2];
    hist[0] = new TH2F(" 0 "," 0 ",200,-1.,1.01,200,-1.,1);           // subNetCut performance histogram
    hist[1] = new TH2F(" 1 "," 1 ",100,0.,1,100,0.,1);           // subNetCut performance histogram
    TH1F* his = new TH1F(" 3 "," 3 ",100,0.,4*512.);           // subNetCut performance histogram
    TH1F* super = new TH1F(" 4 "," 4 ",300,0.,30.);           // supercluster performance histogram
    
    for(j=0; j<nRES; j++) pwdm[j]->setTDFilter(12, 1);   // set low-rate TD filters
    NET.setDelayIndex(hot[0]->rate()); 
    
// set time-delay filters

//   for(j=0; j<nRES; j++) pwdm[j]->setTDFilter(12, 1);   // set low-rate TD filters
//   for(j=0; j<nRES; j++) pWDM[j]->setTDFilter(12, 4);   // set high-rate TD filters

   for(k=0; k<lagSize; k++){
//   for(k=0; k<1; k++){
      gSystem->Exec("date"); // GetProcInfo();
      sprintf(file,"%s/pix_%d_%s_%d_%d_%d.lag",
              nodedir,int(Tb),output_label,int(k),iii,rnID);
      
      wc.clear();
      wc.read(file);
      cout<<file<<endl;
      cout<<"process lag "<<k<<" "<<NET.e2or<<endl;
      cout<<"loaded clusters|pixels: "<<wc.csize()<<"|"<<wc.size()<<endl;
            
      m = wc.supercluster('L', NET.e2or, TFgap, false, super);
      cout<<"super  clusters|pixels: "<<wc.esize()<<"|"<<wc.psize()<<endl;
      
      pwc = NET.getwc(k);
      pwc->cpf(wc, false);            
      
      gSystem->Exec("date"); // GetProcInfo();

      for(i=0; i<nIFO; i++) pD[i]->sclear();

      for(j=0; j<nRES; j++) { // loop over TF resolutions
         for(i=0; i<nIFO; i++) {
            pTF[i]->Forward(*hot[i],*pwdm[j]);
            pD[i]->addSTFmap(pwc);
         }
      } 

      pwc->setcore(false);               // release all pixels

      gSystem->Exec("date"); // GetProcInfo();

      n = 0;
      while(1){

         //for(j=0; j<nRES; j++) {           // loop over TF resolutions
         //   for(i=0; i<nIFO; i++) pTF[i]->Forward(*hot[i],*pwdm[j]);
         //   count = pwc->loadTDamp(NET, 'a', 10000, 100);            
         //}

         count = pwc->loadTDampSSE(NET, 'a', 10000, 300);            
         n += NET.subNetCut(k,subnet,hist[0]);         
         cout<<"selected pixels: "<<n<<", fraction: "<<n/double(pwc->psize(1)+pwc->psize(-1))<< endl;
         if(count<10000) break;
      }

      nnn += pwc->psize(-1);
      mmm += pwc->psize(1)+pwc->psize(-1);

      cout<<"events in the buffer: "<<NET.events()<<"|"<<nnn<<"|"<<nnn/double(mmm)<<"\n";
      
      m = pwc->defragment(Tgap,Fgap);
      cout<<"duper  clusters|pixels: "<<NET.events()<<"|"<<pwc->psize()<<endl;

      pwc->setcore(false);
      pwc->setcuts();

      gSystem->Exec("date"); // GetProcInfo();

      while(1){

         count = pwc->loadTDampSSE(NET, 'a', 10000,9999);
         NET.likelihood2G(search,k,550,hist[1]);            
         if(count<10000) break;
      }

      // return;
   }
      //fclose(fp);   
   
/*
      fp = fopen(file,"rb");
      pwc->clear();
      count = pwc->read(fp,0);
      n = 0;
      while(count) {
         count = pwc->read(fp,1500);
         n += NET.likelihoodMRA('E',k,0.01,0.1,hist[1],4);         
//         n += NET.subNetCut(0.2,k,hist[1]);         
      }
      cout<<"selected pixels: "<<n<<", fraction: "<<n/double(pwc->psize(1)+pwc->psize(-1))<< endl;
      fclose(fp);   
*/
//      gSystem->Exec("date"); // GetProcInfo();
//   }

   return;
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// likelihood
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   if(healpix) NET.setSkyMaps(4);
   else        NET.setSkyMaps(angle,Theta1,Theta2,Phi1,Phi2);
   
   NET.setAntenna(); 
   NET.setDelay(refIFO);

   int nm;

   for(k=0; k<lagSize; k++){
      gSystem->Exec("date"); // GetProcInfo();
      for(j=0; j<nRES; j++) pwdm[j]->setTDFilter(12, 4);   // set low-rate TD filters
      NET.setDelayIndex(hot[0]->rate()*4); 

      sprintf(file,"%s/PIXX_%d_%s_%d_%d_%d.lag",
              nodedir,int(Tb),output_label,int(k),iii,rnID);
      fp = fopen(file,"rb");

      pwc = NET.getwc(k);
      pwc->clear();
      count = pwc->read(fp,0);
      n = 0;
      while(count) {
         count = pwc->read(fp,1500);
         nm = NET.likelihoodMRA('E',k,0.01,0.2,hist[1],4);  
         if(!nm) count=0;
         n += nm;         
//         n += NET.subNetCut(0.2,k,hist[1]);         
      }
      cout<<"selected pixels****: "<<n<<", fraction: "<<n/double(pwc->psize(1)+pwc->psize(-1))<< endl;
      fclose(fp);   

      gSystem->Exec("date"); // GetProcInfo();
   }


   return;

/*
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
*/
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// save data in root file
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//    if(dump) netburst.dopen(outDump,"w");

//    live.output(live_tree,&NET);
/*
    if(simulation) {
      netburst.output(net_tree,&NET,factor);
      mdc.output(mdc_tree,&NET,factor);
    }
    else {
      netburst.output(net_tree,&NET);
      for(i=0; i<nIFO; i++) {
	noiserms.output(noise_tree,&pD[i].nRMS,i+1,R/2);
      }
    }
    
    froot->Write();
    froot->Close();
*/
  }

/*
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
*/
  cout<<"Stopping the job "<<runID<<endl;
  gSystem->Exec("date");
  
  return 0;
}







