{
  double xcor,esnr,rsnr,null,etot,ecor,asnr,acor,a2or;
  double enrg[5];
  int j,n,m,k;
  size_t indx;

  wavearray<double>  sMDC;
  wavearray<double>   eff[32];
  wavearray<double>   err[32];
  wavearray<double>   e00[32];
  wavearray<double>  nMDC[32];
  wavearray<double>  nTRG[32];

  bool save;
  char fname[256];
  
  int ntrg = sim.GetEntries();
  int nmdc = mdc.GetEntries();

  cout<<"  total injections: "<<nmdc<<endl;
  
// histograms
  
  TH2F* HH[nIFO][ninj+1];
  TH1F* hT[ninj+1];
  TH1F* hF[ninj+1];
  TH1F* hG[ninj+1];
  
  for(int i=0; i<ninj+1; i++){
    hT[i] = new TH1F("hT","",250,-0.05,0.05);
    hT[i]->SetTitleOffset(1.3,"Y");
    hT[i]->GetXaxis()->SetTitle("detected-injected time, sec");
    hT[i]->GetYaxis()->SetTitle("events");
    sprintf(fname,iname[i]);
    hT[i]->SetTitle(fname);
    
    hF[i] = new TH1F("hF","",512,f_low[i],fhigh[i]);
    hF[i]->SetTitleOffset(1.3,"Y");
    if(i<ninj) hF[i]->GetXaxis()->SetTitle("frequency, Hz");
    else       hF[i]->GetXaxis()->SetTitle("detected-injected, Hz");
    hF[i]->GetYaxis()->SetTitle("events");
    sprintf(fname,iname[i]);
    hF[i]->SetTitle(fname);

    for(int j=0; j<nIFO; j++){
      HH[j][i] = new TH2F(ifoname[j],"",500,-23.,-18.,500,-23.,-18.);
      HH[j][i]->SetTitleOffset(1.3,"Y");
      HH[j][i]->GetXaxis()->SetTitle("detected");
      HH[j][i]->GetYaxis()->SetTitle("injected");
      HH[j][i]->SetStats(kFALSE);
      sprintf(fname,"%s: %s",ifoname[j],iname[i]);
      HH[j][i]->SetTitle(fname);
      HH[j][i]->SetMarkerColor(2);
    }
      
  }


  TH1F* hcor = new TH1F("hcor","",100,0,1.);
  hcor->SetTitleOffset(1.3,"Y");
  hcor->GetXaxis()->SetTitle("network correlation");
  hcor->GetYaxis()->SetTitle("events");
  hcor->SetStats(kFALSE);
  
  TH1F* nCut = new TH1F("nCut","",500,0,100.);
  nCut->SetTitleOffset(1.3,"Y");
  nCut->GetXaxis()->SetTitle("#sqrt{#rho_{eff}}");
  nCut->GetYaxis()->SetTitle("events");

  TH1F* eSNR = new TH1F("eSNR","",200,0,10.);
  eSNR->SetTitleOffset(1.3,"Y");
  eSNR->GetXaxis()->SetTitle("#sqrt{#rho_{eff}}");
  eSNR->GetYaxis()->SetTitle("events");
  eSNR->SetStats(kFALSE);
  
  TH2F* rsxc = new TH2F("rsxc","",100,0.,1.,500,0,5.);
  rsxc->SetTitleOffset(1.3,"Y");
  rsxc->GetXaxis()->SetTitle("network correlation");
  rsxc->GetYaxis()->SetTitle("log10(average SNR)");
  rsxc->SetStats(kFALSE);
  
  TH2F* skyc = new TH2F("skyc","",360,-360,360.,180,-180.,180.);
  skyc->SetTitleOffset(1.3,"Y");
  skyc->GetXaxis()->SetTitle("phi, deg.");
  skyc->GetYaxis()->SetTitle("theta, deg.");
  skyc->SetStats(kFALSE);
  
  TH2F* c2orS = new TH2F("c2orS","",100,0.,1.,500,0,10.);
  c2orS->SetTitleOffset(1.3,"Y");
  c2orS->GetXaxis()->SetTitle("network correlation");
  c2orS->GetYaxis()->SetTitle("average SNR");
  c2orS->SetMarkerColor(1);
  c2orS->SetMarkerStyle(20);
  c2orS->SetMarkerSize(0.2);
  c2orS->SetStats(kFALSE);

  TH2F* hrs1[nIFO];
  TH1F* ifoT[nIFO];
  TH2F* hrs2[nIFO];
  
  for(int i=0; i<nIFO; i++){
    
    ifoT[i] = new TH1F("ifoT","",250,-0.05,0.05);
    ifoT[i]->SetTitleOffset(1.3,"Y");
    ifoT[i]->GetXaxis()->SetTitle("detected-injected time, sec");
    ifoT[i]->GetYaxis()->SetTitle("events");
    ifoT[i]->SetTitle(ifoname[i]);
    
    hrs1[i] = new TH2F("hrs1","",500,-23.,-18.,500,-23.,-18.);
    hrs1[i]->SetTitleOffset(1.3,"Y");
    hrs1[i]->GetXaxis()->SetTitle("detected");
    hrs1[i]->GetYaxis()->SetTitle("injected");
    hrs1[i]->SetStats(kFALSE);
    hrs1[i]->SetTitle(ifoname[i]);
    hrs1[i]->SetMarkerColor(2);
    
    hrs2[i] = new TH2F("hrs2","",500,-23.,-18.,500,-23.,-18.);
    hrs2[i]->SetTitleOffset(1.3,"Y");
    hrs2[i]->SetMarkerColor(2);
    hrs2[i]->SetStats(kFALSE);
    
  }
    
// loop over injection tree

  for(i=0; i<nmdc; i++){
    mm.GetEntry(i);
    
    save = true;
    indx = inDex[mm.type-1];
    if(indx >= ninj) continue;   // skip unwanted injections
    n = sMDC.size();
    for(j=0; j<n; j++){
      if(fabs(sMDC.data[j]-mm.strain)<1.e-24) { 
	save = false; nMDC[indx].data[j]+=1.; break; 
      }
    }
    
    if(save) { 
      sMDC.append(mm.strain); 
      for(k=0; k<32; k++){ nMDC[k].append(1.); }
    }

  }
  
  n = sMDC.size();
  for(i=0; i<32; i++){
    nTRG[i].resize(n); nTRG[i]=0.;
    err[i].resize(n);  err[i]=0.;
    e00[i].resize(n);  e00[i]=0.;
    eff[i].resize(n);  eff[i]=0.;
  }
  
  cout<<"total events: "<<ntrg<<endl;

// loop over trigger tree
  
  for(i=0; i<ntrg; i++){
    ww.GetEntry(i);

   if(ww.ecor<0) continue;
    
    if(fabs(ww.time[0]-ww.time[nIFO])>0.1) continue;    // skip random coincidence
    indx = inDex[ww.type[1]-1];   
    if(indx >= ninj) continue;                          // skip unwanted injections
        
    null = etot = rsnr = 0.;
    for(j=0; j<nIFO; j++) {
      null += ww.null[j];
      rsnr += ww.rSNR[j]/nIFO;
      enrg[j]= ww.snr[j]-ww.null[j];
      etot += enrg[j];
    }

    a2or = 1.e50;
    for(j=0; j<nIFO; j++) {
      if(a2or > etot-enrg[j]) a2or = etot-enrg[j]; 
    }
    if(a2or<0.) continue;

    acor = 2*sqrt(ww.ecor/nIFO);
    xcor = ww.netcor;                  // network correlation
    esnr = pow(rsnr,xcor/2);           // effective SNR
    
    asnr = sqrt(a2or/2.);
    if(asnr > acor) asnr = acor;       // SNR per detector

    c2orS->Fill(xcor,asnr);

    asnr = pow(asnr,sqrt(xcor));

    hcor->Fill(xcor);
    nCut->Fill(esnr);
    eSNR->Fill(esnr);
    rsxc->Fill(xcor,log10(rsnr));
    
    if(xcor < T_cor) continue;

    if(ww.frequency[0]>200 && asnr < T_cut) continue;
    if(ww.frequency[0]<200 && asnr < T_CUT) continue;
    
    
    skyc->Fill(ww.phi[0]-ww.phi[1],ww.theta[0]-ww.theta[1]); 

    for(j=0; j<nIFO; j++){
      HH[j][ninj]->Fill(log10(ww.hrss[j]),log10(ww.hrss[j+nIFO]));
      HH[j][indx]->Fill(log10(ww.hrss[j]),log10(ww.hrss[j+nIFO]));
      ifoT[j]->Fill(ww.time[j]-ww.time[j+nIFO]);
   }
      
    hT[indx]->Fill(ww.time[0]-ww.time[nIFO]);
    hF[indx]->Fill(float(ww.frequency[0]));
    hT[ninj]->Fill(ww.time[0]-ww.time[nIFO]);
    hF[ninj]->Fill(float(ww.frequency[0]-(f_low[indx]+fhigh[indx])/2));
    
    
    n = sMDC.size();
    for(j=0; j<n; j++){
      if(fabs(sMDC.data[j]-ww.strain[1])<1.e-24) { 
	nTRG[indx].data[j]+=1.; break; 
      }
    }
  }


  n = sMDC.size();
  for(i=0; i<ninj; i++){
    cout<<endl;
    for(j=0; j<n; j++){
      if(i>=0) {cout<<nTRG[i].data[j]<<"/"<<nMDC[i].data[j]<<endl;}
      if(nMDC[i].data[j] <= 0.) continue;
      eff[i].data[j] = nTRG[i].data[j]/nMDC[i].data[j];
      err[i].data[j] = eff[i].data[j]>1 ? 0.01 : 
	sqrt(eff[i].data[j]*(1-eff[i].data[j])/nMDC[i].data[j]);
    }
  }


/*
TCanvas *c1 = new TCanvas("c","C",0,0,800,800);
c1->SetBorderMode(0);
c1->SetFillColor(0);
c1->SetBorderSize(2);
c1->SetLogx(kFALSE);
c1->SetGridx();
c1->SetGridy();
c1->SetRightMargin(0.0517039);
c1->SetTopMargin(0.0772727);
c1->SetBottomMargin(0.143939);

 for(i=0; i<ninj; i++){
   c1->SetLogx(kFALSE);
   c1->SetLogy();
   
   c1->Clear();
   hF[i]->Draw();
   sprintf(fname,"%s/f_%s.gif",simname,iname[i]);
   c1->SaveAs(fname);
   
   c1->Clear();
   hT[i]->Draw();
   sprintf(fname,"%s/t_%s.gif",simname,iname[i]);
   c1->SaveAs(fname);
   
   c1->SetLogx(kFALSE);
   c1->SetLogy(kFALSE);
   for(j=0;j<nIFO;j++){
     c1->Clear();
     HH[j][i]->Draw();
     sprintf(fname,"%s/hrss_%s_%s.gif",simname,ifoname[j],iname[i]);
     c1->SaveAs(fname); 
   }
 }

 for(j=0;j<nIFO;j++){
   c1->Clear();
   HH[j][ninj]->Draw();
   sprintf(fname,"%s/hrss_%s_%s.gif",simname,ifoname[j],iname[i]);
   c1->SaveAs(fname); 

   c1->Clear();
   c1->SetLogy(kTRUE);
   ifoT[j]->Draw();
   sprintf(fname,"%s/time_%s_%s.gif",simname,ifoname[j],iname[ninj]);
   c1->SaveAs(fname);
   c1->SetLogy(kFALSE);
 }


 c1->Clear();
 c1->SetLogy(kTRUE);
 c1->SetLogx(kFALSE);
 hcor->Draw();
 sprintf(fname,"%s/netcor_%s.gif",simname,iname[ninj]);
 c1->SaveAs(fname);
 c1->SetLogy(kFALSE);

 c1->Clear();
 c1->SetLogy(kTRUE);
 nCut->Draw();
 sprintf(fname,"%s/effSNR_%s.gif",simname,iname[ninj]);
 c1->SaveAs(fname);
 c1->SetLogy(kFALSE);

 c1->Clear();
 c1->SetLogy(kTRUE);
 eSNR->Draw();
 sprintf(fname,"%s/effSNR_%s_10.gif",simname,iname[ninj]);
 c1->SaveAs(fname);
 c1->SetLogy(kFALSE);

 c1->Clear();
 rsxc->Draw();
 sprintf(fname,"%s/rsnr_xcor_%s.gif",simname,iname[ninj]);
 c1->SaveAs(fname);

 c1->Clear();
 skyc->Draw();
 sprintf(fname,"%s/coordinate_%s.gif",simname,iname[ninj]);
 c1->SaveAs(fname);

// Efficiency
*/
TCanvas *c1 = new TCanvas("c","C",0,0,1000,800);
c1->SetBorderMode(0);
c1->SetFillColor(0);
c1->SetBorderSize(2);
c1->SetLogx(kFALSE);
c1->SetGridx();
c1->SetGridy();
c1->SetRightMargin(0.0517039);
c1->SetTopMargin(0.0772727);
c1->SetBottomMargin(0.143939);

 TGraphErrors* EFF[ninj];
 double chi2, hrss50, sigma, betam, betap, hrssEr;


 n = sMDC.size();

 sprintf(fname,"%s/fit_parameters_%s_%d_%d.txt",simname,iname[ninj],int(T_cor*100),int(T_cut*10));
 in = fopen(fname,"w");

 legend->Clear();
for(i=0; i<ninj; i++){

  c1->Clear();
  EFF[i]=new TGraphErrors(n,sMDC.data,eff[i].data,e00[i].data,err[i].data);
  EFF[i]->GetXaxis()->SetTitle("hrss, #frac{strain}{#sqrt{Hz}}");
  EFF[i]->GetYaxis()->SetTitle("Efficiency    ");
  EFF[i]->GetXaxis()->SetTitleOffset(1.3);
  EFF[i]->GetYaxis()->SetTitleOffset(1.3);

  c1->SetLogx();
  f4->SetParameters(-21.0,0.7,1.,1.);
  f4->SetParNames("hrss50","sigma","betam","betap");
  f4->SetLineColor(colors[i]);
  f4->SetParLimits(0,-22.,-18.5);
  f4->SetParLimits(1,0.1,10.);
  f4->SetParLimits(2,0.1,4.);
  f4->SetParLimits(3,0.1,4.);
  EFF[i]->Fit("logNfit");
  chi2=f4->GetChisquare();
  hrss50=pow(10.,f4->GetParameter(0));
  hrssEr=(pow(10.,f4->GetParError(0))-1.)*hrss50;
  sigma=f4->GetParameter(1);
  betam=f4->GetParameter(2);
  betap=f4->GetParameter(3);

  printf("%d %s %E %E +- %E %E %E %E\n",
	 i,iname[i],chi2,hrss50,hrssEr,sigma,betam,betap);
  fprintf(in,"%2d %9.3E %9.3E +- %9.3E %9.3E %9.3E %9.3E %s\n",
	  i,chi2,hrss50,hrssEr,sigma,betam,betap,iname[i]);
  
  sprintf(fname,"%s, hrss50=%5.2E", iname[i],hrss50);
  EFF[i]->SetTitle(fname);
  gStyle->SetOptStat(01);
  f4->SetLineColor(colors[i]);
  EFF[i]->SetLineColor(colors[i]);
  EFF[i]->SetMarkerStyle(markers[i]);
  EFF[i]->SetMarkerColor(colors[i]);
  EFF[i]->SetMarkerSize(2);

  sprintf(fname,"%s, %5.2E",iname[i], hrss50);
  legend->AddEntry(EFF[i], fname, "lp");
  EFF[i]->Draw("AP");
//  sprintf(fname,"%s/%s.gif",simname,iname[i]);

//  c1->SaveAs(fname);
  c1->SetLogx(kFALSE);
  c1->Clear();
}
fclose(in);

c1->Clear();
c1->SetLogx();
c1->SetLogy(kFALSE);
sprintf(fname,"%s #rho_{avr}=%3.1f, cc=%3.1f",iname[ninj],T_cut,T_cor);
EFF[0]->SetTitle(fname);
EFF[0]->Draw("AP");

for(i=1;i<ninj;i++){ 
  EFF[i]->Draw("SAMEP"); 
}
legend->Draw("SAME");
sprintf(fname,"%s/eff_%s_%d_%d.gif",simname,iname[ninj],int(T_cor*100),int(T_cut*10));
c1->SaveAs(fname);


}



