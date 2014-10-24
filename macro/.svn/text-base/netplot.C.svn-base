{
// thresholds

double T_cor = 0.6;
double T_cut = 3.4; // s5b 3.5
double T_CUT = 4.5; // s5b 4.2

 double ecor, xcor, acor, rsnr, esnr, null, etot, asnr, a2or;
double enrg[5];
int i,j,n,m,k;
bool save;

size_t ntype = 0;
char fname[256];

 wavearray<double> Xcut(50);
 wavearray<double> Ycut(50); Ycut = 0.;
 wavearray<double> yCUT(50); yCUT = 0.;
 wavearray<double> Xerr(50); Xerr = 0.;
 wavearray<double> Yerr(50);
 wavearray<double> yERR(50);

// calculate live time

 float xlag;
 Int_t xrun, rUn, iday;
 double xlive, sTARt;
 double liveTot = 0.;
 double Live;

 wavearray<double> Trun(50000); Trun = 0.;
 wavearray<double> Nday(500); 
 wavearray<double> Eday(500); Eday = 0.; 
 wavearray<double> Rerr(500); Rerr = 0.; 
 wavearray<double> rerr(500); rerr = 0.; 
 wavearray<double> rcor(500); rcor = 0.; 
 wavearray<double> rday(500); rday = 0.; 
 wavearray<double> Rday(500); Rday = 0.; 
 wavearray<double> Tday(500); Tday = 0.; 

 for(i=0; i<500; i++) Nday.data[i] = i+1;

 wavearray<double> Wlag;
 wavearray<double> Tlag;
 wavearray<double> Rlag;
 wavearray<double> Elag;
 wavearray<double> Olag;

 liv->SetBranchAddress("lag",&xlag);
 liv->SetBranchAddress("run",&xrun);
 liv->SetBranchAddress("live",&xlive);
 int ntrg = liv.GetEntries();
 for(i=0; i<ntrg; i++){
   liv->GetEntry(i);
   Live = xlag>0 ? xlive : 0.;
   liveTot += Live;
   Trun.data[xrun-1] += Live;
   save = true;
   for(j=0; j<Wlag.size(); j++){
     if(float(Wlag.data[j]) == xlag) { 
       Tlag.data[j] += xlive;
       save=false; 
       break; 
     }
   }
   if(save) { Wlag.append(xlag); Tlag.append(double(xlive)); }
 }
 Rlag = Tlag; Rlag = 0.;
 Elag = Tlag; Elag = 0.;
 Olag = Tlag; Olag = 0.;

 sprintf(fname,"%s/live.txt",netdir);
 FILE* flive = fopen(fname,"w");
 W.GetEntry(1);
 fprintf(flive,"# %s shift = %6.3f \n",ifoname[0],W.lag[0]);
 fprintf(flive,"# %s shift = %6.3f \n",ifoname[1],W.lag[1]);
 fprintf(flive,"# %s shift = %6.3f \n",ifoname[2],W.lag[2]);

 for(j=0; j<Wlag.size(); j++){
   fprintf(flive,"lag=%5.2f  live=%9.2f\n",Wlag.data[j],Tlag.data[j]);
 }
 fprintf(flive,"total live time: non-zero lag = %10.1f \n",liveTot); 
 fclose(flive);
 printf("total live time: non-zero lag = %10.1f \n",liveTot); 

// histograms

 TH1F* xCor = new TH1F("xCor","",100,0,1.);
 xCor->SetTitleOffset(1.3,"Y");
 xCor->SetTitle("red: frequency > 200 Hz");
 xCor->GetXaxis()->SetTitle("network correlation");
 xCor->GetYaxis()->SetTitle("events");

 TH1F* asnrP = new TH1F("asnrP","",200,0,20.);
 asnrP->SetTitleOffset(1.3,"Y");
 asnrP->SetTitle("");
 asnrP->GetXaxis()->SetTitle("average SNR");
 asnrP->GetYaxis()->SetTitle("events");

 TH1F* rateA = new TH1F("rateA","",366,0,365.);
 rateA->SetTitleOffset(1.3,"Y");
 rateA->SetTitle("rate");
 rateA->GetXaxis()->SetTitle("day");
 rateA->GetYaxis()->SetTitle("rate, Hz");
 rateA->SetLineColor(2);
 TH1F* rateB = new TH1F("rateB","",366,0,365.);
 rateB->SetTitleOffset(1.3,"Y");
 rateB->SetTitle("rate");
 rateB->GetXaxis()->SetTitle("day");
 rateB->GetYaxis()->SetTitle("rate, Hz");
 rateB->SetLineColor(4);
 TH1F* rateC = new TH1F("rateC","",366,0,365.);
 rateC->SetTitleOffset(1.3,"Y");
 rateC->SetTitle("rate");
 rateC->GetXaxis()->SetTitle("day");
 rateC->GetYaxis()->SetTitle("rate, Hz");
 rateC->SetLineColor(3);

 TH1F* cCut = new TH1F("cCut","",200,1,10.);
 cCut->SetTitleOffset(1.3,"Y");
 cCut->SetTitle("red: frequency > 200 Hz");
 cCut->GetXaxis()->SetTitle("#sqrt{#rho_{eff}}");
 cCut->GetYaxis()->SetTitle("events");

 TH1F* eCor = new TH1F("eCor","",200,0,50.);
 eCor->SetTitleOffset(1.3,"Y");
 eCor->SetTitle("red: frequency > 200 Hz");
 eCor->GetXaxis()->SetTitle("correlated amplitude");
 eCor->GetYaxis()->SetTitle("events");

 TH1F* rSNR = new TH1F("rSNR","",100,0,5.);
 rSNR->SetTitleOffset(1.3,"Y");
 rSNR->SetTitle("red: frequency > 200 Hz");
 rSNR->GetXaxis()->SetTitle("log10(average SNR)");
 rSNR->GetYaxis()->SetTitle("events");

 TH1F* Like = new TH1F("Like","",100,0,5.);
 Like->SetTitleOffset(1.3,"Y");
 Like->SetTitle("red: frequency < 200 Hz");
 Like->GetXaxis()->SetTitle("log10(likelihood)");
 Like->GetYaxis()->SetTitle("events");

 TH1F* xCorf = new TH1F("xCorf","",100,0,1.);
 xCorf->SetTitleOffset(1.3,"Y");
 xCorf->SetTitle("frequency > 200 Hz");
 xCorf->GetXaxis()->SetTitle("network correlation");
 xCorf->GetYaxis()->SetTitle("events");
 xCorf->SetLineColor(2);
 xCorf->SetFillColor(2);

 TH1F* cCutf = new TH1F("cCutf","",200,1,6.);
 cCutf->SetTitleOffset(1.3,"Y");
 cCutf->SetTitle("frequency > 200 Hz");
 cCutf->GetXaxis()->SetTitle("#sqrt{#rho_{eff}}");
 cCutf->GetYaxis()->SetTitle("events");
 cCutf->SetFillColor(2);
 cCutf->SetLineColor(2);

 TH1F* eCorf = new TH1F("eCorf","",200,0,50.);
 eCorf->SetTitleOffset(1.3,"Y");
 eCorf->SetTitle("frequency > 200 Hz");
 eCorf->GetXaxis()->SetTitle("correlated amplitude");
 eCorf->GetYaxis()->SetTitle("events");
 eCorf->SetFillColor(2);
 eCorf->SetLineColor(2);

 TH1F* rSNRf = new TH1F("rSNRf","",200,0,5.);
 rSNRf->SetTitleOffset(1.3,"Y");
 rSNRf->SetTitle("frequency > 200 Hz");
 rSNRf->GetXaxis()->SetTitle("log10(average SNR)");
 rSNRf->GetYaxis()->SetTitle("events");
 rSNRf->SetFillColor(2);
 rSNRf->SetLineColor(2);

 TH1F* Likef = new TH1F("Likef","",100,0,5.);
 Likef->SetTitleOffset(1.3,"Y");
 Likef->SetTitle("frequency > 200 Hz");
 Likef->GetXaxis()->SetTitle("log10(likelihood)");
 Likef->GetYaxis()->SetTitle("events");
 Likef->SetFillColor(2);
 Likef->SetLineColor(2);

 TH2F* cutF = new TH2F("cutF","",1260,60,2048.,300,0,6);
 cutF->SetTitleOffset(1.2,"Y");
 cutF->GetXaxis()->SetTitle("frequency, Hz");
 cutF->GetYaxis()->SetTitle("#sqrt{#rho_{eff}}");
 cutF->SetStats(kFALSE);
 cutF->SetMarkerStyle(20);
 cutF->SetMarkerSize(0.4);
 cutF->SetMarkerColor(2);

 TH2F* rsxc = new TH2F("rsxc","",100,0.,1.,500,0,5.);
 rsxc->SetTitleOffset(1.3,"Y");
 rsxc->GetXaxis()->SetTitle("network correlation");
 rsxc->GetYaxis()->SetTitle("log10(average SNR)");
 rsxc->SetStats(kFALSE);
 rsxc->SetMarkerStyle(20);
 rsxc->SetMarkerSize(0.4);
 rsxc->SetMarkerColor(2);

 TH2F* ecxc = new TH2F("ecxc","",100,0.,1.,500,0,50.);
 ecxc->SetTitleOffset(1.3,"Y");
 ecxc->GetXaxis()->SetTitle("network correlation");
 ecxc->GetYaxis()->SetTitle("correlated amplitude");
 ecxc->SetStats(kFALSE);

 TH2F* c2orP = new TH2F("c2orP","",100,0.,1.,500,0,10.);
 c2orP->SetTitleOffset(1.3,"Y");
 c2orP->GetXaxis()->SetTitle("network correlation");
 c2orP->GetYaxis()->SetTitle("average SNR");
 c2orP->SetMarkerStyle(20);
 c2orP->SetMarkerColor(2);
 c2orP->SetMarkerSize(0.2);
 c2orP->SetStats(kFALSE);

// read events

 sprintf(fname,"%s/events.txt",netdir);
 FILE* ftrig = fopen(fname,"w");
 fprintf(ftrig,"# correlation threshold = %f \n",T_cor);
 fprintf(ftrig,"# network SNR threshold = %f  %f\n",T_cut, T_CUT);

 int ntrg = net.GetEntries();
 cout<<"total events: "<<ntrg<<endl;

 W.GetEntry(0); sTARt = W.gps;
 printf("data start GPS: %15.5f \n",sTARt);
 rUn = -1;

 for(i=0; i<ntrg; i++){
   W.GetEntry(i);

   if(W.lag[0]==0) continue;
   if(W.ecor<0) continue;

   iday = int((W.gps - sTARt)/3600./24.);

   if(W.run != rUn) {
     rUn = int(W.run+0.1);
     Tday.data[iday] += Trun.data[rUn];
   }

   acor = 2*sqrt(W.ecor/nIFO);
   xcor = W.netcor;                  // network correlation

   null = etot = rsnr = 0.;
   for(j=0; j<nIFO; j++) {
     null += W.null[j];
     rsnr += W.rSNR[j]/nIFO;
     enrg[j]= W.snr[j]-W.null[j];
     etot += enrg[j];
   }

   a2or = 1.e50;
   for(j=0; j<nIFO; j++) {
     if(a2or > etot-enrg[j]) a2or = etot-enrg[j]; 
   }
   if(a2or<0.) continue;

   Rday.data[iday] += 1.;

   asnr = sqrt(a2or/2.);
   if(asnr > acor) asnr = acor;       // SNR per detector

   //   if(a2or < T_cut) continue;
  
   c2orP->Fill(xcor,asnr);

   esnr = pow(rsnr,xcor/2);           // effective rank SNR
   asnr = pow(asnr,sqrt(xcor));       // effective SNR
   //   asnr = pow(asnr,xcor);             // effective SNR

   rday.data[iday] += 1.;
   
   xCor->Fill(xcor);
   if(W.frequency[0]>200) xCorf->Fill(xcor);


   if(xcor < T_cor) continue;

   asnrP->Fill(asnr);

   rcor.data[iday] += 1.;
   
   rSNR->Fill(log10(rsnr));
   eCor->Fill(sqrt(2*W.ecor));
   cCut->Fill(sqrt(esnr));
   Like->Fill(log10(W.likelihood));

   if(W.frequency[0]>200) {
     rSNRf->Fill(log10(rsnr));
     eCorf->Fill(sqrt(2*W.ecor));
     cCutf->Fill(sqrt(esnr));
     Likef->Fill(log10(W.likelihood));
   }      
   
   for(j=0; j<Wlag.size(); j++){
     if(W.lag[0]==Wlag.data[j] && asnr>2.5) Rlag.data[j] += 1.;
   }
   
   for(j=0; j<Xcut.size(); j++){
     Xcut.data[j] = 1+j*0.1;
     if(asnr >  Xcut.data[j]) {Ycut.data[j] += 1.;}
     if(asnr >  Xcut.data[j] && W.frequency[0]>200) {yCUT.data[j] += 1.;}
   }
   
   if(W.frequency[0]> 200 && asnr < T_cut) continue;
   if(W.frequency[0]<=200 && asnr < T_CUT) continue;
   
   //   printf("%15.5f  %15.5f  %15.5f  %15.5f  %15.5f  %15.5f  %15.5f  %15.5f\n",
   //	  W.start[0]-0.00,W.stop[0]+0.00,
   //	  W.start[1]-0.00,W.stop[1]+0.00,
   //	  W.start[2]-0.00,W.stop[2]+0.00,
   //     W.start[3]-0.00,W.stop[3]+0.00);

   
   fprintf(ftrig,
	   "%13.3f  %13.3f  %4.1f  %4.3f  %5.0f  %4.2f  %4.0f  %4.0f  %5.1e  %5.1e  %5.1e  %5.1e  %5.1e  %5.1e  %3d  %3.0f  %d\n",
	   W.start[0],W.start[1],rsnr,xcor,W.likelihood,sqrt(esnr),W.frequency[0],
           W.high[0]-W.low[0],W.snr[0],W.snr[1],W.snr[2],
	   W.hrss[0],W.hrss[1],W.hrss[2],W.size[0],W.rate[0],int(W.run));

   //    fprint("%13.3f  %13.3f  %4.1f  %4.3f  %5.0f  %4.2f  %4.0f  %3d  %3.0f  %d \n",
   //	   W.start[0],W.start[1],rsnr,xcor,W.likelihood,sqrt(esnr),W.frequency[0],
   //	   W.size[0],W.rate[0],int(W.run));

 }
 fclose(ftrig);
 

 for(j=0; j<Wlag.size(); j++){
   Elag.data[j] = sqrt(Rlag.data[j])/Tlag.data[j];
   Rlag.data[j]/= Tlag.data[j];
 }

 for(j=0; j<Xcut.size(); j++){
   Yerr.data[j] = sqrt(Ycut.data[j])/liveTot;
   Ycut.data[j] = Ycut.data[j]/liveTot;
   yERR.data[j] = sqrt(yCUT.data[j])/liveTot;
   yCUT.data[j] = yCUT.data[j]/liveTot;
 }
 
i = 0;

 for(j=0; j<Tday.size(); j++) {
   if(Tday.data[j] > 0.) {
     rerr.data[j]  = sqrt(rday.data[j]);
     Rerr.data[j]  = sqrt(Rday.data[j]);
     Rday.data[j] /= Tday.data[j];
     rday.data[j] /= Tday.data[j];
     rcor.data[j] /= Tday.data[j];
     Rerr.data[j] /= Tday.data[j];
     rerr.data[j] /= Tday.data[j];
   }
   rateA->Fill(float(i),Rday.data[j]);
   rateB->Fill(float(i),rday.data[j]);
   rateC->Fill(float(i),rcor.data[j]);
   i++;
 }

 
TCanvas *c1 = new TCanvas("c","C",0,0,800,600);
c1->SetBorderMode(0);
c1->SetFillColor(0);
c1->SetBorderSize(2);
c1->SetGridx();
c1->SetGridy();
//c1->SetRightMargin(0.0517039);
//c1->SetTopMargin(0.0772727);
//c1->SetBottomMargin(0.143939);

/*
grr=new TGraphErrors(i,Nday.data,rday.data,Eday.data,rerr.data);
grr->SetTitle("rate vs time");
grr->SetLineColor(1);
grr->SetMarkerStyle(20);
grr->SetMarkerColor(1);
grr->SetMarkerSize(0.2);
grr->SetMinimum(0.5/3600./24.);
c1->SetLogy(kTRUE);
grr->Draw("AP");
sprintf(fname,"day since %d",int(sTARt));
grr->GetHistogram()->SetXTitle(fname);
grr->GetHistogram()->SetYTitle("rate, Hz");
sprintf(fname,"%s/rate_day.gif",netdir);
c1->SaveAs(fname);


c1->SetLogy(kTRUE);
rateA->Draw();
gStyle->SetOptStat(kFALSE);
rateA->SetMinimum(1.e-7);
sprintf(fname,"%s/rateA_day.gif",netdir);
c1->SaveAs(fname);
rateB->Draw();
gStyle->SetOptStat(kFALSE);
rateB->SetMinimum(1.e-7);
sprintf(fname,"%s/rateB_day.gif",netdir);
c1->SaveAs(fname);
rateC->Draw();
gStyle->SetOptStat(kFALSE);
rateC->SetMinimum(1.e-7);
sprintf(fname,"%s/rateC_day.gif",netdir);
c1->SaveAs(fname);

rateA->Draw();
gStyle->SetOptStat(kFALSE);
rateB->Draw("same");
gStyle->SetOptStat(kFALSE);
sprintf(fname,"%s/rate2_day.gif",netdir);
c1->SaveAs(fname);

rateA->Draw();
gStyle->SetOptStat(kFALSE);
rateB->Draw("same");
gStyle->SetOptStat(kFALSE);
rateC->Draw("same");
gStyle->SetOptStat(kFALSE);
sprintf(fname,"%s/rate_day.gif",netdir);
c1->SaveAs(fname);
*/

c1->Clear();
c1->SetLogy(kFALSE);
gr2=new TGraphErrors(Wlag.size(),Wlag.data,Rlag.data,Olag.data,Elag.data);
gr2->SetTitle("rate vs lag");
gr2->SetLineColor(1);
gr2->SetMarkerStyle(20);
gr2->SetMarkerColor(1);
gr2->SetMarkerSize(1);
gr2->Draw("AP");
gr2->GetHistogram()->SetXTitle("lag, sec");
gr2->GetHistogram()->SetYTitle("rate, Hz");

sprintf(fname,"%s/rate_lag.gif",netdir);
c1->SaveAs(fname);

c1->Clear();
gr1=new TGraphErrors(Ycut.size(),Xcut.data,Ycut.data,Xerr.data,Yerr.data);
gr1->SetTitle("rate vs threshold");
gr1->SetLineColor(1);
gr1->SetMarkerStyle(20);
gr1->SetMarkerColor(1);
gr1->SetMarkerSize(1);
gr1->SetMinimum(0.5/liveTot);
c1->SetLogy(kTRUE);
gr1->Draw("AP");
gr1->GetHistogram()->SetXTitle("#sqrt{#rho_{eff}}");
gr1->GetHistogram()->SetYTitle("rate, Hz");

grf=new TGraphErrors(Ycut.size(),Xcut.data,yCUT.data,Xerr.data,yERR.data);
grf->SetTitle("rate vs threshold");
grf->SetLineColor(1);
grf->SetMarkerStyle(20);
grf->SetMarkerColor(2);
grf->SetMarkerSize(1);
grf->SetMinimum(0.5/liveTot);
c1->SetLogy(kTRUE);
grf->Draw("P");
grf->GetHistogram()->SetXTitle("#sqrt{#rho_{eff}}");
grf->GetHistogram()->SetYTitle("rate, Hz");

sprintf(fname,"%s/rate_threshold.gif",netdir);
c1->SaveAs(fname);

/*

gStyle->SetOptStat(kTRUE);
c1->Clear();
c1->SetLogy(kTRUE);
xCor->Draw("");
xCorf->Draw("same");
sprintf(fname,"%s/netcor.gif",netdir);
c1->SaveAs(fname);

c1->Clear();
eCor->Draw("");
eCorf->Draw("same");
sprintf(fname,"%s/Ecor.gif",netdir);
c1->SaveAs(fname);

c1->Clear();
rSNR->Draw("");
rSNRf->Draw("same");
sprintf(fname,"%s/ranksnr.gif",netdir);
c1->SaveAs(fname);

c1->Clear();
cCut->Draw("");
cCutf->Draw("same");
sprintf(fname,"%s/effSNR.gif",netdir);
c1->SaveAs(fname);

c1->Clear();
Like->Draw("");
Likef->Draw("same");
sprintf(fname,"%s/likelihood.gif",netdir);
c1->SaveAs(fname);

c1->SetLogy(kFALSE);
c1->Clear();
rsxc->Draw();
rsxc->SetTitle("");
Draw(net,RSN+":"+XuC,L36+"&&"+CUT,"same");
sprintf(fname,"%s/rsnr_xcor.gif",netdir);
c1->SaveAs(fname);

c1->Clear();
rsxc->Draw();
rsxc->SetTitle("frequency>200 Hz");
Draw(net,RSN+":"+XuC,L36+"&&"+CUT+"&&frequency[0]>200","same");
sprintf(fname,"%s/rsnr_xcor_gt200Hz.gif",netdir);
c1->SaveAs(fname);

c1->Clear();
ecxc->Draw();
ecxc->SetTitle("");
Draw(net,"sqrt("+cE+"):"+XuC,L36+"&&"+CUT,"same");
sprintf(fname,"%s/ecor_xcor.gif",netdir);
c1->SaveAs(fname);

c1->Clear();
ecxc->Draw();
ecxc->SetTitle("frequency>200 Hz");
Draw(net,"sqrt("+cE+"):"+XuC,L36+"&&"+CUT+"&&frequency[0]>200","same");
sprintf(fname,"%s/ecor_xcor_gt200Hz.gif",netdir);
c1->SaveAs(fname);
 
c1->Clear();
c1->SetLogx(kTRUE);
Draw(net,"pow(10,"+RSN+"*"+XuC+"/2.):frequency[0]>>cutF",L36+"&&"+CUT+"&&"+XuC+">0.65","");
sprintf(fname,"%s/effSNR_frequency.gif",netdir);
c1->SaveAs(fname);
*/
}



