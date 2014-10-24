{
// thresholds
double T_like = 50.;
double T_gcor = 0.7;
double T_pcor = 0.4;
double T_chi2 = 3.;
double a,gs,sum;
double gsf[3];
int n,m,k;
int index[3];

char fname[256];

int nb = net.GetEntries();
cout<<nb<<endl;

float xcor;

/*
TH1F* h1chi = new TH1F("h1chi","",128,0.,16.); 
h1chi->SetTitleOffset(1.5,"Y");
h1chi->GetXaxis()->SetTitle("chi2");
h1chi->GetYaxis()->SetTitle("events");
*/
TH1F* hlik = new TH1F("hlik","",200,0.,200.); 
hlik->SetTitleOffset(1.5,"Y");
hlik->GetXaxis()->SetTitle("chi2");
hlik->GetYaxis()->SetTitle("events");


TH1F* hgs = new TH1F("hgs","",100,0.,5.); 
hgs->SetTitleOffset(1.,"Y");
hgs->GetXaxis()->SetTitle("geometric significance)");
hgs->GetYaxis()->SetTitle("events");
sprintf(fname,"");
hgs->SetTitle(fname);


TH1F* hchi[6];
TH1F* hgsf[6];
TH1F* hcsf[6];
TH1F* hcor[6];
TH1F* hind[6];
for(int i=0; i<6; i++){
   hchi[i] = new TH1F("hchi","",100,0.,30.);
   hchi[i]->SetTitleOffset(1.5,"Y");
   hchi[i]->GetXaxis()->SetTitle("-log10(chi2 probability)");
   hchi[i]->GetYaxis()->SetTitle("events");
   sprintf(fname,"NetBurst");
   hchi[i]->SetTitle(fname);

   hgsf[i] = new TH1F("hgsf","",100,0.,30.);
   hgsf[i]->SetTitleOffset(1.5,"Y");
   hgsf[i]->GetXaxis()->SetTitle("-log10(chi2 probability)");
   hgsf[i]->GetYaxis()->SetTitle("events");
   sprintf(fname,"NetBurst");
   hgsf[i]->SetTitle(fname);

   hcsf[i] = new TH1F("hcsf","",100,0.,30.);
   hcsf[i]->SetTitleOffset(1.5,"Y");
   hcsf[i]->GetXaxis()->SetTitle("-log10(chi2 probability)");
   hcsf[i]->GetYaxis()->SetTitle("events");
   sprintf(fname,"NetBurst");
   hcsf[i]->SetTitle(fname);
   if(i>2) hcsf[i]->SetLineColor(2);

   hcor[i] = new TH1F("hcor","",100,-1.,1.);
   hcor[i]->SetTitleOffset(1.5,"Y");
   hcor[i]->GetXaxis()->SetTitle("network x-correlation");
   hcor[i]->GetYaxis()->SetTitle("events");
   sprintf(fname,"NetBurst");
   hcor[i]->SetTitle(fname);
   if(i>2) hcor[i]->SetLineColor(2);

   hind[i] = new TH1F("hind","",30,0,15);
   hind[i]->SetTitleOffset(1.5,"Y");
   hind[i]->GetXaxis()->SetTitle("coincidence index");
   hind[i]->GetYaxis()->SetTitle("events");
   sprintf(fname,"NetBurst");
   hind[i]->SetTitle(fname);
   if(i>2) hind[i]->SetLineColor(2);

}

for(i=0; i<nb; i++){
  W.GetEntry(i);

  if(W.likelihood<T_like) continue;

  gsf[0] = 10*W.gSF[0]/log(10);
  if(gsf[0] > W.chi2[0]) gsf[0] = W.chi2[0];
  gsf[1] = 10*W.gSF[1]/log(10);
  if(gsf[1] > W.chi2[1]) gsf[1] = W.chi2[1];
  gsf[2] = 10*W.gSF[2]/log(10);
  if(gsf[2] > W.chi2[2]) gsf[2] = W.chi2[2];

  a = W.snr[0]+W.snr[1]+W.snr[2]-2*W.likelihood;
  if(a<3*W.size[0]) a=3*W.size[0];
  xcor = 4*(W.ndm[1]+W.ndm[2]+W.ndm[4])/(a+4*abs(W.ndm[1]+W.ndm[2]+W.ndm[4]));

  if(W.lag[0]==0) hcor[0]->Fill(xcor);
  if(W.lag[0]!=0) hcor[3]->Fill(xcor);

  /*
  if(W.lag[0]==0 && gsf[0]>4) hcor[1]->Fill(xcor);
  if(W.lag[0]==0 && gsf[0]<4) hcor[2]->Fill(xcor);
  if(W.lag[0]!=0 && gsf[0]>4) hcor[4]->Fill(xcor);
  if(W.lag[0]!=0 && gsf[0]<4) hcor[5]->Fill(xcor);
  */

  if(xcor<T_gcor) continue;

  gs = (TMath::Log(W.gSF[0])+TMath::Log(W.gSF[1])+TMath::Log(W.gSF[2]))/3;
  gs = exp(gs)/log(10.);
  hgs->Fill(float(gs));

  if(W.lag[0]==0) hcsf[0]->Fill(float(gsf));
  if(W.lag[0]!=0) hcsf[3]->Fill(float(gsf));
  if(W.lag[0]==0) hcsf[1]->Fill(float(gsf));
  if(W.lag[0]!=0) hcsf[4]->Fill(float(gsf));
  if(W.lag[0]==0) hcsf[2]->Fill(float(gsf));
  if(W.lag[0]!=0) hcsf[5]->Fill(float(gsf));

  index[0]=index[1]=index[2]=0;

  n = 0;
  if(gsf[0]>T_chi2) { n++; index[0]=1; }
  if(gsf[1]>T_chi2) { n++; index[1]=1; } 
  if(gsf[2]>T_chi2) { n++; index[2]=1; }

  m = n;
  
  a = W.null[0]+W.null[1]; if(a<2*W.size[0]) a = 2*W.size[0];
  a = 4*W.ndm[1]/(a+4*abs(W.ndm[1]));
  if(a>T_pcor) { m++; index[0]=1; index[1]=1; }
  a = W.null[0]+W.null[2]; if(a<2*W.size[0]) a = 2*W.size[0];
  a = 4*W.ndm[2]/(a+4*abs(W.ndm[2]));
  if(a>T_pcor) { m++; index[0]=1; index[2]=1; }
  a = W.null[1]+W.null[2]; if(a<2*W.size[0]) a = 2*W.size[0];
  a = 4*W.ndm[4]/(a+4*abs(W.ndm[4]));
  if(a>T_pcor) { m++; index[1]=1; index[2]=1; }

  if(n==1) m=0;
  k = index[0] + index[1] + index[2];
  k+= n*n-2*n+m;

  hlik[0]->Fill(float(W.likelihood));
  
  hchi[0]->Fill(float(W.chi2[0]));
  hchi[1]->Fill(float(W.chi2[1]));
  hchi[2]->Fill(float(W.chi2[2]));

  hgsf[0]->Fill(float(10*W.gSF[0]/log(10)));
  hgsf[1]->Fill(float(10*W.gSF[1]/log(10)));
  hgsf[2]->Fill(float(10*W.gSF[2]/log(10)));

  if(W.lag[0]==0) hind[0]->Fill(float(k));
  if(W.lag[0]==0 && gsf[0]>4) hind[1]->Fill(float(k));
  if(W.lag[0]==0 && gsf[0]<4) hind[2]->Fill(float(k));
  if(W.lag[0]!=0) hind[3]->Fill(float(k));
  if(W.lag[0]!=0 && gsf[0]>4) hind[4]->Fill(float(k));
  if(W.lag[0]!=0 && gsf[0]<4) hind[5]->Fill(float(k));
  
  if(W.lag[0]==0 && k>6) hcor[1]->Fill(xcor);
  if(W.lag[0]!=0 && k>6) hcor[4]->Fill(xcor);


}

/*
TCanvas *c1 = new TCanvas("c","C",0,0,800,600);
c1->SetBorderMode(0);
c1->SetFillColor(0);
gPad->SetBorderMode(0);
gPad->SetFillColor(0);
//gPad->SetLogy();
gPad->SetGrid();
*/

}



