/*-------------------------------------------------------
 * Package:     Wavelet Analysis Tool
 * File name:   WTSpectrum.C
 *
 * This macro file is for ROOT interactive environment.
 * Macro plots wavelet tree spectrum.
 *-------------------------------------------------------
*/ 

struct TFplot {
  TCanvas* canvas;
  TH2F*    histogram;
};

void clear(TFplot &p) { 
  if(p.canvas) delete p.canvas; 
  if(p.histogram) delete p.histogram; 
  p.canvas = NULL;
  p.histogram = NULL;
}

void null(TFplot &p) { 
  p.canvas = NULL;
  p.histogram = NULL;
}


void WSpectrum(WSeries<double> &w,int opt=2,int pal=0)
{
   if(w.IsTree==1) WTSpectrum(w,opt,pal,0,0,"COLZ");
   else WLSpectrum(w,opt,pal,0,0,"COLZ");
}

double procOpt(int opt, double val)
{	switch(opt){
		case 1: return val*val;
		case 2: return fabs(val);
		default: return val;
	}
}

char* getName(char* prefix, char* suff=0)
{  static int cntr = 0;
   static char res[100];
   if(suff)sprintf(res, "%s_%d%s", prefix, ++cntr, suff);
   else sprintf(res, "%s_%d", prefix, ++cntr);
   return res;
}

TFplot WDMPlot(WSeries<double> &w,int opt, int pal, double t1, double t2, char* copt)
{	
  WDM<double>* wdm = (WDM<double>*) w.pWavelet;
  int M = w.getLevel();
  double* map00 = wdm->pWWS;
  double tsRate = w.rate();
  int mF = int(w.size()/wdm->nSTS); 
  int nTC = w.size()/(M+1)/mF;                   // # of Time Coefficients
  double* map90 = map00 + (mF-1)*(M+1)*nTC;
  
  //printf("nTC = %d, rate = %d  M = %d\n", nTC, (int)w.rate(), M); 
  
  // make Y bins:
  double* yBins = new double[M+2];
  double dF = tsRate/M/2.;
  yBins[0] = 0; 
  yBins[1] = dF/2;
  for(int i=2; i<=M; ++i) yBins[i] = yBins[1] + (i-1)*dF;
  yBins[M+1] = tsRate/2.;
  
  TFplot pP;
  TCanvas* c = new TCanvas(getName("WDMPlotC"), "");
  c->SetFrameBorderMode(0);
  const double scale = 1./w.wrate(); 
  TH2F* h = new TH2F(getName("WDMPlotH"), "", 2*nTC, w.start(), nTC*scale, M+1, yBins);
  
  double v;
  int it1 = t1/scale;
  int it2 = t2/scale;
  if(it2<=it1 || it2>nTC)it2 = nTC;
  
  for(int i=it1; i<it2; ++i){ 
      if(i){
         v = ( procOpt(opt, map00[0]) + procOpt(opt, map90[0]) ) /2 ; //first half-band
         h->SetBinContent( 2*i-1 , 1, v); 		
         h->SetBinContent( 2*i , 1, v);
	  
         v = ( procOpt(opt, map00[M]) + procOpt(opt, map90[M]) ) /2 ; //last half-band
         h->SetBinContent( 2*i-1 , M+1, v);
         h->SetBinContent( 2*i , M+1, v);
      }
      //printf("%d\n", i);
      for(int j=1; j<M; ++j){
         v = ( procOpt(opt, map00[j]) + procOpt(opt, map90[j]) ) /2;
         h->SetBinContent( 2*i , j+1, v);
         h->SetBinContent( 2*i+1 , j+1, v);
      }
      map00+=M+1; map90+=M+1;
  }
  
  h->SetStats(0);
  h->Draw("colz");
  h->SetXTitle("Time [s]");
  h->SetYTitle("Frequency [Hz]");
  h->SetTitleOffset(1.25, "Y");
  //if(min!=0.0)h->SetMinimum(min);
  c->SetRightMargin(0.125);
  c->SetFillColor(0);
  c->SetBorderMode(0);
  //	if(opt)c->SetLogz();
  pP.histogram = h;
  pP.canvas = c;
  delete [] yBins;
  return pP;
}

TFplot WTSpectrum(WSeries<double> &w,int opt=0, int pal=1, double t1=0., double t2=0.,char* copt)
{
   if(w.isWDM())return WDMPlot(w, opt, pal, t1, t2, copt);
  
  TFplot pP; null(pP);
  float x;
  double rate=w.rate();

  t1 = t1==0. ? w.start() : t1;
  t2 = t2==0. ? w.start()+w.size()/rate : t2;

  int ni = 1<<w.pWavelet->m_Level;
  int nb = int((t1-w.start())*rate/ni);
  int nj = int((t2-t1)*rate)/ni;
  int ne = nb+nj;
  double freq=w.frequency(1)-w.frequency(0);
  rate = rate/ni;

//  cout<<rate<<endl;

  pP.histogram=new TH2F("WTS","", nj, t1-w.start(), t2-w.start(), ni, 0., freq*ni);
  pP.histogram->SetXTitle("time, sec");
  pP.histogram->SetYTitle("frequency, Hz");

  Int_t colors[30]={101,12,114,13,115,14,117,15,16,17,166,18,19,
                   167,0,0,167,19,18,166,17,16,15,117,14,115,13,114,12,101};
  if(pal==0)gStyle->SetPalette(30,colors);
  else gStyle->SetPalette(1,0);

  wavearray<double> wl;
  double avr,rms;

  if(opt==0)
  for(int i=0;i<ni;i++)
  { 
	w.getLayer(wl,i);	
	for(int j=nb;j<ne;j++)
	{
	   x=wl.data[j];
	   pP.histogram->Fill(j/rate,(i+0.5)*freq,x);
	}
  }

  if(opt==1)
  for(int i=0;i<ni;i++)
  { 
	w.getLayer(wl,i);	
	for(int j=nb;j<ne;j++)
	{
	   x=wl.data[j]*wl.data[j];
	   pP.histogram->Fill(j/rate,(i+0.5)*freq,x);
	}
  }

  if(opt==2)
  for(int i=0;i<ni;i++)
  { 
	w.getLayer(wl,i);	
	for(int j=nb;j<ne;j++)
	{
	   x = fabs(wl.data[j]);
//	   if(x>0.1) x=TMath::Log(x);
	   pP.histogram->Fill(j/rate,(i+0.5)*freq,x);
	}
  }

  if(opt==4)
  for(int i=0;i<ni;i++)
  { 
	w.getLayer(wl,i);	
	for(int j=nb;j<ne;j++)
	{
	   x=(wl.data[j] > 0.) ? 1. : -1.;
	   pP.histogram->Fill(j/rate,(i+0.5)*freq,x);
	}
  }

  if(opt==5)
  for(int i=0;i<ni;i++)
  { 
	w.getLayer(wl,i);	
	for(int j=nb;j<ne;j++)
	{
	   x=(wl.data[j] > 0.) ? 1. : -1.;
	   if(wl.data[j] == 0.) x=0.;
	   pP.histogram->Fill(j/rate,(i+0.5)*freq,x);
	}
  }

  double sum = 0.;
  int nsum = 0;

  if(opt==3){
  for(int i=0;i<ni;i++)
  { 
	w.getLayer(wl,i);
	wl.getStatistics(avr,rms);
	for(int j=nb;j<ne-0;j++)
	{
	   x=(wl.data[j]-avr)/rms;
           x*=x;
	   if(x>1){ sum += x; nsum++; }
	   pP.histogram->Fill(j/rate,(i+0.5)*freq,x);
	}
  }
//  cout << "nsum = " << nsum << "  chi2/nsum = " << sum/nsum << "\n";
  }
  
  pP.canvas = new TCanvas("WTSpectrum","", 60, 60,600,400);
  pP.canvas->SetBorderMode(0);
  pP.canvas->SetFillColor(0);
  
  gPad->SetBorderMode(0);
  gPad->SetFillColor(0);
  pP.histogram->SetStats(kFALSE);
  pP.histogram->SetTitleOffset(1.3,"Y");
  pP.histogram->Draw(copt);

  return pP;
}

TFplot SMSpectrum(skymap &sm,int opt=0,int pal=1)
{

  TFplot pP; null(pP); 
  float x;

  int ni = sm.size(0);         // number of theta layers 
  int nj = 0;                  // number of phi collumns

  for(int i=1; i<=ni; i++) if(nj<sm.size(i)) nj = sm.size(i);

  double t1 = sm.theta_1;
  double t2 = sm.theta_2;
  double p1 = sm.phi_1;
  double p2 = sm.phi_2;
  double dt = ni>1 ? (t2-t1)/(ni-1) : 0.; 
  double dp = nj>0 ? (p2-p1)/nj : 0.; 

  pP.histogram=new TH2F("WTS","", nj,p1,p2, ni,90.-t2,90.-t1);
  pP.histogram->SetXTitle("phi, deg.");
  pP.histogram->SetYTitle("theta, deg.");

  Int_t colors[30]={101,12,114,13,115,14,117,15,16,17,166,18,19,
                   167,0,0,167,19,18,166,17,16,15,117,14,115,13,114,12,101};
  if(pal==0)gStyle->SetPalette(30,colors);
  else gStyle->SetPalette(1,0);

  wavearray<double> wl;
  double avr,rms;
  double theta, phi;

//  if(opt==0)
  for(int i=0; i<ni; i++) { 
    theta = (t2+t1)/2.+(i-ni/2)*dt;
    for(int j=0; j<nj; j++) {
      phi = (p2+p1)/2.+(j-nj/2)*dp;
      pP.histogram->Fill(phi,90.-theta,sm.get(theta,phi));
    }
  }

  pP.canvas = new TCanvas("SMSpectrum","", 50, 50,800,600);
  pP.canvas->SetBorderMode(0);
  pP.canvas->SetFillColor(0);
  pP.canvas->SetGridx();
  pP.canvas->SetGridy();
  
  gPad->SetBorderMode(0);
  gPad->SetFillColor(0);
  pP.histogram->SetStats(kFALSE);
  pP.histogram->Draw("COLZ");
  
  return pP;
}


void WLSpectrum(WSeries<double> &w,int opt=2,int pal=0)
{
  TFplot pP; null(pP);
  float x;

  int nj=w.size()>>1;
  int ni=w.pWavelet->m_Level;
  double rate=w.rate()/2;
  Float_t* at = new Float_t[nj+1];
  Float_t* af = new Float_t[ni+1];
  float df = rate/pow(2,ni-1);
  af[0] = 0.;
  for(int i=1; i<=ni; i++){
    af[i] = df;
    df *= 2.;
  }

  for(int i=0; i<=nj; i++){
    at[i] = i/rate;
  }

  pP.histogram=new TH2F("WLS", "", nj, at, ni, af);
//  pP.histogram=new PP.HISTOGRAMF("WLS", "", nj, 0., nj/rate, ni, 0, nj);
// PP.HISTOGRAMF(const char* name, const char* title, Int_t nbinsx,
// Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup)
  pP.histogram->SetXTitle("time");
  pP.histogram->SetYTitle("frequency, Hz");
  Int_t colors[30]={101,12,114,13,115,14,117,15,16,17,166,18,19,
                   167,0,0,167,19,18,166,17,16,15,117,14,115,13,114,12,101};
  if(pal==0)gStyle->SetPalette(30,colors);
  else gStyle->SetPalette(1,0);

  wavearray<double> wl;
  int lay=0;
	w.getLayer(wl,0);
	int nj1=wl.size();	
	int kf=nj/nj1;	
//	cout<<"layer="<<0<<" number="<<nj1;
	int j1=0;
	if(opt==0)
	for(int j=0;j<nj1;j++)
	{
	   x=wl.data[j];
	   for(int k=0;k<kf;k++)
		   pP.histogram->Fill((j1++)/rate,lay,x);
	}
	if(opt==1)
	for(int j=0;j<nj1;j++)
	{
	   x=wl.data[j]*wl.data[j];
	   for(int k=0;k<kf;k++)
		   pP.histogram->Fill((j1++)/rate,lay,x);
	}
	if(opt==2)
	for(int j=0;j<nj1;j++)
	{
	   x=fabs(wl.data[j]);
	   for(int k=0;k<kf;k++)
		   pP.histogram->Fill((j1++)/rate,lay,x);
	}

	df = rate/2.;

	if(opt==0)
	   for(int i=ni-1;i>0;i--)
	   {
	      w.getLayer(wl,i);
	      nj1=wl.size();
	      kf=nj/nj1;
//cout<<"kf="<<kf<<" ";	
//	cout<<"layer="<<lay<<" number="<<nj1;
	      j1=0;
	      for(int j=0;j<nj1;j++)
	      {
		 x=wl.data[j];
		 for(int k=0;k<kf;k++)
		    pP.histogram->Fill((j1++)/rate,df,x);
	      }
	      df /= 2.;
	   }
	if(opt==1)
	   for(int i=ni-1;i>0;i--)
	   {
	      w.getLayer(wl,i);
	      nj1=wl.size();
	      kf=nj/nj1;
//cout<<"kf="<<kf<<" ";	
//	cout<<"layer="<<lay<<" number="<<nj1;
	      j1=0;
	      for(int j=0;j<nj1;j++)
	      {
		 x=wl.data[j]*wl.data[j];
		 for(int k=0;k<kf;k++)
		    pP.histogram->Fill((j1++)/rate,df,x);
	      }
	      df /= 2.;
	   }
	if(opt==2)
	   for(int i=ni-1;i>0;i--)
	   {
	      w.getLayer(wl,i);
	      nj1=wl.size();
	      kf=nj/nj1;//cout<<"kf="<<kf<<" ";	
//	cout<<"layer="<<lay<<" number="<<nj1;
	      j1=0;
	      for(int j=0;j<nj1;j++)
	      {
		 x=log(wl.data[j]*wl.data[j]);
		 for(int k=0;k<kf;k++)
		    pP.histogram->Fill((j1++)/rate,df,x);
	      }
	      df /= 2.;
	   }

	gPad->SetBorderMode(0);
	gPad->SetFillColor(0);
	pP.histogram->SetStats(kFALSE);
	pP.histogram->Draw("COLZ");
}












