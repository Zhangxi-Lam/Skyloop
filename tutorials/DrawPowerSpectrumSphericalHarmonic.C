//
// this example draw power spectrum of the spherical harmonic coefficients
// Author : Gabriele Vedovato
// fitsName = fits file name


//#define DELAY		// use as input a delay skymap between 2 detectors
#define SRC_FREQ 200	// main frequency of input signal
#define SRC_TH   90     // source theta
#define SRC_PH   0      // source phy

//#define FP_DPF	// use as input the Fp component of network antenna pattern in the DPF
//#define FX_DP	        // use as input the Fx component of network antenna pattern in the DPFF
#define FITS		// use as input the fits skymap (fitsName = fits file name)
//#define ALMS		// use as imput a skymap with user defined spherical harmonic coefficients
//#define PIXEL		// use as input a user defined skymap (pixel defined by user) 

#define PSPHA 1.	// power spectrum threshold 
//#define PSPHA 0.999999
//#define PSPHA 0.99

#define NLMAX 256	// l max of the spherical harmonic coefficients
//#define NLMAX 512
//#define NLMAX 1024

#define HEALPIX_ORDER 7	// healpix skymap order

void DrawPowerSpectrumSphericalHarmonic(TString fitsName="") {

  #include <complex>

  TString ifo[3] = {"L1","H1","V1"};
  gnetwork* gNET = new gnetwork(3,ifo);
  detector* pD[3];
  for(int i=0;i<3;i++) pD[i] = gNET->getifo(i);

  double frequency = SRC_FREQ;
  double sph = SRC_PH;
  double sth = SRC_TH;

  skymap sm(int(HEALPIX_ORDER));
  int rings=1;
  for(int i=0;i<=HEALPIX_ORDER;i++) rings=2*rings+1;
  cout << "rings " << rings << endl;
  if(NLMAX>(rings+1)/2) {
    cout << "Error : helpix resolution is " << HEALPIX_ORDER 
         << " number of rings " << rings << endl;
    cout << " NLMAX=" << NLMAX << " must be <= (rings+1)/2=" << (rings+1)/2 << endl;
    exit(1);
  }
  int L = sm.size();
  for(int l=0;l<L;l++) {
    double th = sm.getTheta(l);
    double ph = sm.getPhi(l);

#ifdef DELAY
    double ifo0d1_pos_delay=gNET->GetDelay(ifo[0], ifo[1], sph, sth);
    double delay01=pD[0]->getTau(th,ph)-pD[1]->getTau(th,ph);
    delay01-=ifo0d1_pos_delay;
    if(frequency>0) {
      double fdelay=fabs(fmod(delay01,0.5/frequency));
      int idelay = int((fabs(delay01)-fdelay)/(0.5/frequency));
      delay01 = idelay%2==0 ? fdelay : (0.5/frequency)-fdelay;
    }
    double ifo1d2_pos_delay=gNET->GetDelay(ifo[1], ifo[2], sph, sth);
    double delay12=pD[1]->getTau(th,ph)-pD[2]->getTau(th,ph);
    delay12-=ifo1d2_pos_delay;
    if(frequency>0) {
      double fdelay=fabs(fmod(delay12,0.5/frequency));
      int idelay = int((fabs(delay12)-fdelay)/(0.5/frequency));
      delay12 = idelay%2==0 ? fdelay : (0.5/frequency)-fdelay;
    }
    double ifo0d2_pos_delay=gNET->GetDelay(ifo[0], ifo[2], sph, sth);
    double delay02=pD[0]->getTau(th,ph)-pD[2]->getTau(th,ph);
    delay02-=ifo0d2_pos_delay;
    if(frequency>0) {
      double fdelay=fabs(fmod(delay02,0.5/frequency));
      int idelay = int((fabs(delay02)-fdelay)/(0.5/frequency));
      delay02 = idelay%2==0 ? fdelay : (0.5/frequency)-fdelay;
    }
    //sm.set(l,delay01);      
    sm.set(l,delay01+delay12);      
    //sm.set(l,delay01+delay12+delay02);      
#endif

    //sm.set(l,gNET->GetDelay("L1","V1",ph,th));   //     
    //sm.set(l,gNET->GetDelay("H1","L1",ph,th));   //     
#ifdef FX_DPF
    sm.set(l,gNET->GetAntennaPattern(ph, th, 0, 0));   // Fx    
#endif
#ifdef FP_DPF
    sm.set(l,gNET->GetAntennaPattern(ph, th, 0, 1));   // Fp    
#endif
  }

#ifdef FITS
  skymap smf(fitsName.Data());
  sm = smf;
  int order = sm.getOrder();
  // get number of rings
  int rings=1;
  for(int i=0;i<=order;i++) rings=2*rings+1;
  cout << "rings " << rings << endl;
  if(NLMAX>(rings+1)/2) {
    cout << "Error : helpix resolution is " << order 
         << " number of rings " << rings << endl;
    cout << " NLMAX=" << NLMAX << " must be <= (rings+1)/2=" << (rings+1)/2 << endl;
    exit(1);
  }
//  wat::Alm galm = sm.getAlm(NLMAX);
//  galm(0,0)=complex<double>(1,0);
//  sm.setAlm(galm);
#endif

#ifdef PIXEL
  sm.set(100000,1);
#endif

#ifdef ALMS
  int nlmax = NLMAX;
  wat::Alm alm(nlmax,nlmax);
  //alm(0,0)=complex<double>(1,0);
  alm(3,2)=complex<double>(1,0);
  sm.setAlm(alm);
#endif

  cout << endl;
  wat::Alm oalm = sm.getAlm(NLMAX);
  double norm=0;
  int lmax=0;
  double psmax=0;
  for(int l=0;l<=oalm.Lmax();l++) {
    int limit = TMath::Min(l,oalm.Mmax());
    double ps=0;
    for (int m=0; m<=limit; m++) {
      double mod = pow(oalm(l,m).real(),2)+pow(oalm(l,m).imag(),2);
      norm+=mod;
      ps+=mod;
    }
    if(ps>psmax) {psmax=ps;lmax=l;}
  }
  cout << "lmax : " << lmax << endl;

  int n1=0;
  double x1[NLMAX+1];
  double y1[NLMAX+1];
  int n2=0;
  double x2[NLMAX+1];
  double y2[NLMAX+1];
  for(int l=0;l<=oalm.Lmax();l++) {
    double ps=0;
    int limit = TMath::Min(l,oalm.Mmax());
    for (int m=0; m<=limit; m++) {
      double mod = pow(oalm(l,m).real(),2)+pow(oalm(l,m).imag(),2);
      ps+=mod;
    }
    if(l%2) {x1[n1]=l;y1[n1]=ps/norm;n1++;}
    else    {x2[n2]=l;y2[n2]=ps/norm;n2++;}
    //cout << l << " " << y[l] << endl;
  }
  TCanvas* c = new TCanvas(); 
  c->SetLogy();
  TGraph* gr1 = new TGraph(n1,x1,y1);
  gr1->SetMarkerColor(kBlack);
  gr1->SetLineColor(kBlack);
  TGraph* gr2 = new TGraph(n2,x2,y2);
  gr2->SetMarkerColor(kRed);
  gr2->SetLineColor(kRed);
  //gr->GetHistogram()->GetYaxis()->SetRangeUser(0,1e-4);
  TMultiGraph* mg = new TMultiGraph();
  mg->Add(gr2);
  mg->Add(gr1);
  mg->Draw("alp");
//return;

  cout << "norm : " << norm << endl;
  double snorm=0;
  wat::Alm ialm(NLMAX,NLMAX);

  double pspha = PSPHA;
  if(pspha<=1) {
    int N = oalm.Lmax()*(2*oalm.Mmax()+1);

    double* alm_a = new double[N];
    int* alm_l = new int[N];
    int* alm_m = new int[N];
    N=0;
    for(int l=0;l<=oalm.Lmax();l++) {
      int limit = TMath::Min(l,oalm.Mmax());
      for (int m=0; m<=limit; m++) {
        alm_a[N] = pow(oalm(l,m).real(),2)+pow(oalm(l,m).imag(),2);
        alm_l[N] = l;
        alm_m[N] = m;
        N++;
      }
    }

    int* index = new int[N];
    TMath::Sort(N, alm_a, index, true);

    double norm=0;
    for(int i=0;i<N;i++) norm+=alm_a[index[i]];
    for(int i=0;i<N;i++) alm_a[index[i]]/=norm;


    double cum=0;
    int icum=0;
    for (int i=0;i<N;i++) {
      cum+=alm_a[index[i]];
      if(cum>pspha) {icum=i+1;break;}
    }
    for(int i=0;i<icum;i++) {
      int ll = alm_l[index[i]];
      int mm = alm_m[index[i]];
      ialm(ll,mm)=complex<double>(oalm(ll,mm).real(),oalm(ll,mm).imag());
      double mod = pow(oalm(ll,mm).real(),2)+pow(oalm(ll,mm).imag(),2);
//      cout << i << " " << ll << " " << mm << " " 
//           << oalm(ll,mm).real() << " " << oalm(ll,mm).imag() << " " << 100*mod/norm << endl;
    }
    delete [] index;
    delete [] alm_a;
    delete [] alm_l;
    delete [] alm_m;
  }

  sm=0;
/*
  for(int l=0;l<=ialm.Lmax();l++) {
    int limit = TMath::Min(l,ialm.Mmax());
//    for (int m=0; m<=limit; m++) if(l%2) ialm(l,m)=0;  // set 0 odd components
    for (int m=0; m<=limit; m++) if(!(l%2)) ialm(l,m)=0; // set 0 even components
  }
*/
  sm.setAlm(ialm);
//  sm.rotate(0,90,90);  // rotate skymap 
  gskymap* gsm = new gskymap(sm);
  gsm->Draw();
}
