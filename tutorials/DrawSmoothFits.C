//
// this example show how to applies filters to the healpix skymap
// Author : Gabriele Vedovato
// fitsName = fits file name


//#define SMOOTHING 0.4 	// gaussian filter with radius = 0.4 degrees
//#define MEDIAN 0.4		// median filter with radius = 0.4 degrees
//#define ALMS			// setting cuts using spherical harmonic coefficients 

#define NLMAX 256		// l max of spherical harmonic coefficients
#define NLMAX_THR 256		// l threshold of spherical harmonic coefficients

#define GETSKYAREA		// compute the sky area error from probability.fits skymap

#define TH_INJ (90-30)		// theta source direction
#define PH_INJ 60		// phy source direction

void DrawSmoothFits(TString fitsName) {

  #include <complex>

  skymap sm(fitsName.Data());
#ifdef SMOOTHING
  int nlmax = 256;
  int num_iter = 0;
  double fwhm = SMOOTHING; // degrees
  sm.smoothing(fwhm, nlmax, num_iter);
#endif
#ifdef MEDIAN
  double radius = MEDIAN;  // degrees
  sm.median(radius);
#endif
#ifdef ALMS
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
  // set alm=0 for l>NLMAX_THR
  wat::Alm alm = sm.getAlm(NLMAX);
  for(int l=0;l<=alm.Lmax();l++) {
    int limit = TMath::Min(l,alm.Mmax());
    for (int m=0; m<=limit; m++) if(l>NLMAX_THR) alm(l,m)=0;
    //for (int m=0; m<=limit; m++) cout << alm(l,m).real() << " " << alm(l,m).imag() << endl;
  }
  sm.setAlm(alm);
#endif

  gskymap* gsm = new gskymap(sm);
  gsm->Draw();

  int L = sm.size();

  double ds = 4*TMath::Pi()*(180.*180./(TMath::Pi()*TMath::Pi()))/L;

  cout << "L = " << L << endl;
  cout << "sqrt(ds) = " << sqrt(ds) << endl;

#ifdef GETSKYAREA

  double th_inj = TH_INJ;
  double ph_inj = PH_INJ;

  int l_inj = sm.getSkyIndex(th_inj, ph_inj);

  int* index = new int[L];
  double* prob = new double[L];
  for(int l=0; l<L; l++) prob[l] = sm.get(l);
  TMath::Sort(L, prob, index, true);

  int l_max=0;
  double vol = 0.;
  for(int l=0; l<L; l++) {
    vol += ds;
    if(l<20) cout << l << " " << index[l] << " " << prob[index[l]] << endl;
    if(index[l] == l_inj) {l_max=l;break;}
  }
  double eA = sqrt(vol);
  cout << "eA = " << eA << " l_inj " << l_inj << " l_max " << l_max << endl;

#endif

  // find max
  double max=-100;
  int imax=0;
  for(int l=0;l<L;l++) if(sm.get(l)>max) {max=sm.get(l);imax=l;}
  double th = sm.getTheta(imax);
  double ph = sm.getPhi(imax);
  CwbToGeographic(ph,th,ph,th);
  gsm->DrawMarker(ph,th, 29, 2.0, kBlack);

  double th_inj = TH_INJ;
  double ph_inj = PH_INJ;
  CwbToGeographic(ph_inj,th_inj,ph_inj,th_inj);
  gsm->DrawMarker(ph_inj,th_inj, 29, 2.0, kWhite);

}
