//
// This example show how the resampling helps in the reconstruction of the unbias maximum position
// Author : Gabriele Vedovato

#define HEALPIX_ORDER 4			// initial skymap resolution
#define RESAMPLING 4			// resampling from initial resolution

#define DISPLAY_FULL_SKYMAP    

void ResampleSkymapFunction() {

  #include <complex>

  #define N 5		// set the frequency of the injected function
  #define OFFSET 0.3	// offset in degrees of the injected function

  skymap sm(int(HEALPIX_ORDER));		// initial skymap resolution
  int L = sm.size();
  // fill skymap with sin^2 + cos^2 funcion
  for(int l=0;l<L;l++) {
    double th = sm.getTheta(l);
    double ph = sm.getPhi(l); 
    double p = sin(TMath::TwoPi()*(th+45-OFFSET)/180.*N);
    double q = cos(TMath::TwoPi()*(ph-OFFSET)/180.*N/2.);
    //sm.set(l,p*p);
    //sm.set(l,q*q);
    sm.set(l,p*p+q*q);
  }
  // resample skymap to the new resolution
  if(RESAMPLING!=0) {
    int order = sm.getOrder();
    cout << "resampling : " << order << " to " << order+RESAMPLING << endl;
    sm.resample(int(order+RESAMPLING));
  }

  int nlmax = 256;
  wat::Alm alm = sm.getAlm(nlmax);

  // compute power in the sph har decomposition domain
  double norm=0;
  for(int l=0;l<=alm.Lmax();l++) { 
    int limit = TMath::Min(l,alm.Mmax());
    for (int m=0; m<=limit; m++) {
      double mod = pow(alm(l,m).real(),2)+pow(alm(l,m).imag(),2);
      norm+= m==0 ? mod : 2*mod;
    }
  }
  norm = norm/(4*TMath::Pi());
  cout << "norm : " << norm << endl;

  // compute power using the skymap pixels
  double en=0;
  int L = sm.size();
  for(int i=0;i<L;i++) en+=pow(sm.get(i),2);
  double dw = 1./L;
  cout << "EN " << en*dw << endl;

  // pixel resolution
  double ds = 4*TMath::Pi()*(180.*180./(TMath::Pi()*TMath::Pi()))/L;
  cout << "L = " << L << endl;
  cout << "sqrt(ds) = " << sqrt(ds) << endl;

  // rearch maximum around th=0, ph=0
  #define RADIUS 10

  // find max
  double max=-100;
  int imax=0;
  for(int l=0;l<L;l++) {
    double th = sm.getTheta(l);
    double ph = sm.getPhi(l);
    if(ph>180) ph-=360;
#ifdef DISPLAY_FULL_SKYMAP    
    if(sqrt(pow(th-90,2)+pow(ph-0,2))>RADIUS) continue;
#else
    if(sqrt(pow(th-90,2)+pow(ph-0,2))>RADIUS) {sm.set(l,0);continue;}
#endif
    if(sm.get(l)>max) {max=sm.get(l);imax=l;}
  }
  double mth = sm.getTheta(imax);
  double mph = sm.getPhi(imax);
  // print unbias maximum position
  cout << "imax " << imax << endl;
  cout << "max  " << max << endl;
  cout << "mth  " << mth << endl;
  cout << "mph  " << mph << endl;

  gskymap* gsm = new gskymap(sm);
  gsm->Draw();
}
