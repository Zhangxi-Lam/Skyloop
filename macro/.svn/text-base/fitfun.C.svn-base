double ChiSquare(double *a, double *p)
{
  double r = p[1]/2.;
//  double y = a[0]*p[2];
  double y = a[0]/2.;
  if(r<=0. || y<=0. || a[0]<=0.) return 0.;
  return p[0]*pow(a[0],r-1)*exp(-y)/TMath::Gamma(r);
}

double ChiSquareB(double *a, double *p)
{
  double r = p[2];
  double y = a[0]*p[1];
  if(r<=0. || y<=0. || a[0]<=0. || p[3]<0.) return 0.;
  return TMath::BesselI(0,a[0]*p[3])*p[0]*2.*pow(a[0],r-1)*exp(-y)/TMath::Gamma(r);
}

double LogNorm(double *a, double *p)
{
  double f = p[3]*sqrt(log(4));
         f = TMath::SinH(f)/f;
  double x = 1.+p[3]*f*(a[0]-p[1])/p[2];
         if(f<=0. || p[2]<0. || x<=0.) return 0.;
         x = log(x)*log(x)/p[3]/p[3]+p[3]*p[3]; 
  return p[0]*exp(-x/2.)/sqrt(2*PI);
}
