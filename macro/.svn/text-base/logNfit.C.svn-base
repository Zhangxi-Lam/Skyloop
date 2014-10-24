Double_t logNfit(Double_t *x, Double_t *par)
{
/*
  double y = (log10(x[0])-par[0]);
  //  double s = y<0 ? fabs(y*exp(-y*par[2])/par[1]) : par[3]*log(1+y/(par[3]*par[1]));
  double s = y<0 ? par[1]*exp(y*par[2]) : par[3]*log(1+y/(par[3]*par[1]));
  if(y<0) return   TMath::Erfc(s)/2;
  else    return 1-TMath::Erfc(s)/2;
*/

  double y = (log10(x[0])-par[0]);
  double s = y<0 ? par[1]*exp(y*par[2]) : par[1]*exp(y*par[3]);

  if(y>0) {
    if(par[3]>1./y) {s = par[1]*par[3]*exp(1.); y = 1.;}
    y = s>0 ? fabs(y/s) : 100.; 
    return 1-TMath::Erfc(y)/2; 
  }

  if(y<0) {
    y = s>0 ? fabs(y/s) : 100.;
    return TMath::Erfc(y)/2;
  }
  return 0.5;
}
