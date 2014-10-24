/*-------------------------------------------------------
 * Package:     Wavelet Analysis Tool
 * File name:   AddPulse.C
 *
 * This macro file is for ROOT interactive environment.
 * Adds white noise with Gauss distribution and RMS=v. 
 *-------------------------------------------------------*/

void addGauss(wavearray<double> &td, double v, double u=0.)
{
  int n=td.size();
  for (int i=0; i < n; i++){
     td.data[i] += v*gRandom->Gaus(0.,1.)+u; 
  }
}

void addExp(wavearray<double> &td, double v, int M)
{
  int i,j;
  double x,y;
  int m=abs(M);
  int n=td.size();
  for (i=0; i<n; i++){
     y = 0.;
     for (j=0; j<m; j++){
	x = gRandom->Exp(v);
	if(M>0) y += x;
	else if(x>y) y=x;
     } 
     td.data[i] += y; 
  }
}

/*-------------------------------------------------------
 * Adds sin-Gauss with amplitude a, frequency f and gaussian  RMS s. 
 *-------------------------------------------------------*/

void addSGBurst(wavearray<double> &td, double a, double f, double s, double d=0.)
{
  int n=td.size();
  double r = td.rate();
  double T = double(n)/r;
  int m;
  double g, t;
  double sum = 0.;
  double delay = d; 

  m = int(6*s*r);
  if(m > n/2-1) m = n/2-2;

  for (int i=0; i < m; i++){
     t = i/r;
     g = 2*TMath::Exp(-t*t/2/s/s)*TMath::Sin(2*PI*f*t); 
     sum += g*g;
  } 
  a *= TMath::Sqrt(r/sum);

  td.data[n/2+int(delay*r)] += 0;  
  for (int i=1; i < m; i++){
     t = i/r;
     g = a*TMath::Exp(-t*t/2/s/s)*TMath::Sin(2*PI*f*t); 
     td.data[n/2+i+int(delay*r)] += g;
     td.data[n/2-i+int(delay*r)] -= g;
  } 
}

/*-------------------------------------------------------
 * Adds cos-Gauss with amplitude a, frequency f and gaussian  RMS s. 
 *-------------------------------------------------------*/

void addCGBurst(wavearray<double> &td, double a, double f, double s, double d=0.)
{
  int n=td.size();
  double r = td.rate();
  double T = double(n)/r;
  int m;
  double g, t;
  double sum = 0.;
  double delay = d; 

  m = int(6*s*r);
  if(m > n/2-1) m = n/2-2;

  for (int i=0; i < m; i++){
     t = i/r;
     g = 2*TMath::Exp(-t*t/2/s/s)*TMath::Cos(2*PI*f*t); 
     sum += g*g;
  } 
  a *= TMath::Sqrt(r/sum);

  td.data[n/2+int(delay*r)] += a;  
  for (int i=1; i < m; i++){
     t = i/r;
     g = a*TMath::Exp(-t*t/2/s/s)*TMath::Cos(2*PI*f*t); 
     td.data[n/2+i+int(delay*r)] += g;
     td.data[n/2-i+int(delay*r)] += g;
  } 
}

/*-------------------------------------------------------
 * Adds windowed Gaussian noise with amplitude a and gaussian  RMS s. 
 *-------------------------------------------------------*/

void addWGNoise(wavearray<double> &td, double a, double s)
{
  int n=td.size();
  double r = td.rate();
  double T = double(n)/r;
  int m;
  double g, t;
  double sum = 0.;
  double delay = 0.; 

  m = int(3*s*r);
  if(m > n/2-1) m = n/2-2;
  wavearray<double> gn(2*m);


  for (int i=0; i < m; i++){
     t = i/r;
     g   = TMath::Exp(-t*t/2/s/s);
     gn.data[m+i]   = g*gRandom->Gaus(0.,1.); 
     gn.data[m-i-1] = g*gRandom->Gaus(0.,1.); 
     sum += gn.data[m+i]*gn.data[m+i] + gn.data[m-i-1]*gn.data[m-i-1];
  } 
  a *= TMath::Sqrt(r/sum);

  for (int i=0; i < m; i++){
     t = i/r;
     td.data[n/2+i] += a*gn.data[m+i];
     td.data[n/2-i] += a*gn.data[m-i-1];
  } 
}
