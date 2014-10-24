/*-------------------------------------------------------
 * Package:     Wavelet Analysis Tool
 * File name:   AddSignals.C
 *
 * This macro file is for ROOT interactive environment.
 * Adds white noise with Gauss distribution and RMS=v. 
 *-------------------------------------------------------
*/

#define PI 3.141592653589793
void AddGaus()
{
  cout
  <<"*****************************************************************\n"
  <<" HELP: Macro Gauss adds random numbers to the existent data in\n"
  <<" specified object. Added data simulate white Gauss noise.\n\n"
  <<" Call: AddGauss(td, r)\n\n"
  <<" td - is either object or pointer to object wavearray\n"
  <<" r - is root mean square of Gauss distribution,\n"
  <<"*****************************************************************\n";
}

void AddSine()
{
  cout
  <<"*****************************************************************\n"
  <<" HELP: Macro AddSine() adds data simulating monohromatic signal\n"
  <<" to the existent data in specified\n"
  <<" object.\n\n"
  <<" Call: AddSine(td, a, f, phi)\n\n"
  <<" td - is either object or pointer to object wavearray\n"
  <<" a - is the amplitude of added signal,\n"
  <<" f - is the frequency of added signal,\n"
  <<" phi - is the phase shift of added signal,\n" 
  <<"*****************************************************************\n";
}

void AddChirp()
{
  cout
  <<"*****************************************************************\n"
  <<" HELP: Macro AddChirp() adds data simulating chirping signal\n"
  <<" to the existent data in specified\n"
  <<" object.\n\n"
  <<" Call: AddChirp(td, m1, m2, a, d)\n\n"
  <<" td - is either object or pointer to object wavearray\n"
  <<" m1 - mass of first component,\n"
  <<" m2 - mass of second component,\n"
  <<" f  - Starting frequency,\n" 
  <<" d  - Distance from source in Mpc,\n"
  <<" Note: simulation exists when GW frequency reaches LSO frequency.\n"
  <<"*****************************************************************\n";
}


void AddGaus(wavearray<double> &td, double v)
{
  int n=td.size();
  double r;
  for (int i=0; i < n; i++){
//     r = 2.*(gRandom->Rndm(11)-0.5);
//     if(r<0) r=-1;
//     else r=1.;
     td.data[i] += v*(gRandom->Gaus(0.,1.));
  }
//  for (int i=0; i < n; i++) td.data[i] += v*2.*(gRandom->Rndm(11)-0.5);
}

void AddGaus(wavearray<double> *td, double v)
{ AddGauss(*td, v); }

void AddPoisson(wavearray<double> &td, double v)
{
  int n=td.size();
  double mu=v;
  double x;
  int m,j,i=0;
  while(i<n){
     m=gRandom->Poisson(mu);
//     m=mu;
     x=gRandom->Rndm(11)-0.5;
//     if(x<=0.) x=-1.;
//     if(x>0.)  x= 1.;
     for (j=i; j<i+m && j<n; j++) td.data[j] += x;
     i+=m;
  }
}

void AddSine(wavearray<double> &td, double a, double f, double phi0=0.)
{
  if (td.rate() <= 0.) {
    cout <<" AddSine error: invalid sampling rate ="<< td.rate() <<"\n";
    return ;
  }

  int n=td.size();
  double phi, dphi;
  phi = phi0;
  dphi = 2.*PI*f/td.rate();

  for (int i=0; i < n; i++) {
    td[i] += a*sin(phi);
    phi += dphi;
  }
}

void AddSine(wavearray<double> *td, double a, double f, double phi0=0.)
{ AddSine(*td, a, f, phi0); }

void AddChirp(wavearray<double> &td, double m1, double m2, double f0, double d)
{
  if (td.rate() < 2048.) {
    cout <<" AddChirp error: invalid sampling rate ="<< td.rate() <<", should be 2048 Hz of higher.\n";
    return ;
  }
  d*=3.08567758*pow(10,22);
  m1*= 1.989*pow(10,30);
  m2*= 1.989*pow(10,30);

  int n=td.size();
  double f, df, A, phi=0, deltat=1/td.rate();
  double G=6.67384*pow(10,-11), c=3*pow(10,8), Mt=m1+m2, Mc=pow(m1*m2,.6)/pow(Mt,.2);
  
  for (int i=0; i < n; i++) {
	 f = pow((8./3.)*((3./8.)*pow(f0,-(8./3.))-(96./5.)*pow(PI,(8./3.))*pow(G*Mc/(c*c*c),(5./3.))*deltat), -(3./8.));
//	 cout<<f<<" "<<phi<<" "<<i<<endl;
	 df = (96/5.) * pow(PI,8./3.) * pow((G*Mc)/(c*c*c),5./3.) * pow(f, 11./3.);
	 phi += 2*PI*(f*deltat+.5*df*deltat*deltat);
	 A = 4*(G/(c*c))*(Mc/d)*pow((G/(c*c*c))*PI*f*Mc,2./3.);
	 td[i] += A*cos(phi);
	 if(f>4400.*1.989*pow(10,30)/Mt)  { cout<<"Chirp ends at frequency "<<f<<" and has length "<<i*deltat<<"s"<<endl; break; }
	 f0=f;     
  }
  if(f<=4400.*1.989*pow(10,30)/Mt)  cout<<"Chirp ends at frequency "<<f<<" and has length "<<i*deltat<<"s"<<endl;
}

void AddChirp(wavearray<double> *td, double a, double f, double phi0=0.)
{ AddChirp(*td, m1, m2, f0, d); }

