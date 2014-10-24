/********************************************************/
/* Wavelet Analysis Tool                                */
/* file xcor.C                                          */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/* Macro calculates averaged cross correlation function */
/* for signals represented by two wavearrays.           */
/*                                                      */
/* WAT ver. >= 2.1 required				*/
/********************************************************/

#define PI 3.141592653589793 

wavearray<double>* xcor(
                         wavearray<double> &td1,
                         wavearray<double> &td2,
                         int nn=0,
                         double w=1.)
{
  int n = td1.size();
  if (nn <= 0) nn = n;
  double x, y;
  int nn2 = nn/2;

  if (td1.rate() <= 0.) {
    cout <<" xcor() error: invalid sample rate ="<< td1.Rate <<"\n'";
    return NULL;
  }

  if (td1.rate() != td2.rate()) {
    cout <<" xcor() error: sample rates are not equal, f1="<< td1.Rate 
    <<", f2="<< td2.Rate<<endl;
    return NULL;
  }

  int ns = n/nn;
  if (ns == 0) {
    cout << "xcor() error: data too short for specified window length="
         << nn << "\n";
    return NULL;
  }

  wavearray<double> xs1(nn);
  wavearray<double> xs2(nn);
  wavearray<double>* xs = new wavearray<double>(nn);
  xs->Rate=td1.Rate;
  xs1.Rate=td1.Rate;
  xs2.Rate=td1.Rate;

  double enrg1=0.;
  double enrg2=0.;
  *xs = 0.;
 
  for (int j=0;  j <= (n - nn) ; j+=nn) {

     xs1.cpf(td1, nn, j);
     xs1.FFT();
     xs2.cpf(td2, nn, j);
     xs2.FFT();

// set to zero the power density at zero frequency
//   xs1[0]=0.;
//   xs2[0]=0.;

// calculate power cross-spectrum from Fourier coefficients
// nn assumed to be even number

   for (int i = 0; i < nn2; i++) {
      x = xs1[2*i]*xs2[2*i] + xs1[2*i + 1]*xs2[2*i + 1];
      y = xs1.data[2*i]*xs2.data[2*i + 1] - xs2.data[2*i]*xs1.data[2*i + 1];
      enrg1 += xs1.data[2*i]*xs1.data[2*i]+xs1.data[2*i + 1]*xs1.data[2*i + 1];
      enrg2 += xs2.data[2*i]*xs2.data[2*i]+xs2.data[2*i + 1]*xs2.data[2*i + 1];
      xs->data[2*i] += x;
      xs->data[2*i + 1] += y;
    }

  }

   (*xs) *= nn/td1.Rate/ns;
   enrg1 *= nn/td1.Rate/ns;
   enrg2 *= nn/td1.Rate/ns;
   (*xs) *= (w <= 0.) ? 0.5/sqrt(enrg1*enrg2) : w; 

// set to zero the power density at zero frequency
//   (*xs)[0]=0.;

   xs->FFT(-1);
   double norm = 0.5/sqrt(enrg1*enrg2);
   cout << "norm = " << norm << "  r(0)="<<xs->data[0]<<"\n";

  if ( norm != 0. ) (*xs) *= norm;
  return xs;

}

wavearray<double>* xcor(
                         wavearray<float> &td1,
                         wavearray<float> &td2,
                         int nn=0,
                         double w=1.)
{
   if ( td1.size() != td1.size() ) 
     cout <<" xcor: error - data arrays must be the same size"<<endl;

   int n = td1.size();
   wavearray<double> dd1(n);
   wavearray<double> dd2(n);

   dd1.Rate = td1.Rate;
   dd2.Rate = td2.Rate;

   for (int i=0; i<n; i++)
   {
     dd1[i] = double(td1[i]);
     dd2[i] = double(td2[i]);
   }

   return xcor(dd1, dd2, nn, w);

}

