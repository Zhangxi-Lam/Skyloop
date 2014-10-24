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
                         int nn=0)
{
  int n = td1.size();
  if (nn <= 0) nn = n;
  int nn2 = nn/2;
  if ( 2*nn2 != nn ) {
    cout <<" xcor() error: odd number of data in input "<<nn<<endl;
    return NULL;
  }

  if (td1.rate() <= 0.) {
    cout <<" xcor() error: invalid sample rate ="<< td1.rate() <<endl;
    return NULL;
  }

  if (td1.rate() != td2.rate()) {
    cout <<" xcor() error: sample rates are not equal, f1="<< td1.rate() 
    <<", f2="<< td2.rate()<<endl;
    return NULL;
  }

  int ns = n/nn;
  if (ns == 0) {
    cout << "xcor() error: data too short for specified window length="
         << nn << endl;
    return NULL;
  }

  wavearray<double> _xs1(nn);
  wavearray<double> _xs2(nn);
  wavearray<double>* _xs = new wavearray<double>(nn);
  _xs->rate(td1.rate());
  _xs1.rate(td1.rate());
  _xs2.rate(td1.rate());

  double enrg1=0.;
  double enrg2=0.;
  double xs1r, xs2r, xs1i, xs2i;
  (*_xs) = 0.;
 
  for (int j=0;  j <= (n - nn) ; j+=nn)
  {
     _xs1.cpf(td1, nn, j);
     _xs1.FFT();
     _xs2.cpf(td2, nn, j);
     _xs2.FFT();

// set to zero the power density at zero frequency
     _xs1[0]=0.;
     _xs2[0]=0.;

// calculate power cross-spectrum from Fourier coefficients
// nn assumed to be even number, in this case FFT(0) is real
// and stored in data[0],
// FFT(N/2) is real too and it is stored in data[1]

     _xs->data[0] += _xs1[0]*_xs2[0];
     _xs->data[1] += _xs1[1]*_xs2[1];

     for (int i = 1; i < nn2; i++)
     {
       xs1r = _xs1[2*i];		// real part of spectrum 
       xs1i = _xs1[2*i + 1];		// imaginary part of spectrum

       xs2r = _xs2[2*i];
       xs2i = _xs2[2*i + 1];

       enrg1 += xs1r*xs1r + xs1i*xs1i;
       enrg2 += xs2r*xs2r + xs2i*xs2i;

       _xs->data[2*i] += xs1r*xs2r + xs1i*xs2i;
       _xs->data[2*i + 1] += xs1i*xs2r - xs1r*xs2i;
     }
  }

   (*_xs) *= nn/td1.rate()/ns;
   enrg1 *= nn/td1.rate()/ns;
   enrg2 *= nn/td1.rate()/ns;
//   cout <<" Energies: "<<enrg1<<", "<<enrg2<<endl;

   _xs->FFT(-1);

   double norm = 2.*sqrt(enrg1*enrg2); // 2 is because single-side spectrum used
   if ( norm != 0. ) (*_xs) *= 1./norm;
   cout << "norm = " << norm << "  r(0)="<<_xs->data[0]<<endl;

   return _xs;
}

wavearray<double>* xcor(
                         wavearray<float> &td1,
                         wavearray<float> &td2,
                         int nn=0)
{
   if ( td1.size() != td1.size() ) 
     cout <<" xcor: error - data arrays must be the same size"<<endl;

   int n = td1.size();
   wavearray<double> dd1(n);
   wavearray<double> dd2(n);

   dd1.rate(td1.rate());
   dd2.rate(td2.rate());

   for (int i=0; i<n; i++)
   {
     dd1[i] = double(td1[i]);
     dd2[i] = double(td2[i]);
   }

   return xcor(dd1, dd2, nn);

}

