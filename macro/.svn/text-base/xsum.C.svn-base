/********************************************************/
/* Wavelet Analysis Tool                                */
/* file xsum.C                                          */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/* Macro calculates average of data as function of      */
/* sum length.                                          */
/*                                                      */
/* WAT ver. >= 2.1 required				*/
/********************************************************/

wavearray<double>* xsum( wavearray<double> &td1, int nn, int k = 256 )
{
  int n = td1.size();
  if (nn <= 0) nn = n;

  if (td1.rate() <= 0.) {
    cout <<" xvar() error: invalid sample rate ="<< td1.Rate <<"\n'";
    return NULL;
  }

  if (n < nn) {
    cout << "xvar() error: data too short for specified length="
         << nn << "\n";
    return NULL;
  }

  double s = 0.;
  if ( k > nn ) k = nn;
  int m = int(nn/k);
  wavearray<double>* v=new wavearray<double>(k+1);
  v->Rate=td1.Rate/m;
  (*v)[0] = 0.;
  int j = 0;
 
  for (int i = 1; i <= k; i++ )
  {

    while( j < m*i ) s += td1[j++];

    (*v)[i] = s/sqrt(double(m*i));
  }

  return v;

}

wavearray<double>* xsum( wavearray<float> &td1, int nn=0, int k = 1)
{
   int n = td1.size();
   wavearray<double> dd1(n);

   dd1.Rate = td1.Rate;

   for (int i=0; i<n; i++)
   {
     dd1[i] = double(td1[i]);
   }

   return xsum(dd1, nn, k);

}

