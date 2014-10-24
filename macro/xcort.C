/********************************************************/
/* Wavelet Analysis Tool                                */
/* file xcort.C                                         */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/* Macro calculates average correlation time based on   */
/* integral of autocorrelation function.                */
/* function. k points calculated for different          */
/* integration time.                                    */
/*                                                      */
/* WAT ver. >= 2.1 required				*/
/********************************************************/

wavearray<double>* xcort( wavearray<double> &td1, int nn, int k = 256)
{
// nn is data length to procees
// k is number of points of plot to calculate
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
  wavearray<double>* v = new wavearray<double>(k);

  if ( v )
    v->Rate = td1.Rate/m;
  else
    return NULL;

  for (int i=1; i <= k; i++ )
  {
    s = 0.;

    for (int j = 0;  j <  m*i; j++) 
      s += td1[j]/td1.Rate;

    (*v)[i-1] = s;
  }

  return v;

}

wavearray<double>* xcort( wavearray<float> &td1, int nn, int k = 256 )
{
   int n = td1.size();
   wavearray<double> dd1(n);

   dd1.Rate = td1.Rate;

   for (int i=0; i<n; i++)
   {
     dd1[i] = double(td1[i]);
   }

   return xcort(dd1, nn, k);

}

