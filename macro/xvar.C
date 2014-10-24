/********************************************************/
/* Wavelet Analysis Tool                                */
/* file xvar.C                                          */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/* Macro calculates variance based on autocorrelation   */
/* function. k points calculated for different          */
/* collection time.                                     */
/*                                                      */
/* WAT ver. >= 2.1 required				*/
/********************************************************/

wavearray<double>* xvar( wavearray<double> &td1, int nn, int k = 256)
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

// sum over upper triangle of covariance matrix,
// i*m-j is the length of j-th subdiagonal of 
// band-diagonal matrix (i*m)x(i*m)

    for (int j = 1;  j <  m*i; j++) 
      s += (i*m-j)*td1[j];

// v is variance divided by matrix size N=m*i
// v = (2*sum_upper_triangle + N*diagonal_element)/N;
    (*v)[i-1] = 2.*s/(m*i)+1.;
  }

  return v;

}

wavearray<double>* xvar( wavearray<float> &td1, int nn, int k = 256 )
{
   int n = td1.size();
   wavearray<double> dd1(n);

   dd1.Rate = td1.Rate;

   for (int i=0; i<n; i++)
   {
     dd1[i] = double(td1[i]);
   }

   return xvar(dd1, nn, k);

}

