/********************************************************/
/* Wavelet Analysis Tool                                */
/* file libcor.C                                        */
/********************************************************/

WaveData prob(WaveData &td, int n=0)
{
  if (n <= 0) n = td.N;
  int ns = td.N/n;
  if(ns==0) n=td.N;
  int nm,np;

  WaveData pr(n);
  pr.Rate=1;
  pr=0.;

  for (int i=1;  i<=n; i++) {
     nm=np=0;
     for (int j=0;  i+j<td.N ; j++) {
	if(td.data[i+j]*td.data[j]<0.)
	   nm++;
	else
	   np++;
     }
     pr.data[i-1] = double(np-nm)/(np+nm);
  }
  return pr;

}











