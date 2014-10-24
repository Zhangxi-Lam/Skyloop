/********************************************************/
/* Wavelet Analysis Tool                                */
/* file lagcor.C                                          */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/* Macro calculates the cross correlation as a function */
/* lag time.                                            */
/********************************************************/

wavearray<double> lagcor(
   wavearray<double> &td1,
   wavearray<double> &td2,
   int n=0)
{
   int k;
   double ave, rms;
   wavearray<double> a1;
   wavearray<double> a2;
   wavearray<double> out(2*n+1);

   for(int i=-n; i<=n; i++){

      if(i>0){
	 k = td1.size()-2*i;
	 if(k>td2.size()) k = td2.size();
      }
      else{
	 k = td2.size()+2*i;
	 if(k>td1.size()) k = td1.size();
      }
      
      a1.resize(k);
      a2.resize(k);
      a1.rate(td1.rate());
      a2.rate(td2.rate());

      if(i>0){
	 a1.cpf(td1,k,i);
	 a2.cpf(td2,k);
      }
      else{
	 a1.cpf(td1,k);
	 a2.cpf(td2,k,-i);
      }
      
      a1 *= a2;
      a1.getStatistics(ave,rms);

      out.data[i+n] = ave;
   }
   return out;
}











