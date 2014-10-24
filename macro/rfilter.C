/*-------------------------------------------------------
 * Package:     Wavelet Analysis Tool
 * File name:   filter.C
 *
 * This macro file is for ROOT interactive environment.
 * filters sign T-F plot 
 *-------------------------------------------------------
*/

void rfilter(WSeries<double> &w)
{

   int n=w.size();
   int m=0, max=1;
   double c,x;

   wavearray<double> a0;
   wavearray<double> aa;

   int ni=1<<w.pWavelet->m_Level;
   int nj=w.size()/ni;

   w.getLayer(a0,0);	

   int nj=a0.size();

   aa = a0;
   aa = 0.;

   for(int i=0;i<ni;i++){
      w.getLayer(a0,i);	      
      aa = 0.;
      for(int j=0;j<nj;j++){
	 aa.data[j] = a0.data[j];
	 if(fabs(a0.data[j]) < fabs(a0.data[int((nj-9)*gRandom->Rndm(11)+8)]))
	    aa.data[j] = 0.;
	 if(fabs(a0.data[j]) < fabs(a0.data[int((nj-9)*gRandom->Rndm(11)+8)]))
	    aa.data[j] = 0.;
	 if(fabs(a0.data[j]) < fabs(a0.data[int((nj-9)*gRandom->Rndm(11)+8)]))
	    aa.data[j] = 0.;
	 if(fabs(a0.data[j]) < fabs(a0.data[int((nj-9)*gRandom->Rndm(11)+8)]))
	    aa.data[j] = 0.;
	 if(fabs(a0.data[j]) < fabs(a0.data[int((nj-9)*gRandom->Rndm(11)+8)]))
	    aa.data[j] = 0.;
	 if(fabs(a0.data[j]) < fabs(a0.data[int((nj-9)*gRandom->Rndm(11)+8)]))
	    aa.data[j] = 0.;
	 if(fabs(a0.data[j]) < fabs(a0.data[int((nj-9)*gRandom->Rndm(11)+8)]))
	    aa.data[j] = 0.;
      }	
      w.putLayer(aa,i);	
   }    
}






















