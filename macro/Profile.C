/*-------------------------------------------------------
 * Package:     Wavelet Analysis Tool
 * File name:   Profile.C
 *
 * This macro file is for ROOT interactive environment.
 * Macro makes energy vs time profile of  wavelet scalogram.
 *-------------------------------------------------------
*/ 


void Profile(WSeries<double> &w, size_t opt, size_t col, double t1=0., double t2=0.)
{

  float x;

  t1 = t1==0. ? w.start() : t1;
  t2 = t2==0. ? w.start()+w.size()/w.rate() : t2;

  int ni = 1<<w.pWavelet->m_Level;
  int nb = int((t1-w.start())*w.rate())/ni;
  int nj = int((t2-t1)*w.rate())/ni;
  int ne = nb+nj;
  double rATe=w.rate()/ni;

  wavearray<double> vP(nj);
  wavearray<double> bP(ni);
  vP.start(t1);
  vP.rate(rATe);
  bP.rate(w.rate());

  double rms;

  for(int i=0;i<nj;i++)
  { 
	bP.cpf(w,ni,(nb+i)*ni);
	rms = bP.rms();
	vP.data[i] = rms*rms*ni;
  }
  Plot(vP,opt,col,t1,t2,"");
}













