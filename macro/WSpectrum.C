/*-------------------------------------------------------
 * Package:     Wavelet Analysis Tool
 * File name:   WSpectrum.C
 *
 * This macro file is for ROOT interactive environment.
 * Macro plots wavelet spectrum.
 *-------------------------------------------------------
*/

id WSpectrum(Wseries<double> *w, int nc)
{
  cout
  <<"*****************************************************************\n"
  <<" HELP: Macro WSpectrum plots wavelet spectrum of wavelet decomposition\n"
  <<" represented by object 'wd'.\n\n"
  <<" Call: WSpectrum(wd, n)\n\n"
  <<" wd - is either object or pointer to object Wseries<double>\n"
  <<"*****************************************************************\n";
}

void WSpectrum(Wseries<double> *w, int nc)
{ WSpectrum(*w, nc ); }

void WSpectrum(Wseries<double> &w, int nc)
{
// nc - number of columns (layers)
// n  - number of rows

  float x;
  int n = w.N >> 1;

  th2=new TH2F("WS1", "Wavelet Spectrum", n, 0., n, nc, 0., nc);
// TH2F(const char* name, const char* title, Int_t nbinsx,
// Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup)

  for (int i=0; i < n; i++)
  {
    for (int j = 0; j < (nc - 1); j++) {
      x = w.data[(2 * (i >> j) + 1) << j];
      th2->Fill(i,j,x);
    }
    x = w.data[(i >> nc) << (nc + 1)];
    th2->Fill(i,nc-1,x);
  }
  th2->Draw("COLZ");
}

//------------------------------------------------------

void WTSpectrum(WaveletL &w)
{

  float x;
  
  if(w.IsTree==0) return;  

  int ni=1<<w.Level;
  int nj=w.pWDC->N/ni;

  th2=new TH2F("WS1", "Wavelet Tree Spectrum", nj, 0., nj, ni, 0., ni);
// TH2F(const char* name, const char* title, Int_t nbinsx,
// Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup)

  WaveData l;
  for(int i=0;i<ni;i++)
  {
	w.getFreqLayer(l,i);	
	for(int j=0;j<nj;j++) th2->Fill(j,i,l.data[j]);
  }
  th2->Draw("COLZ");
}

void WTSpectrum(WaveletD &w)
{

  float x;
  
  if(w.IsTree==0) return;  

  int ni=1<<w.Level;
  int nj=w.pWDC->N/ni;

  th2=new TH2F("WS1", "Wavelet Tree Spectrum", nj, 0., nj, ni, 0., ni);
// TH2F(const char* name, const char* title, Int_t nbinsx,
// Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup)

  WaveData l;
  for(int i=0;i<ni;i++)
  {
	w.getFreqLayer(l,i);	
	for(int j=0;j<nj;j++) th2->Fill(j,i,l.data[j]);
  }
  th2->Draw("COLZ");
}

void WTSpectrum(WaveletI &w)
{

  float x;
  
  if(w.IsTree==0) return;  

  int ni=1<<w.Level;
  int nj=w.pWDC->N/ni;

  th2=new TH2F("WS1", "Wavelet Tree Spectrum", nj, 0., nj, ni, 0., ni);
// TH2F(const char* name, const char* title, Int_t nbinsx,
// Axis_t xlow, Axis_t xup, Int_t nbinsy, Axis_t ylow, Axis_t yup)

  WaveData l;
  for(int i=0;i<ni;i++)
  {
	w.getFreqLayer(l,i);	
	for(int j=0;j<nj;j++) th2->Fill(j,i,l.data[j]);
  }
  th2->Draw("COLZ");
}














