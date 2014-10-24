
/********************************************************/
/* Wavelet Analysis Tool                                */
/* file Histogram.C                                     */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/********************************************************/

TH1F* Histogram()
{
  cout
  <<"*****************************************************************\n"
  <<" HELP: Macro Histogram() plots histogram of data stored in object.\n";
  <<" \n"
  <<" Call: Histogram(wd, n, opt, col)\n\n"
  <<" wd - is either object or pointer to object WaveData or WaveData|WD\n"
  <<" n - is number of bins in histogram, n = 100 by default\n";
  <<" int   opt - 0 means create new canvas, 1 - plot on current canvas\n"
  <<" int col   - color number for plot, 4 by default (blue).\n"
  <<"*****************************************************************\n";
  return NULL;
}

TH1F* Histogram(WSeries<float> &wf, int n = 100, int opt = 0, int col = 4,
		   double xx1=0., double xx2=0.)
{ 
  wavearray<float> *pf = &wf;
  return Histogram(*pf, n, opt, col, xx1, xx2); 
}

TH1F* Histogram(WSeries<double> &wd, int n = 100, int opt = 0, int col = 4,
		   double xx1=0., double xx2=0.)
{ 
  wavearray<double> *pd = &wd;
  return Histogram(*pd, n, opt, col, xx1, xx2); 
}

TH1F* Histogram(wavearray<float> &tf, int n = 100, int opt = 0, int col = 4,
		   double xx1=0., double xx2=0.)
{ 
  wavearray<float> *pf = &tf;
  wavearray<double> td;
  waveAssign(td,tf);
  return Histogram(td, n, opt, col, xx1, xx2); 
}

TH1F* Histogram(wavearray<double> &td, int n = 100, int opt = 0, int col = 4, 
		   double xx1=0., double xx2=0.)
{
  if (n <= 0) n = 100;
  int nmax = td.size();;
  Axis_t *x;
  x = new Axis_t[nmax];
  float xmin, xmax;
  xmin=td.data[0];
  xmax=td.data[0];
  double a;

  for (int i=0; i < nmax; i++) {
    a = td.data[i];
    x[i] = a;
    if (xmin >  a) xmin=a;
    if (xmax <= a) xmax=a;
  }
  a = (xmax-xmin)*0.1;
  xmin -= a;
  xmax += a;

  if(xx1 < xx2){
     xmin = xx1;
     xmax = xx2;
  }

  TH1F *hh;

  if (xmin == xmax){ xmin--; xmax++; }

//  cout << xmin << "\n";
//  cout << xmax << "\n";

  hh = new TH1F("", "", n, xmin, xmax);

  hh->FillN(nmax,x,NULL);

  hh->SetLineColor(col);

  if ((opt&1) == 0) { 
    hh_canvas= new TCanvas("Histogram","", 50, 50, 1000, 700);
    hh_canvas->SetBorderMode(0);
    hh_canvas->SetFillColor(0);
//    hh_canvas->SetTitle("H1 power for loudest cluster");
//    hh_canvas->Divide(2,2);
    hh->Draw("");
    return hh;
  }
  else {
    hh->Draw("same");
//    hh_canvas->SetTitle("H1 power for loudest cluster");
    return hh;
  }
}
