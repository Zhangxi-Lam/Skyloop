
/********************************************************/
/* Wavelet Analysis Tool                                */
/* file Hist2D.C                                     */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/********************************************************/

TCanvas* Hist2D()
{
  cout
  <<"*****************************************************************\n"
  <<" HELP: Macro Histogram() plots histogram of data stored in object.\n";
  <<" \n"
  <<" Call: Hist2D(wd, n, opt, col)\n\n"
  <<" wd - is either object or pointer to object WaveData or WaveData|WD\n"
  <<" n - is number of bins in histogram, n = 100 by default\n";
  <<" int   opt - 0 means create new canvas, 1 - plot on current canvas\n"
  <<" int col   - color number for plot, 4 by default (blue).\n"
  <<"*****************************************************************\n";
  return NULL;
}

TCanvas* Hist2D(wavearray<double> *x, wavearray<double> *y,
		int n = 20, float xmin = 0., float xmax = 0.,
		int m = 20, float ymin = 0., float ymax = 0.,
		char* opt = "", int col=1)
{ return Hist2D(*x, *y, n, xmin, xmax, m, ymin, ymax, opt); }

TCanvas* Hist2D(wavearray<float> &x, wavearray<float> &y,
		int n = 20, float xmin = 0., float xmax = 0.,
		int m = 20, float ymin = 0., float ymax = 0.,
		char* opt = "", int col=1)
{ 
   wavearray<double> xx(x.size());
   wavearray<double> yy(y.size());
   for(int i=0; i<x.size(); i++) xx.data[i] = double(x.data[i]);
   for(int i=0; i<y.size(); i++) yy.data[i] = double(y.data[i]);
   return Hist2D(xx, yy, n, xmin, xmax, m, ymin, ymax, opt); 
}

TCanvas* Hist2D(wavearray<double> &x, wavearray<double> &y,
		int n = 20, float xmin = 0., float xmax = 0.,
		int m = 20, float ymin = 0., float ymax = 0.,
		char* opt = "", int col=1)
{
  if (n <= 0) n = 10;
  if (m <= 0) m = 10;

  int nmax = (x.size() > y.size()) ? y.size() : x.size();

  float min, max;

  if (xmin == xmax){         // find min and max from data
    min=x.data[0];
    max=x.data[0];
    for (int i=0; i < nmax; i++) {
      if (min > x.data[i]) min=x.data[i];
      if (max < x.data[i]) max=x.data[i];
    }
    xmin = min;
    xmax = max;
  }

  if (ymin == ymax){         // find min and max from data
    min=y.data[0];
    max=y.data[0];
    for (int i=0; i < nmax; i++) {
      if (min > y.data[i]) min=y.data[i];
      if (max < y.data[i]) max=y.data[i];
    }
    ymin = min;
    ymax = max;
  }


  TH2F *hh;
  hh = new TH2F("h2d", "", n, xmin, xmax, m, ymin, ymax);

  for (int i=0; i < nmax; i++)
     hh->Fill(x.data[i],y.data[i]);
//    if(x.data[i]>0 && y.data[i]>0.) hh->Fill(x.data[i],y.data[i]);

//  hh->SetLineColor(col);
  hh->SetMarkerColor(col);
  hh->SetMarkerStyle(20);
  hh->SetMarkerSize(1);

  if (col == 1) { 
    hh_canvas= new TCanvas("Hist2D","", 200, 20, 700, 700);
    hh_canvas->SetBorderMode(0);
    hh_canvas->SetFillColor(0);
    hh->Draw("");
    hh->Draw(opt);
    return hh_canvas;
  }
  else {
    hh->Draw("same");
    return NULL;
  }
}



