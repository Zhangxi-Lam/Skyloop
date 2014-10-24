/********************************************************/
/* Wavelet Analysis Tool                                */
/* file Plot.C                                          */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/********************************************************/

struct Tplot {
  TCanvas* canvas;
  TGraph*  graph;
};

void clear(Tplot &p) { 
  if(p.canvas) { delete p.canvas;  p.canvas = NULL; }
  if(p.graph)  { delete p.graph; p.graph = NULL; }
}

void null(Tplot &p) { 
  p.canvas = NULL;
  p.graph = NULL;
}

Tplot Plot()
{
  cout
  <<"*****************************************************************\n"
  <<" HELP: Macro Plot plots as function of time the data from wavearray<float>\n"
  <<" objects for specifient interval t1..t2\n\n"
  <<" Call: Plot(td, t1, t2, opt, col)\n\n"
  <<" td - is either object or pointer to object wavearray<float>\n"
  <<" double t1 - start of time interval to plot in seconds\n"
  <<" double t2 - end of time interval to plot in seconds\n"
  <<" int   opt - 0 means create new canvas, 1 - plot on current canvas\n"
  <<" int col   - color number for plot, 4 by default (blue).\n"
  <<"*****************************************************************\n";
  Tplot pP; null(pP);
  return pP;
}

Tplot Plot(wavearray<float> *td, int opt = 0, int col = 4, 
	      double t1=0., double t2=0., char* c=NULL)
{ 
   return Plot(*td, opt, col, t1, t2, c); 
}

Tplot Plot(wavearray<double> *td, int opt = 0, int col = 4, 
	      double t1=0., double t2=0., char* c=NULL)
{ 
   wavearray<float> tf;
   waveAssign(tf,*td);
   return Plot(tf, opt, col, t1, t2, c); 
}

Tplot Plot(wavearray<double> &td, int opt = 0, int col = 4, 
	      double t1=0., double t2=0., char* c=NULL)
{ 
   wavearray<float> tf;
   waveAssign(tf,td);
   return Plot(tf, opt, col, t1, t2, c); 
}

Tplot Plot(wavearray<int> *td, int opt = 0, int col = 4, 
	      double t1=0., double t2=0., char* c=NULL)
{ 
   wavearray<float> tf;
   waveAssign(tf,*td);
   return Plot(tf, opt, col, t1, t2, c); 
}

Tplot Plot(wavearray<short> &td, int opt = 0, int col = 4, 
	      double t1=0., double t2=0., char* c=NULL)
{ 
   wavearray<float> tf;
   waveAssign(tf,td);
   return Plot(tf, opt, col, t1, t2, c); 
}

Tplot Plot(wavearray<int> &td, int opt = 0, int col = 4, 
	      double t1=0., double t2=0., char* c=NULL)
{ 
   wavearray<float> tf;
   waveAssign(tf,td);
   return Plot(tf, opt, col, t1, t2, c); 
}

// main function 

Tplot Plot(wavearray<float> &td, int opt = 0, int col = 4, 
	      double t1=0., double t2=0., char* c=NULL)
{
  Tplot pP; null(pP);

  if ( t2 < t1) {
     cout<<" Plot(td,t1,t2) error: t2 must be greater then t1."<<t1<<" "<<t2<<endl;
     return pP;
  }

  if(t2==0.) { t1 = td.start(); t2 = t1+td.size()/td.rate(); }

  double Ts = (td.rate() == 0)? 1.: 1./td.rate();
  int    i1 = (t1-td.start())/Ts; 
  int    i2 = (t2 == 0.)? td.size() : (t2-td.start())/Ts;
  int  nmax = i2-i1;
  wavearray<float> _x(nmax);
  wavearray<float> _y(nmax);
  wavearray<float> ex(nmax);
  wavearray<float> ey(nmax);
  double xmin, xmax, dx;
  double ymin, ymax, dy;
  xmin=0.;
  xmax=(nmax-1)*Ts;
  ymin=0.;
  ymax=0.;
  double x0 = i1*Ts+td.start(); 

  for (int i=0; i < nmax; i++) {
    _x.data[i] = x0 + Ts*i - 0.;
    _y.data[i] = td.data[i+i1];
    ex.data[i] = 0.0;
    ey.data[i] = 0.05;
    if (ymin > _y.data[i]) ymin=_y.data[i];
    if (ymax < _y.data[i]) ymax=_y.data[i];
  }

//  new TGraphErrors(nmax,x,y,ex,ey);
  pP.graph = new TGraph(nmax,_x.data,_y.data);
  pP.graph->SetLineColor(col);


//  pP.graph->Fit("fit","VMR");

  if (opt == 0) { 
    pP.canvas= new TCanvas("Plot","", 200, 10, 700, 500);
    pP.canvas->SetBorderMode(0);
    pP.canvas->SetFillColor(0);

    if(c) pP.graph->SetTitle(c);

//  pP.graph->Draw("ACP");
    pP.graph->Draw("APL");

    pP.canvas->Update();
    pP.graph->GetHistogram()->SetXTitle("time, s");
    pP.graph->GetHistogram()->SetYTitle("magnitude");

  }
  else {
    if(opt == 1) pP.graph->Draw("L");
    else         pP.graph->Draw("APL");
  }
  return pP;
}



Tplot Plot(wavearray<double> &a, wavearray<double> &t, 
	      int opt = 0, int col = 4, double x1 = 0., double x2 = 0., char* c)
{ 
   wavearray<float> af;
   wavearray<float> tf;
   waveAssign(af,a);
   waveAssign(tf,t);
   return Plot(af, tf, opt, col, x1, x2, c); 
}


Tplot Plot(wavearray<float> &td, wavearray<float> &tt, int opt = 0, 
	      int col = 4, double t1=0., double t2=0., char* c)
{
  Tplot pP; null(pP);

  if ( t2 < t1) {
    cout<<" Plot(td,t1,t2) error: t2 must be greater then t1.\n";
    return pP;
  }

  int N = td.size();
  if(N>tt.size()) N = tt.size();

  double Ts = (td.rate() == 0)? 1.: 1./td.rate();
  int i1=N;
  int i2=0; 
  double tt1 = fabs(tt[0]) + fabs(tt[N-1]); 
  double tt2 = tt[0] - tt[N-1]; 

  for (int i=0; i < N; i++) {
     if(tt[i] <= t1 || tt[i] >= t2) continue;
     if(tt[i] < tt1 && i1>=i) {i1 = i; tt1 = tt[i]; }
     if(tt[i] > tt2 && i2<=i) {i2 = i; tt2 = tt[i]; }
  }
     
  if(i2<i1) { i1=0; i2=N;}
  int nmax=i2-i1;
  wavearray<float> _x(nmax);
  wavearray<float> _y(nmax);
  wavearray<float> ex(nmax);
  wavearray<float> ey(nmax);
  double xmin, xmax, dx;
  double ymin, ymax, dy;
  xmin=0.;
  xmax=(nmax-1)*Ts;
  ymin=0.;
  ymax=0.;
  double x0 = tt.data[i1]; 

  for (int i=0; i < nmax; i++) {
    _x.data[i] = tt.data[i+i1];
    _y.data[i] = td.data[i+i1];
    ex.data[i] = 0.0;
    ey.data[i] = 0.05;
    if (ymin > _y.data[i]) ymin=_y.data[i];
    if (ymax < _y.data[i]) ymax=_y.data[i];
  }

//  pP.graph=new TGraphErrors(nmax,x,y,ex,ey);
  pP.graph=new TGraph(nmax,_x.data,_y.data);
  pP.graph->SetLineColor(col);

  pP.graph->SetMarkerStyle(col+18);

//  pP.graph->Fit("fit","VMR");

  if ((opt&1) == 0) { 
    pP.canvas= new TCanvas("Plot","", 200, 10, 700, 500);
    pP.canvas->SetBorderMode(0);
    pP.canvas->SetFillColor(0);

    pP.graph->SetTitle(c);

//  pP.graph->Draw("ACP");
//  pP.graph->Draw("APL");
    pP.graph->Draw("AP");

    pP.canvas->Update();
    pP.graph->GetHistogram()->SetXTitle("time, s");
    pP.graph->GetHistogram()->SetYTitle("magnitude");

  }
  else {
//  pP.graph->Draw("L");
    pP.graph->Draw("P");
  }
  return pP;
}














