/********************************************************/
/* Wavelet Analysis Tool                                */
/* file lineFinder.C                                     */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/* Macro calculates and plots averaged power spectrum   */
/* for signal represented by object of class WaveData.*/
/********************************************************/

#define PI 3.141592653589793 

double lineFinder()
{
  cout <<"*****************************************************************\n"
       <<" HELP: Macro lineFinder calculates and plots power spectrum\n"
       <<" of signal taking samples in window of 'nn' points length\n"
       <<" and averaging result over all samples.\n\n"
       <<" Call: lineFinder( td, f1, f2, nn, opt, col)\n\n";
  cout <<" td  - either object or pointer to object WaveData,\n"
       <<"       td is the only required argument, other are optional,\n"
       <<" float f1, f2 - lower and upper frequency of range to plot,\n"
       <<"        by default, f1=0., f2=0., this means plot full range,\n"
       <<" int nn   - number of points in time window, 10000 by default,\n";
  cout <<" int opt  - select drawing options, 2 by default\n"
       <<"          = 0 - axis X,Y linear\n"
       <<"          = 1 - axis X - logarithmic ,Y - linear\n"
       <<"          = 2 - axis X - linear ,Y - logarithmic\n"
       <<"          = 3 - axis X,Y logarithmic\n"
       <<"          = 4 - do not create TCanvas, use current\n"
       <<"            this allow to draw new plot over existent\n"
       <<"          = 128 + n, same as for 'n' but use Hann window (1-cos)\n"
       <<" int col  - choose color to plot spectrum, 4 by default (blue).\n"
       <<"*****************************************************************\n";
  return NULL;
}

double lineFinder(wavearray<float> &tf, float f1=0., float f2=0.,
                   int nn=1024, int opt=2, int col=4)
{
   wavearray<double> td(tf.size());
   td.rate(tf.rate());
   waveAssign(td,tf);
   return lineFinder(td,f1,f2,nn,opt,col);
}


double lineFinder(wavearray<double> &td, float f1=0., float f2=0.,
                   int nn=1024, int opt=2, int col=4)
{
  if (nn <= 0) nn =td.size();
  wavearray<double> sp(1);
  double *f, *p;
  double x, y;
  int n = td.size();
  int nn2 = nn/2;
  int i1, i2;
  double dwavefft = td.rate() / nn;
  double apsd = 0.;

  if (td.rate() <= 0.) {
    cout <<" lineFinder() error: invalid sample rate ="<< td.rate() <<"\n'";
    return NULL;
  }

  f = new double[nn2];
  p = new double[nn2];
  for (int k=0; k<nn2; k++) p[k]=0.;

  i1 = (f1 > dwavefft && f1 < nn2*dwavefft)? int(f1/dwavefft) : 1;
  if (f2 <= i1*dwavefft) i2 = nn2;
  else if (f2 > nn2*dwavefft) i2 = nn2;
       else i2 = int (f2/dwavefft);

  int ns = n/nn;
  if (ns == 0) {
    cout << "lineFinder() error: data too short for specified window length="
         << nn << "\n";
    return NULL;
  }
  sp.resize(nn);
  sp.rate(td.rate());

  for (int j=0;  j <= (n - nn) ; j+=nn) {
    sp.cpf(td, nn, j);

    if ((opt&128) != 0) {
// multiply data to Hann window
      double dphi=2.*PI/nn;
      double w=sqrt(2./3.);
      for (int i=0; i < nn; i++) sp.data[i]*=w*(1.-cos(i*dphi));
    }
    sp.FFT();

// calculate power spectrum from Fourier coefficients
// without f=0;
//cout<<"nn="<<nn<<" Rate="<<sp.rate()<<endl; 
   for (int i = i1; i < i2; i++) {
      x = sp.data[2*i];
      y = sp.data[2*i + 1];
      p[i - i1] += (x*x + y*y)*nn/td.rate();
    }
  }

  for (int i = 0; i < (i2-i1); i++) {
    f[i] = (i+i1)*dwavefft; 
    p[i] /= ns;
    apsd += p[i];
  }
  apsd /= i2-i1;
//  p[0]=0.;

// find lines

  double snr;
  double sline;
  int nwin = 9;
  int iF, iL;
  double sumN = 0;

  for(int i=12; i < (i2-i1)-12; i++) {
    iF = i-nwin/2;
    iL = i+nwin/2;
    sumN = 0.;

    for(j=0; j<nwin; j++)
      sumN += p[iF+j];
    
    snr = sumN;
    snr -= p[i];
    snr /= nwin-1;
    snr = sqrt(double(ns))*(p[i]-snr)/snr;
    if(snr > 9.) 
       cout<<"frequency="<<f[i]<<"  snr="<<snr<<"  width=1"<<endl;
    
    sumN += p[iF-1]+p[iF-2]+p[iF-3]+p[iL]+p[iL+1]+p[iL+2];
    sline = (p[i]+p[i-1]+p[i+1])/3.;
    snr = sumN;
    snr -= sline*3.;
    snr /= nwin+3;
    snr = sqrt(double(ns))*(sline-snr)/snr;
    if(snr > 9.) 
       cout<<"frequency="<<f[i]<<"  snr="<<snr<<"  width=3"<<endl;

  }  

/*
  gr1=new TGraph(i2 - i1, f, p);
//  gr1->SetMarkerStyle(21);
  gr1->SetLineColor(col);

  if ((opt&4) == 0) {
    c_sp = new TCanvas("lineFinder","", 200, 10, 700, 500);
    c_sp->SetBorderMode(0);

    if ( opt & 1 ) c_sp -> SetLogx();
    if ( opt & 2 ) c_sp -> SetLogy();
    gr1->SetTitle("");
    gr1->Draw("ALP");
    gr1->GetHistogram()->SetXTitle("frequency");
    gr1->GetHistogram()->SetYTitle("power");
    c_sp->Update();
  }
  else {
    gr1->Draw("L");
  }
*/
  return apsd;
//  if ((opt&4) == 0) return c_sp; else return NULL;
}


double lineFinder(wavearray<double> &td, 
		wavearray<double> *psd, 
		float f1=0., float f2=0.,
		int nn=1024, int opt=2, int col=4)
{
  if (nn <= 0) nn =td.size();
  wavearray<double> sp(1);
  double *f, *p;
  double x, y;
  int n = td.size();
  int nn2 = nn/2;
  int i1, i2;
  double dwavefft = td.rate() / nn;

  if (td.rate() <= 0.) {
    cout <<" lineFinder() error: invalid sample rate ="<< td.rate() <<"\n'";
    return NULL;
  }

  f = new double[nn2];

  if(psd.size() != nn2){
     psd->resize(nn2);
     psd->rate(0.);
     *psd = 0.;
  }
  else
     *psd *= psd->rate();
  
  p=&(psd->data[0]);

//  for (int k=0; k<nn2; k++) p[k]=0.;

  i1 = (f1 > dwavefft && f1 < nn2*dwavefft)? int(f1/dwavefft) : 1;
  if (f2 <= i1*dwavefft) i2 = nn2;
  else if (f2 > nn2*dwavefft) i2 = nn2;
       else i2 = int (f2/dwavefft);

  int ns = n/nn;
  if (ns == 0) {
    cout << "lineFinder() error: data too short for specified window length="
         << nn << "\n";
    return NULL;
  }
  sp.resize(nn);
  sp.rate(td.rate());

  for (int j=0;  j <= (n - nn) ; j+=nn) {
    sp.cpf(td, nn, j);

    if ((opt&128) != 0) {
// multiply data to Hann window
      double dphi=2.*PI/nn;
      double w=sqrt(2./3.);
      for (int i=0; i < nn; i++) sp.data[i]*=w*(1.-cos(i*dphi));
    }
    sp.FFT();

// calculate power spectrum from Fourier coefficients
// without f=0;
//cout<<"nn="<<nn<<" Rate="<<sp.rate()<<endl; 
   for (int i = i1; i < i2; i++) {
      x = sp.data[2*i];
      y = sp.data[2*i + 1];
//      p[i - i1] += (x*x + y*y)*nn/td.rate();
    }
  }

//  psd->Rate+=ns;
  for (int i = 0; i < (i2-i1); i++) {
    f[i] = (i+i1)*dwavefft; 
//    p[i] /= psd->Rate;
//    p[i] = (p[i]);
  }

//  p[0]=0.;

  gr1=new TGraph(i2 - i1, f, p);
//  gr1->SetMarkerStyle(21);
  gr1->SetLineColor(col);

  if ((opt&4) == 0) {
    c_sp = new TCanvas("lineFinder","", 200, 10, 700, 500);
    c_sp->SetBorderMode(0);

    if ( opt & 1 ) c_sp -> SetLogx();
    if ( opt & 2 ) c_sp -> SetLogy();
    gr1->SetTitle("");
    gr1->Draw("ALP");
    gr1->GetHistogram()->SetXTitle("frequency");
    gr1->GetHistogram()->SetYTitle("power");
    c_sp->Update();
  }
  else {
    gr1->Draw("L");
  }

  delete f;
  return 0.;
//  if ((opt&4) == 0) return c_sp; else return NULL;
}










