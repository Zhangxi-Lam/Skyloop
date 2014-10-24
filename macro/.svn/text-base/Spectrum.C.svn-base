/********************************************************/
/* Wavelet Analysis Tool                                */
/* file Spectrum.C                                     */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/* Macro calculates and plots averaged power spectrum   */
/* for signal represented by object of class WaveData.*/
/********************************************************/

#define PI 3.141592653589793 

double Spectrum()
{
  cout <<"*****************************************************************\n"
       <<" HELP: Macro Spectrum calculates and plots power spectrum\n"
       <<" of signal taking samples in window of 'nn' points length\n"
       <<" and averaging result over all samples.\n\n"
       <<" Call: Spectrum( td, f1, f2, nn, opt, col)\n\n";
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

TCanvas* Spectrum(wavearray<float>* tf, float f1=0., float f2=0.,
                  int nn=1024, int opt=2, int col=4)
{
   wavearray<double> td(tf.size());
   td.rate(tf.rate());
   waveAssign(td,*tf);
   return Spectrum(td,f1,f2,nn,opt,col);
}

TCanvas* Spectrum(wavearray<float> &tf, float f1=0., float f2=0.,
                  int nn=1024, int opt=2, int col=4)
{
   wavearray<double> td(tf.size());
   td.rate(tf.rate());
   waveAssign(td,tf);
   return Spectrum(td,f1,f2,nn,opt,col);
}

TCanvas* Spectrum(wavearray<double>* tf, float f1=0., float f2=0.,
                   int nn=1024, int opt=2, int col=4)
{
   return Spectrum(*tf,f1,f2,nn,opt,col);
}


TCanvas* Spectrum(wavearray<double> &td, float f1=0., float f2=0.,
                  int nn=1024, int opt=2, int col=4)
{
   int n = td.size();
   int J = int(td.edge()*td.rate());
   if (nn <= 0) nn = n-2*J;
   wavearray<double> sp(1);
   double *f, *p;
   double x, y;
   int nn2 = nn/2;
   int i1 = 1;
   int i2 = nn2;
   double dwavefft = td.rate()/nn;
   double apsd = 0.;
   TCanvas* c_sp;
   
   if (td.rate() <= 0.) {
      cout <<" Spectrum() error: invalid sample rate ="<< td.rate() <<"\n'";
      return NULL;
   }
   
   f = new double[nn2];
   p = new double[nn2];
   for (int k=0; k<nn2; k++) p[k]=0.;
   
   if(f1<f2 && f2<=td.rate()/2.) {
      i1 = int(f1/dwavefft);
      i2 = int(f2/dwavefft);
   }
   if(i1<1) i1 = 1;
   if(i2<=i1) i2 = nn2;
   
   int ns = n/nn;
   if (ns == 0) {
      cout << "Spectrum() error: data too short for specified window length="
           << nn << "\n";
      return NULL;
   }
   sp.resize(nn);
   sp.rate(td.rate());
   
   for (int j=J;  j <= (n-nn-J) ; j+=nn) {
      sp = 0.;
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
         p[i] += (x*x + y*y)*nn/td.rate();
//      cout<<sqrt(x*x+y*y)<<endl;
      }
   }

   for (int i = i1; i < i2; i++) {
      f[i] = i*dwavefft; 
      p[i] /= ns;
      p[i] = sqrt(p[i]);
      apsd += sqrt(p[i]);
   }
   apsd /= i2-i1;
//  p[0]=0.;

   gr1=new TGraph(i2 - i1, f+i1, p+i1);
//  gr1->SetMarkerStyle(21);
   gr1->SetLineColor(col);

   if ((opt&4) == 0) {
      c_sp = new TCanvas("Spectrum","", 200, 10, 900, 600);
      c_sp->SetFillColor(0);
      c_sp->SetBorderMode(0);

      if ( opt & 1 ) c_sp -> SetLogx();
      if ( opt & 2 ) c_sp -> SetLogy();
      gr1->SetTitle("");
      gr1->Draw("ALP");
      gr1->GetHistogram()->SetXTitle("frequency, Hz");
      gr1->GetHistogram()->SetYTitle("amplitude/#sqrt{Hz}");
      c_sp->Update();
   }
   else {
      gr1->Draw("L");
   }
   
   if ((opt&4) == 0) return c_sp; else return NULL;
}


double Spectrum(wavearray<double> &td, 
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
      cout <<" Spectrum() error: invalid sample rate ="<< td.rate() <<"\n'";
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
      cout << "Spectrum() error: data too short for specified window length="
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
      c_sp = new TCanvas("Spectrum","", 200, 10, 700, 500);
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










