/********************************************************/
/* Wavelet Analysis Tool                                */
/* file Phasecor.C                                     */
/*                                                      */
/* This macro file is for ROOT interactive environment. */
/* Macro calculates and plots averaged power spectrum   */
/* for signal represented by object of class WaveData.*/
/********************************************************/

#define PI 3.141592653589793 

TCanvas *phasecor()
{
  cout <<"*****************************************************************\n"
       <<" HELP: Macro phasecor() calculates and plots phase correlation   \n"
       <<" coefficient taking samples with window of 'nn' points length\n"
       <<" and averaging result over all samples.\n\n"
       <<" Call: phasecor( td1, td2, f1, f2, nn, opt, col)\n\n";
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

TCanvas *phasecor(WaveData &td1, WaveData &td2, 
		  float f1=0., float f2=0.,
		  int nn=1024, int opt=2, int col=4)
{

  if (td1.N != td2.N) {
    cout <<" phasecor() error: input arrays have different length"<<"\n'";
    return NULL;
  }

  if (td1.Rate <= 0.) {
    cout <<" phasecor() error: invalid sample rate ="<< td.Rate <<"\n'";
    return NULL;
  }

  if (nn <= 0) nn =td1.N;
  WaveData sp(1);
  WaveData sp1(1);
  WaveData sp2(1);
  double *f, *p;
  double x, y, r;
  int n = td1.N;
  int nn2 = nn/2;
  int i1, i2;
  int norm=0;
  double dwavefft = td1.Rate / nn;

  f = new double[nn2];
  p = new double[nn2];
  for (int k=0; k<nn2; k++) p[k]=0.;

  i1 = (f1 > dwavefft && f1 < nn2*dwavefft)? int(f1/dwavefft) : 1;
  if (f2 <= i1*dwavefft) i2 = nn2;
  else if (f2 > nn2*dwavefft) i2 = nn2;
       else i2 = int (f2/dwavefft);

  int ns = n/nn;
  if (ns == 0) {
    cout << "phasecor() error: data too short for specified window length="
         << nn << "\n";
    return NULL;
  }
  sp1.Resize(nn);
  sp2.Resize(nn);
   sp.Resize(nn);
   sp=0.;

  for (int j=0;  j <= (n - nn) ; j+=nn) {
    sp1.cpf(td1, nn, j);
    sp2.cpf(td2, nn, j);

    if ((opt&128) != 0) {       // multiply data to Hann window
      double dphi=2.*PI/nn;
      double w=sqrt(2./3.);
      double hann;
      for (int i=0; i < nn; i++){
	 hann=w*(1.-cos(i*dphi));
	 sp1.data[i]*=hann;
	 sp2.data[i]*=hann;
      }
    }
    sp1.FFT();
    sp2.FFT();
    norm++;

   for (int i = i1; i < i2; i++) {
// get rid of amplitude
      x = sp1.data[2*i];
      y = sp1.data[2*i+1];
      r = 1./sqrt(x*x+y*y);
      sp1.data[2*i] *= r;
      sp1.data[2*i+1] *= r;

      x = sp2.data[2*i];
      y = sp2.data[2*i+1];
      r = 1./sqrt(x*x+y*y);
      sp2.data[2*i] *= r;
      sp2.data[2*i+1] *= r;

// phase difference
      x = sp1.data[2*i]*sp2.data[2*i] + sp1.data[2*i+1]*sp2.data[2*i+1];
      y = sp2.data[2*i]*sp1.data[2*i+1] - sp1.data[2*i]*sp2.data[2*i+1];

      sp.data[2*i] += x;
      sp.data[2*i + 1] += y;
   }

  }
// correlation coefficient
  sp*=1./double(norm);
  for (int i = i1; i < i2; i++) {
     x = sp.data[2*i];
     y = sp.data[2*i + 1];
     p[i - i1] = sqrt(x*x + y*y);
//     cout<<i<<"  "<<p[i-i1]<<endl;
  }

  for (int i = 0; i < (i2-i1); i++) {
    f[i] = (i+i1)*dwavefft; 
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
    gr1->GetHistogram()->SetYTitle("phase correlation");
    c_sp->Update();
  }
  else {
    gr1->Draw("L");
  }

  if ((opt&4) == 0) return c_sp; else return NULL;
}









