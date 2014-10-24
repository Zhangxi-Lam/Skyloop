void optw(wavearray<double> &td, double phi)
{
  TCanvas *c_sp;
  int nn = td.size();
  wavearray<double> sp(nn);
  double x, y, z;
  int nn2 = nn/2;

  if (td.rate() <= 0.) {
    cout <<" Spectrum() error: invalid sample rate ="<< td.rate() <<"\n'";
    return NULL;
  }

    td.FFT();

// calculate power spectrum from Fourier coefficients
// without f=0;
//cout<<"nn="<<nn<<" Rate="<<sp.rate()<<endl; 

    x = td.data[0];
    td.data[0] *= x;
    x = td.data[1];
    td.data[1] *= x;

   for (int i = 1; i < nn2; i++) {
      x = td.data[2*i];
      y = td.data[2*i + 1];
//      z = (x*x + y*y)/pow(double(i),3);
      z = (x*x + y*y);
      td.data[2*i] = cos(phi)*z;
      td.data[2*i+1] = sin(phi)*z;
      if(2*i>=nn2) td.data[2*i+1]*=-1.;
    }

    td.FFT(-1);

}
   



