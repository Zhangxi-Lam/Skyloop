#define PHASE_SHIFT 90 		// phase shift degrees
#define SAVE_PLOT		// save plot to file

{
  //
  // DrawPhaseShift.C : Applies phase shift to sinusoid and draw original and shifted signals
  // Author : Gabriele Vedovato

  // create signal
  wavearray<double> x(16384);
  x.rate(512);              // sample rate
  double Pi = TMath::Pi();
  double dt = 1./x.rate();  
  double F = 10;            // frequency Hz
  for(int i=0;i<x.size();i++) x[i] = sin(2*Pi*F*dt*i);

  // init plot stuff
  watplot plot(const_cast<char*>("plot"),200,20,800,500);
  char gtitle[256];
  sprintf(gtitle,"Test Phase Shift : Original(black) - phase shifted(Red)");
  plot.gtitle(gtitle,"time(sec)","amplitude");
  plot.goptions("alp", 1, 0., 0.2);
  // draw signal
  x >> plot;

  // aplies phase shift 
  x.FFTW(1);
  TComplex C;
  double pShift=-PHASE_SHIFT/360.*TMath::TwoPi();   
  cout << "pShift : " << pShift << endl;
  for (int ii=0;ii<(int)x.size()/2;ii++) {
    TComplex X(x.data[2*ii],x.data[2*ii+1]);
    X=X*C.Exp(TComplex(0.,pShift));  // Phase Shift
    x.data[2*ii]=X.Re();
    x.data[2*ii+1]=X.Im();
  }
  x.FFTW(-1);

  // draw phase shift signal
  x >> plot;

#ifdef SAVE_PLOT
  // save plot to file
  TString gfile="phase_shift_plot.png";
  plot >> gfile;
#endif
}
