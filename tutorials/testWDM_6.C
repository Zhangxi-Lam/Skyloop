{  // residuals 
   
   int Rate = 1024;           // sampling rate (Hz)
   int Duration = 100;        // duration (seconds)
   int N = Rate*Duration;     // number of samples
   wavearray<double> ts(N);   //time series container
   ts.rate(Rate);
   
   // time series is filled with white noise data: 
   TRandom3 rnd(0);   
   for(int i=0; i<N; i++) ts[i] = rnd.Gaus();
   
   // produce the TF map:
   WSeries<double> tfmap;           // TF map container
   WDM<double> wdm(32, 64, 4, 8);   // define a WDM transform (32 bands) 
   tfmap.Forward(ts, wdm);          // apply the WDM to the time series 
   
   wavearray<double> ts2(ts);
      
   tfmap.Inverse();          // inverse transform
   tfmap.getLayer(ts2, 0);   // move the time domain data in ts2
   ts -= ts2;                // substract from orginal data and plot:
   
   TH1F* h = new TH1F("h", "Residuals", 100, -0.0001, 0.0001);
   for(int i=0; i<N; ++i)h->Fill(ts[i]);
   h->Draw();
   
   // NOTE: if we call tfmap.Inverse(-2) an inverse transform 
   // that uses the quadrature coefficients is performed
}
