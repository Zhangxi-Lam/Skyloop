{  // apply transformation on white noise and plot it

   int Rate = 1024;           // sampling rate (Hz)
   int Duration = 100;        // duration (seconds)
   int N = Rate*Duration;     // number of samples
   wavearray<double> ts(N);   //time series container
   ts.rate(Rate);
   
   // time series is filled with white noise data: 
   TRandom3 rnd(0);   
   for(int i=0; i<N; i++) ts[i] = rnd.Gaus();
   
   // produce the TF map:
   WDM<double> wdm(32, 64, 4, 8);   // define a WDM transform (32 bands)
   WSeries<double> tfmap;           // TF map container
   tfmap.Forward(ts, wdm);          // apply the WDM to the time series 
   
   // another example: add a sine wave @ 100Hz:
   double T = 1./Rate;
   double omega = TMath::TwoPi()*100;
   for(int i=0; i<N; i++) ts[i] += sin(omega*i*T);
   tfmap.Forward(ts, wdm);
   watplot map;

   // plot the TF map as energy average of 0 and 90 degree phases (quadratures)
   map.plot(tfmap);

   return map.canvas;	// used by THtml doc
}
