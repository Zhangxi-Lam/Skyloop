{  // access TF data

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
   watplot pl;                      // create canvas
   pl.plot(ts,"APL",1);             // plot time series
   
   // extract layer 2
   
   wavearray<double> tmp;     
   tfmap.getLayer(tmp, 2);    // extract TF amplitudes of the 3rd band
   pl.plot(tmp,"SAME",2);     // superimpose layer amplitudes
   
   // accessing the quadrature bands is done using negative numbers:
   tfmap.getLayer(tmp, -2);   // 3rd band of the quadrature
   tfmap.getLayer(tmp, -0.5); // 1st band of the quadrature
   
}
