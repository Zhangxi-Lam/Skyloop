{  // apply transformation on white noise and produce a power spectral density plot

   int Rate = 1024;           // sampling rate (Hz)
   int Duration = 100;        // duration (seconds)
   int N = Rate*Duration;     // number of samples
   wavearray<double> ts(N);   //time series container
   ts.rate(Rate);
   
   // time series is filled with white noise data: 
   // add a sine wave @ 200Hz:                                                   
   double T = 1./Rate;
   double omega = TMath::TwoPi()*200;
   for(int i=0; i<N; i++) ts[i] += sin(omega*i*T);
   TRandom3 rnd(0);   
   for(int i=0; i<N; i++) {
     ts[i] = rnd.Gaus(); 
     ts[i] += sin(omega*i*T);
   }
   
   // produce the TF map:
   WDM<double> wdm(32, 64, 4, 8);   // define a WDM transform (32 bands)
   WSeries<double> tfmap;           // TF map container
   tfmap.Forward(ts, wdm, 0);       // apply the WDM to the time series 
   watplot pl;

   // fft of original ts                                                                         
   pl.plot(ts,"APL",1,4.,ts.stop()-4.,true,0.,0.,true,1./16);   
   
   // compute PSD
   wavearray<double> x;     
   wavearray<double> psd(33); 
   psd.rate(1./16);     
   for(int i=0; i<=32; ++i){
      tfmap.getLayer(x, i);
      psd.data[i] = x.mean()/Rate;
      //printf("%d\n", tmp.size());
      if(i==0 || i==32) psd.data[i] *= 2;
      psd.data[i] = sqrt(psd.data[i]);
   }
   pl.plot(psd,"SAME",2);
}
