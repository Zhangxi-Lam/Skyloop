{  // FFT of basis function from selected layer
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
  
   
   // zero all bands except 3rd (24Hz - 40Hz)
   
   wavearray<double> tmp;     
   for(int i=0; i<=32; ++i)if(i!=2){
      tfmap.getLayer(tmp, i);   
      tmp = 0;
      tfmap.putLayer(tmp, i);
   }
   
   tfmap.Inverse();          // inverse transform
   tfmap.getLayer(tmp, 0);   // move the time domain data in tmp
   // fft of original ts
   pl.plot(ts,"APL",1,4.,ts.stop()-4.,fft=true,0.,0.,psd=true,1.);
   // fft of extracted layer
   pl.plot(tmp,"SAME",2,4.,ts.stop()-4.,fft=true,0.,0.,psd=true,1.);

   return pl.canvas;	// used by THtml doc
}
