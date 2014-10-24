{  // using Time Delay Filters to shift a sine-gaussian in the TF domain

   int Rate = 1024;               // sampling rate (Hz)
   int Duration = 20;             // duration (seconds)
   int N = Rate*Duration;         // number of samples
   wavearray<double> ts(N);       //time series container
   ts.rate(Rate);
   watplot pl;
   
   // time series: sine gaussian  @ 100Hz, Q=8, t = 10s
   double Q = 8;
   double dT = 1./Rate;
   double omega = TMath::TwoPi()*100;
   double aux = omega/Q;
   aux *= aux/2;
   for(int i=0; i<N; i++){
      double tt =  (i*dT-10); 
      ts[i] = sin(omega*tt)*exp(-tt*tt*aux);
   }
   
   pl.plot(ts, "APL", 4, 9.9, 10.1);
   
   WDM<double> wdm(32, 64, 4, 8);   // define a WDM transform (32 bands)
   WSeries<double> tfmap;           // TF map container
   tfmap.Forward(ts, wdm);
   
   
   WDM<double>* pWDM = (WDM<double>*)tfmap.pWavelet;
   pWDM->setTDFilter(12);           // computes TD filters
   
   double* tmp = new double[pWDM->nWWS/2];   // will store time delayed coefficients
   for(int j=9*32; j<11*32; ++j)    // time indeces between 9-11s
      for(int i=0;i<=32; ++i)       // frequency  bands  
         tmp[j*33+i] = (double)pWDM->getPixelAmplitude(i, j,  13, false);  //delay by 13 samples
   
   // overwrite original tmap:
   double* TFMap = pWDM->pWWS;      // all TF coefficients are stored in TFMap
   for(int j=9*32; j<11*32; ++j)    // time indeces between 9-11s
      for(int i=0;i<=32; ++i)       // frequency  bands
         TFMap[j*33+i] = tmp[j*33+i];
         
   // inverse to time domain and plot shifted signal on top of original (ZOOM around 10s to see)
   tfmap.Inverse();
   tfmap.getLayer(ts, 0);
   pl.plot(ts, "SAME", 2);
   
   delete [] tmp;
}
