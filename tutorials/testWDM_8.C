{  // accuracy of time delay filters:
   // compare time shifts done in the TF domain using WDM TD filters vs the correct values 
   // WDM + TD Filters + WDM Inverse vs the trivial time shift of original data  
    
   // TO REMOVE BOUNDARY ARTIFACTS, ONE SECOND OF DATA IS DISCARDED FOR EACH OF THE OPERATIONS:
   //  (i)  APPLICATION OF TD FILTERS  
   //  (ii) INVERSE TRANSFORMATION
   
   int Rate = 1024;               // sampling rate (Hz)
   int Duration = 20;             // duration (seconds)
   int N = Rate*Duration;         // number of samples
   wavearray<double> ts(N);       //time series container
   ts.rate(Rate);
   
   // time series is filled with white noise data: 
   TRandom3 rnd(0);   
   for(int i=0; i<N; i++) ts[i] = rnd.Gaus();
         
   WDM<double> wdm(32, 64, 4, 8);   // define a WDM transform (32 bands)
   WSeries<double> tfmap;           // TF map container
   tfmap.Forward(ts, wdm);
   
   
   WDM<double>* pWDM = (WDM<double>*)tfmap.pWavelet;
   pWDM->setTDFilter(18);           // computes TD filters
   
   double* tmp = new double[pWDM->nWWS/2];   // will store time delayed coefficients
   for(int j=1*32; j<19*32; ++j)    // time indices between 1-19s (1 sec discarded)
      for(int i=0;i<=32; ++i)       // frequency  bands  
         tmp[j*33+i] = (double)pWDM->getPixelAmplitude(i, j,  13, false);  //delay by 13 samples
   
   // overwrite original tmap:
   double* TFMap = pWDM->pWWS;      // all TF coefficients are stored in TFMap
   for(int j=32; j<19*32; ++j)      // time indices between 1-19s
      for(int i=0;i<=32; ++i)       // frequency  bands
         TFMap[j*33+i] = tmp[j*33+i];
         
   // inverse to time domain and plot shifted signal on top of original (ZOOM around 10s to see)
   tfmap.Inverse();
   
   wavearray<double> ts2(N);
   tfmap.getLayer(ts2, 0);
   
   
   TH1F* h = new TH1F("h", "Time delay filter relative error", 100, -0.01, 0.01);
   for(int j=1024*2; j<18*1024; ++j)    // time indices between 2-18s (another second discarded)
      h->Fill(ts2[j]/ts[j-13] -  1);
   h->Draw();
   
   delete [] tmp;
   
   // NOTE: TD FILTERS TRUNCATION DEFINES THEIR PRECISION; 
   // FOR EXAMPLE, USING 
   //    pWDM->setTDFilter(18) 
   // INSTEAD IMPROVES THE RELATIVE ERROR BY MORE THAN AN ORDER OF MAGNITUDE.
   
}
