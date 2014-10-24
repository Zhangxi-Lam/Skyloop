{  
   // show how to plot netcluster with watplot class

   #define nIFO 1

   // create watplot 
   watplot* map = new watplot;

   // define wavearray time series
   int Rate = 512;            // sampling rate (Hz)
   int Duration = 2;          // duration (seconds)
   int N = Rate*Duration;     // number of samples
   wavearray<double> ts(N);   // time series container
   ts.rate(Rate);	      // set rate
   ts=0;		      // set array to zero 

   // time series is filled with SG100Q9: 
   double Q=9.;
   double amplitude=1.;
   double frequency=100.;
   double duration = Q/(TMath::TwoPi()*frequency);
   CWB::mdc::AddSGBurst(ts,amplitude, frequency, duration,0);
   
   // produce the TF map:
   WDM<double> wdm(32, 64, 4, 8);   // define a WDM transform (32 bands)
   WSeries<double> tfmap;           // TF map container
   tfmap.Forward(ts, wdm);          // apply the WDM to the time series 
   
   int layers = tfmap.maxLayer()+1;  // numbers of frequency bins (first & last bins have df/2)
   int slices = tfmap.sizeZero();    // number of time bins

   // create netpixel
   netpixel pix(nIFO);
   pix.core = true;
   pix.rate = tfmap.wrate();
   pix.layers = layers;

   // create netcluster
   netcluster wc;
   wc.clear();
   wc.setlow(0);
   wc.sethigh(Rate/2.);
   wc.start = tfmap.start();
   wc.stop = tfmap.stop();
   wc.rate = tfmap.rate();

   // fill netcluster with netpixels over the threshold
   int ifoID=0;
   int nPix=0;
   double THRESHOLD = 2;
   for(int i=0;i<slices;i++) {
     for(int j=0;j<layers;j++) {
       // get 00 phase amplitude
       double A00 = float(tfmap.pWavelet->pWWS[i*(layers)+j]);
       // get 90 phase amplitude
       double A90 = float(tfmap.pWavelet->pWWS[i*(layers)+j+tfmap.maxIndex()+1]);
       // compute energy average of 0 and 90 degree phases (quadratures)
       double EE = sqrt((A00*A00+A90*A90)/2);
       // select pixels above the THRESHOLD
       if(EE>THRESHOLD) {
         //cout << i << " " << j << " " << EE << endl;
         // fill netpixel
         int index = i*layers+j;    // sample index
         pix.data[ifoID].index = index;
         pix.data[ifoID].asnr = A00;
         pix.time = index;
         pix.frequency = j;
         wc.append(pix);            // save pixels in wc
         nPix++;
       }
     }
   }
   cout << "Selected Pixels : " << nPix << endl;

   wc.cluster(1,1); 		// cluster pixels

   // plot netcluster
   map->plot(&wc,1,1,'L',0);

   return map->canvas;		// used by THtml doc
}
