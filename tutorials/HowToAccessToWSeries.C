// This example shows how access to SSeries 

{  
   #define nLAYERS 	32	// number of frequency layers
   #define RATE		512	// sample rate
   #define DURATION	2	// duration of data

   char title[256];

   // create watplot 
   watplot* plot = new watplot;
   TCanvas* canvas = plot->canvas;
   canvas->Divide(1,2);

   // define wavearray time series
   int Rate = RATE;           	// sampling rate (Hz)
   int Duration = DURATION;   	// duration (seconds)
   int N = Rate*Duration;     	// number of samples
   wavearray<double> ts(N);   	// time series container
   ts.rate(Rate);	      	// set rate
   ts=0;		      	// set array to zero 

   // time series is filled with SG100Q9: 
   double Q=9.;
   double amplitude=1.;
   double frequency=100.;
   double duration = Q/(TMath::TwoPi()*frequency);
   CWB::mdc::AddSGBurst(ts,amplitude, frequency, duration,0);
   
   // produce the TF map:
   WDM<double> wdm(nLAYERS, 2*nLAYERS, 4, 8);   // define a WDM transform (32 bands)
   WSeries<double> tfmap;           		// TF map container
   tfmap.Forward(ts, wdm);          		// apply the WDM to the time series 

   // compute dfxdt pixel resolution
   double df = tfmap.resolution();		// frequency bin resolution (hz)
   double dt = 1./(2*df);			// time bin resolution (sec)
   char tfres[64]; sprintf(tfres,"(1/%g)x(%g) (sec)x(Hz)",2*df,df);

   // plot the TF map as energy average of 0 and 90 degree phases (quadratures)
   canvas->cd(1);
   gPad->SetGridx(); gPad->SetGridy();
   plot->plot(tfmap);
   plot->hist2D->SetName("WSeries-1");;
   sprintf(title,"%s - Res : %s","TF Map : Signal",tfres);
   plot->hist2D->SetTitle(title);;
   
   int layers = tfmap.maxLayer()+1;  // numbers of frequency bins (first & last bins have df/2)
   int slices = tfmap.sizeZero();    // number of time bins

   // set to 0 the  pixels under the THRESHOLD
   double THRESHOLD = 2;
   for(int i=0;i<slices;i++) {
     for(int j=0;j<layers;j++) {
       // the second parameter is double
       // epsilon=0.01 is used to select the 0 layer for phase 90
       float A00 = tfmap.getSample(i,j+0.01);        	// get phase 00 amplitude
       float A90 = tfmap.getSample(i,-(j+0.01));     	// get phase 90 amplitude
       // compute energy average of 0 and 90 degree phases (quadratures)
       double EE = sqrt((A00*A00+A90*A90)/2);
       if(EE<THRESHOLD) {
         //cout << i << " " << j << " " << EE << endl;
         tfmap.putSample(0,i,j+0.01);			// set to 0 the phase 00 amplitude
         tfmap.putSample(0,i,-(j+0.01));		// set to 0 the phase 90 amplitude
       }
     }
   }

   // plot the TF map as energy average of 0 and 90 degree phases (quadratures)
   canvas->cd(2);
   gPad->SetGridx(); gPad->SetGridy();
   plot->hist2D=NULL;
   plot->plot(tfmap);
   plot->hist2D->SetName("WSeries-2");;
   sprintf(title,"%s - Res : %s - pixels above the threshold (sqrt((A00*A00+A90*A90)/2)>%g","TF Map : Signal",tfres,THRESHOLD);
   plot->hist2D->SetTitle(title);;

   return canvas;
}
