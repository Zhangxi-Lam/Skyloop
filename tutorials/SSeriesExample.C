{  
   // show how the sparse map SSeries works

   #define nIFO 1
   //#define TEST_WRITE_READ_SPARSE_MAP
   #define nLAYERS 	32
   #define RATE		512
   #define DURATION	2

   char title[256];

   // create watplot 
   watplot* plot = new watplot;
   TCanvas* canvas = plot->canvas;
   canvas->Divide(2,2);

   // define wavearray time series
   int Rate = RATE;           // sampling rate (Hz)
   int Duration = DURATION;   // duration (seconds)
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
   plot->hist2D->SetName("Sparse-1");;
   sprintf(title,"%s - Res : %s","TF Map : Signal",tfres);
   plot->hist2D->SetTitle(title);;
   
   // add white noise data: 
   TRandom3 rnd(1);   
   for(int i=0; i<N; i++) ts[i] += rnd.Gaus();
   tfmap.Forward(ts, wdm);          // apply the WDM to the time series 
   // plot the TF map as energy average of 0 and 90 degree phases (quadratures)
   canvas->cd(2);
   gPad->SetGridx(); gPad->SetGridy();
   plot->hist2D=NULL;
   plot->plot(tfmap);
   plot->hist2D->SetName("Sparse-2");;
   sprintf(title,"%s - Res : %s","TF Map : Signal + Noise",tfres);
   plot->hist2D->SetTitle(title);;

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
   double THRESHOLD = 4;
   for(int i=0;i<slices;i++) {
     for(int j=0;j<layers;j++) {
       float A00 = tfmap.getSample(i,j+0.01);        // phase 00 amplitude
       float A90 = tfmap.getSample(i,-(j+0.01));     // phase 90 amplitude
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

   // create sseries and fill with netcluster
   SSeries<double> ss;
   ss.SetMap(&tfmap);		// associate tfmap to ss
   double mTau = 0.040;
   ss.SetHalo(mTau);		// set halo
   ss.AddCore(ifoID,&wc);	// add core pixels
   ss.UpdateSparseTable();	// update sparse map
   ss.Shrink();                 // resize to 0 the TF map : leave only the sparse map tables
   wc.clear();			// clear netcluster

   // compute sparse statistic
   int ncore=0;        		// core pixels
   int ncluster=0;     		// core+halo pixels
   int ccluster=0;     		// core+halo pixels associated to each core pixel
   ncore=ss.GetSparseSize();
   ncluster=ss.GetSparseSize(0);
   ccluster = 2*(ss.GetHaloSlice()+ss.GetHaloSlice(true))+1;
   ccluster*= 2*ss.GetHaloLayer()+1;
   cout << "ncore    : " << ncore << endl;
   cout << "ncluster : " << ncluster << endl;
   cout << "ccluster : " << ccluster << endl;

#ifdef TEST_WRITE_READ_SPARSE_MAP
   // write sparse map to root file
   TFile ofile("SparseMapTest.root","RECREATE");
   ss.Write("sparseMap");
   ofile.Close();

   // read sparse map from root file
   TFile ifile("SparseMapTest.root");
   SSeries<double>* pss = (SSeries<double>*)ifile.Get("sparseMap");
   ss = *pss;
   delete pss;
   ifile.Close();
#endif

   // rebuild wseries from sparse table only with core pixels
   bool core = true;
   ss.Expand(core);    	
   // plot the TF ss map as energy average of 0 and 90 degree phases (quadratures)
   canvas->cd(3);
   gPad->SetGridx(); gPad->SetGridy();
   plot->hist2D=NULL;
   plot->plot(ss);
   plot->hist2D->SetName("Sparse-3");;
   sprintf(title,"%s - Res : %s","TF Map : Core Pixels",tfres);
   plot->hist2D->SetTitle(title);;

   // rebuild wseries from sparse table with core+halo pixels
   bool core = false;
   ss.Expand(core);    	
   // plot the TF ss map as energy average of 0 and 90 degree phases (quadratures)
   canvas->cd(4);
   gPad->SetGridx(); gPad->SetGridy();
   plot->hist2D=NULL;
   plot->plot(ss);
   plot->hist2D->SetName("Sparse-4");;
   sprintf(title,"%s - Res : %s","TF Map : Core + Halo Pixels",tfres);
   plot->hist2D->SetTitle(title);;

   return canvas;
}
