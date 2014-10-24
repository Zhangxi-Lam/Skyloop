{  //gSystem->Load("Functor_cc.so");
   gROOT->LoadMacro("macro/AddPulse.C");
   gStyle->SetPalette(1,0);
   gStyle->SetOptStat(0);
   double PI = TMath::Pi();
   
   int TLen = 8;
   wavearray<double> ts(1024*2*TLen);
   //   ts.data[ts.size()/2+0]=1.;
   ts.rate(1024*2);
   
   addSGBurst(ts, 1, 235, 0.01);
   //addGauss(ts, 1, 0);
   
   Plot(ts);
   
   WDM<double> wdtf(512/2, 512, 2, 6); 
   
   WSeries<double> tfmap(wdtf), tfmap2(wdtf), tfmap3(wdtf);
   tfmap.Forward(ts, -1);   
   tfmap2.Forward(ts, -1);  
   
   WDM<double>* pw = (WDM<double>*)tfmap->pWavelet;
   WDM<double>* pw2 = (WDM<double>*)tfmap2->pWavelet;
   pw2->setTDFilter(8, 4);
   
   //system("date");
   //pw2->TimeShiftTest(10);
   //system("date");
   
   pw->SetTFMap();
   for(int m = 20; m<100; ++m)for(int n=12; n<52; ++n){
      //printf("%d %d\n", m, n);
      pw->TFMap00[n][m] = pw2->getPixelAmplitude(m,n, 99);
      pw->TFMap90[n][m] = pw2->getPixelAmplitude(m,n, 99, true);

   }
   
   WTSpectrum(tfmap, 1);
	WTSpectrum(tfmap2, 1);
   
   tfmap.Inverse();
   
   wavearray<double> tsInv(1024*2*TLen);
   tsInv.rate(1024*2);
   for(int i=0; i<TLen*2048; ++i)tsInv[i] = pw->pWWS[i];
   //tsInv -= ts;
   Plot(tsInv,1,2);
   
   tfmap3.Forward(tsInv, -1);  
   WTSpectrum(tfmap3, 1);
}
