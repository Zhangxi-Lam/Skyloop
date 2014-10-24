{  gSystem->CompileMacro("revMonster.cc");
   
   // create the time series and the corresponding TF Maps for 7 resolutions
   
   init();
   
   // here we subtract the signal in TFMap_1 from the signal in TFMap_2
   // using the overlap coefficients in the catalog accessed via the 'monster' class.
   // this code assumes tfmap1 > tfmap2 but in general there is no such restriction
   // for testing purposes, this is fine
   
   int tfmap1 = 2; 
   int tfmap2 = 1; 
   
   testMonster(tfmap1, tfmap2);
   
   // plot TFMap_1 , TFMap_2, and the result of the subtraction:
    
   watplot p1("TFMap1"); 
   p1.plot(pTF[tfmap1], 4, 0, 10);
   
   watplot p2("TFMap2"); 
   p2.plot(pTF[tfmap2], 4, 0, 10);
   
   watplot pdiff("subTFMap2"); 
   pdiff.plot(subTF, 4, 0, 10);
   
   
}
