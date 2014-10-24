#define OVERLAP_CATALOG "MyOverlapCatalog5.bin"

{    
  int layers[6] = {16, 32, 64, 128, 256, 512};
  WDM<double>* wdm[6];
  for(int i=0; i<6; i++)wdm[i] = new WDM<double>(layers[i], layers[i], 4, 8);

  monster x(wdm, 6);
  x.write(OVERLAP_CATALOG);
} 
