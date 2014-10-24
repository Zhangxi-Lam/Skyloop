#define OVERLAP_CATALOG "OverlapCatalog_Lev_32_64_128_256_512_1024_iNu_4_Prec_10.bin"

{ 
  // define resolutions of interest:   
  int layers[6] = {32, 64, 128, 256, 512, 1024};
  
  // create corresponding WDM transforms:
  WDM* wdm[6];
  for(int i=0; i<6; i++)wdm[i] = new WDM(layers[i], layers[i], 4, 10);

  // create the catalog:
  monster x(wdm, 6);

  // write the catalog in a file for future use:
  x.write(OVERLAP_CATALOG);
} 
