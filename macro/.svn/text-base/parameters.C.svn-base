{
  int nIFO = 3;            // number of detectors from the array ifo[]
  int runID = 0;           // run number, set in the production job
  char search = 'B';       // see description below

  char ifo[5][8]={"L1","H1","V1","H2","G1"};
  char refIFO[4]= "L1";    // reference IFO

  // cWB settings

  double bpp   = 0.0001;   // probability for pixel selection
  double Tgap  = 0.05;     // time gap between clusters (sec)
  double Fgap  = 128.;     // frequency gap between clusters (Hz)
  double fLow  = 64.;      // low frequency of the search
  double fHigh = 2048.;    // high frequency of the search
  double Acore = sqrt(2.); // threshold for selection of core pixels
  double Tlpr  = 120.;     // training time for LPR filter

  double x2or  = 2.0;      // 2 OR threshold
  double netRHO= 3.0;      // threshold on rho
  double netCC = 0.5;      // threshold on network correlation  

  // wavelet transformation settings
   
  int levelR   = 2;        // resampling level
  int levelF   = 6;        // level where second LPR filter is applied
  int levelD   = 8;        // decomposition level
  int l_low    = 3;        // low frequency resolution level
  int l_high   = 8;        // high frequency resolution level
  double waveoffset = 8.;  // wavelet boundary offset [sec]

  // time shift analysis settings
  
  size_t  lagSize    = 20;    // number of lags
  double  lagStep    = 1.;    // time interval between lags [sec]
  size_t  lagOff     = 0;     // first lag id (lagOff=0 - include zero lag )
  size_t  lagMax     = 300;   // 0/>0 -  standard/extended lags
  char    lagMode[]  = "w";   // w/r  -  write/read lag list
  char*   lagFile    = NULL;  // file name to dump lags
  size_t* lagSite    = NULL;  // site index starting with 0
  
  // calibration DC corrections - ignored in net.C if value<=0. 
  double dcCal[5]   = {0.89, 1., 1., 1., 1.};

  // simulation parameters

  int simulation = 0;      // 1 for simulation, 0 for production
  int   nfactor = 1;       // number of strain factors
  double factors[] ={1.0}; // array of strain factors
  double gap = 5.;         // analysis time window for injections

  // delay filter
 
  char filter[1024] = "up2";   // delay filter suffix: "", or "up1", or "up2"

  // regulator
 
  double delta = 1.0;       // 0->1: weak-soft regulator
  double gamma = 0.2;       // 0->1: weak-hard regulator; if <0 add superclean option
    
  // sky settings

  bool   EFEC   =   true;  // Earth Fixed / Selestial coordinates
  size_t mode   =   0;     // sky search mode
  double angle  =   0.5;   // angular resolution
  double Theta1 =   0.;    // start theta
  double Theta2 = 180.;    // end theta
  double Phi1   =   0.;    // start theta
  double Phi2   = 360.;    // end theta
  double mask   =   0.;    // sky mask fraction

  // error regions settings

  double precision = 0.001; //  No = nIFO*(K+KZero)+precision*E

  // file dump mode

  bool dump = false;       // dump triggers into ascii file
  bool cedDump = false;    // dump ced plots with rho>cedRHO
  double cedRHO = 3.0;     

  // directories, file names

  char data_dir[1024]  = "/usr1/igor/waveburst/data64";
  char nodedir[1024] = "";
  char finalSegmentList[1024]="";
  char injectionList[1024]="";
  char skyMaskFile[1024]="";
  char fileNamesRaw[5][50]={"llo.lst.","lho1.lst.","lho2.lst.","-","-"};
  char channelNamesRaw[5][50]={"L1:LSC-STRAIN","H1:LSC-STRAIN","H2:LSC-STRAIN","-","-"};
  char channelNamesMDC[5][50]={"L1:GW-H","H1:GW-H","H2:GW-H","-","-"};
  char input_dir[512];     // input directory
  char input_label[512];   // input label
  char output_dir[512];    // output directory
  char output_label[512];  // output label

  // statistics:
  // L  - likelihood
  // c  - network correlation coefficient
  // E  - total energy in the data streams

  // search modes
  // 'B' - un-modeled search, single stream, max(L*c/E) 
  // 'b' - un-modeled search, single stream, max(L*c/E) 
  // 'R' - un-modeled search, dual stream, max(L*c/E) 
  // 'r' - un-modeled search, dual stream, max(L*c/E) 
  // 'I' - elliptical plarisation, max(L*c/E) 
  // 'S' - linear plarisation, max(L*c/E) 
  // 'G' - circular plarisation, max(L*c/E) 
  // 'i' - elliptical plarisation, max(L*c/E) 
  // 's' - linear plarisation, max(L*c/E) 
  // 'g' - circular plarisation, max(L*c/E) 

}
