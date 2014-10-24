{
  char ifo[4][8]={"L1","H1","H2","G1","V1"};

  // WB threshold settings

  double bpp   = 1.e-5;    // probability for pixel selection
  double Acore = sqrt(2);  // threshold for selection of core pixels
  double Tgap  = 0.05;     // time gap between clusters (sec)
  double Fgap  = 128.;     // frequency gap between clusters (Hz)
  double fLow  = 64.;      // low frequency of the search

  double x2or  = 2.0;      // 2 OR threshold
  double eCOR  = 3.;      // threshold on correlated energy
  double netCOR= 0.3;      // threshold on network correlation  

  // constraint
 
  double gamma = 2.;      // snr dependent constraint 
  double delta = 0.02;    // fixed lengh constraint
  
  // wavelet transformation settings
  
  int levelR   = 2;        // resampling level
  int levelD   = 8;        // decomposition level
  int l_low    = 3;        // low frequency resolution level
  int l_high   = 8;        // high frequency resolution level
  double waveoffset = 8.;  // wavelet boundary offset [sec]

  // time shift analysis settings
  
  int lags     = 101;      // number of lags including zero lag 
  double step  = 3.125;    // time interval between lags [sec]
                           //  L1  H1  H2  G1 constant time shifts 
  double shift[5] =         {0., 0., 0., 0., 0.};
  
  // simulation parameters

  int simulation = 0;      // 1 for simulation, 0 for production
  int   nfactor = 1;       // number of strain factors
  double factors[] ={1.0}; // array of strain factors

  // directories, files

  char data_dir[]  = "/usr1/igor/waveburst/data64";
  char nodedir[] = "/archive/home/klimenko/s5/runS5a";
  char finalSegmentList[]="segs_12a.out";
  char injectionList[]="";

  char fileNamesRaw[4][50]={"llo.lst.","lho1.lst.","lho2.lst.","geo.lst."};

  char channelNamesRaw[4][50]={"L1:LSC-STRAIN","H1:LSC-STRAIN","H2:LSC-STRAIN","G1:DER_DATA_H"};

  char channelNamesMDC[4][50]={"L1:GW-H","H1:GW-H","H2:GW-H","G1:GW-H"};

  double factor = 1.0;     // strain factor
  int runID    = 0;        // run number, set in the production job
  char input_dir[512];     // input directory
  char input_label[512];   // input label
  char output_dir[512];    // output directory
  char output_label[512];  // output label

  // sky settings
  double angle  =   1.;    // angular resolution
  double Theta1 =   0.;    // start theta
  double Theta2 = 180.;    // end theta
  double Phi1   =   0.;    // start theta
  double Phi2   = 360.;    // end theta
}
