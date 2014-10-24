{
  int simulation = 0;    //1 for simulation, 0 for production
  int runID    = 0;      // run number

  // WB threshold settings

  double bpp   = 0.001;  // probability for pixel selection
  double Acore = 1.67;   // threshold for selection of core pixels
  double Tgap  = 0.04;   // time gap between clusters (sec)
  double Fgap  = 128.;   // frequency gap between clusters (Hz)
  
  // wavelet transform settings
  
  int levelR   = 2;      // resampling level
  int levelD   = 8;      // decomposition level
  int l_low    = 3;      // low frequency resolution level
  int l_high   = 8;      // high frequency resolution level
  int lpfcut   =64;      // low pass filter cut-off [Hz]
  int w_mode   = 1;      // whitening mode
  double waveoffset = 8.;// wavelet boundary offset [sec]
  bool parity = true;    // use odd-parity wavelet

  // time shift analysis settings
  
  int lags     = 101;      // number of lags including zero lag 
  double step=3.25;      // time interval between lags [sec]
                         //  L1  H1  H2  G1 constant time shifts 
  double shift[3] =         {0., 0., 0.};
  
  // logNormal parameters

  double pln[27]={2.700,0.254,0.200,   /* level 1 */
		  2.500,0.274,0.200,   /* level 2 */
		  2.304,0.298,0.171,   /* level 3 */
		  2.144,0.322,0.137,   /* level 4 */
		  1.965,0.336,0.145,   /* level 5 */
		  1.772,0.356,0.142,   /* level 6 */
		  1.587,0.371,0.146,   /* level 7 */
		  1.430,0.380,0.156};  /* level 8 */
   
  // simulation parameters

  int   nfactor = 1;
  double factor = 1.0;
  double factors[] ={1.0};

  // directories, files

  char data_dir[]  = "/usr1/igor/waveburst/data64";
  char nodedir[50] = "/usr1/igor/waveburst";
  char finalSegmentList[100]="1146592.542-S4H1H2L1G1_final_DQ.A.txt.plain";
  char injectionList[50]="BurstMDC-SG21_V4_S4-Log.txt";

  char fileNamesRaw[4][50]={"llo.lst.","lho1.lst.",
			    "lho2.lst.","geo.lst."};

  char channelNamesRaw[4][50]={"L1:LSC-STRAIN","H1:LSC-STRAIN",
			       "H2:LSC-STRAIN","G1:DER_DATA_H"};

  char channelNamesMDC[4][50]={"L1:GW-H","H1:GW-H",
			       "H2:GW-H","G1:GW-H"};

  double start, end, rate, duration, Ao;
  double *pLN;
}
