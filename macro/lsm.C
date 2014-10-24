{

TF1 *chi = new TF1("chi",ChiSquare,0.,50,2);
chi->SetParameters(4000,2.);
chi->SetParNames ("max","degree"); 
//chi->SetParNames ("max","scale","degree"); 

TF1 *logn = new TF1("logn",LogNorm,0.,10,4);
logn->SetParameters(4000,1,0.3,0.01);
logn->SetParNames ("scale","peak","sigma","as"); 


cout<<"Starting the job"<<endl;
gSystem->Exec("date");
gSystem->Exec("hostname");

int i,j,n,m;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// set parameters
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char* suffix = "tst2d1";
char filtername[128];

int runID         = 0;       // run number

// WB threshold settings

double bpp        = 0.001;   // black pixel probability (for rank statistic)
double Ac         = 1.67;    // Likelihood threshold (in units of noise rms)
double Tgap       = 0.03;    // time gap between clusters
double Fgap       = 64.;     // frequency gap between clusters

// wavelet transform settings

int levelR= 2;               // resampling level
int levelD= 8;               // decomposition level
int l_low = 3;               // low frequency resolution level
int l_high= 8;               // high frequency resolution level
int lpfcut=64;               // low pass filter cut-off [Hz]
int hpfcut=512;              // high pass filter cut-off [Hz]
int w_mode=1;                // whitening mode
double waveoffset = 8.;      // wavelet boundary offset [sec]

// time shift analysis settings

int lags= 21;                // number of time lags including zero lag 
double step=3.25;            // time interval between lags [sec]

// logNormal parameters

// sqrt((a1*a1+a2*a2+a3*a3)/3)
double Ao;
// max delay 0.043 sec
//double pln[27]={2.700,0.254,0.200,   /* level 1 */
//		2.500,0.274,0.200,   /* level 2 */
//              2.304,0.298,0.171,   /* level 3 */
//		2.144,0.322,0.137,   /* level 4 */
//		1.965,0.336,0.145,   /* level 5 */
//		1.772,0.356,0.142,   /* level 6 */
//		1.587,0.371,0.146,   /* level 7 */
//		1.430,0.380,0.156};  /* level 8 */

// max delay 0.011 sec
double pln[27]={2.700,0.254,0.200,   /* level 1 */
		1.950,0.320,0.130,   /* level 2 */
                1.891,0.334,0.145,   /* level 3 */
		1.714,0.357,0.138,   /* level 4 */
		1.546,0.370,0.144,   /* level 5 */
		1.400,0.378,0.156,   /* level 6 */
		1.297,0.382,0.169,   /* level 7 */
		1.215,0.386,0.179};  /* level 8 */

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Meyer<double>         B(1024);   // set wavelet for production
Meyer<double>         S(1024,2); // set wavelet for production

netcluster WC[lags];        // array of cluster structures
netcluster wc;           
netcluster* pwc;

WSeries<float>   vx,vy,vz;   // noise variability
WSeries<double>  nX,nY,nZ;   // noise rms

wavearray<double> yp(16384); // time series for injections
wavearray<double> yx(16384); // time series for injections
wavearray<double> x,y;       // temporary time series
WSeries<double> wB(B);       // original WSeries


detector        L1("L1");   // detector
detector        H1("H1");   // detector
detector        H2("H2");   // detector

network         NET;        // network

NET.add(&L1); NET.add(&H1); NET.add(&H2); // add detectors to network
NET.setSkyMaps(1.);                       // set network skymaps
NET.setTimeShifts(lags,step);
NET.setIndex(&H1);
NET.setAntenna(&L1); 
NET.setAntenna(&H1); 
NET.setAntenna(&H2);
NET.setbpp(0.1);
NET.setRunID(11);
NET.Edge = waveoffset;

//NET.readMDClog("../data/BurstMDC-SG21_V4_S4-Log.txt");
//NET.readSEGlist("segL.txt",2);

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// initialization of output root file 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char outFile[512];
sprintf(outFile,"wave_%d%s.root",int(100.*bpp),suffix);
cout<<"output file name: "<<outFile<<endl;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input data
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

wavearray<double> xw(16384*2); // time series for injections
wavearray<double> yw(16384*2); // time series for injections
wavearray<double> zw(16384*2); // time series for injections
xw.rate(1024*16); xw=0.;
yw.rate(1024*16); yw=0.;
zw.rate(1024*16); zw=0.;

char file[512];

// L1

sprintf(file,"/home/klimenko/wat/wat-4.0.6/plots/llo_l1.lst");
readframes(file,"L1:LSC-STRAIN",x);

double start=x.start();
double rate=x.rate();
double end=start+x.size()/rate;
double duration=end-start-2*waveoffset;

wB.Forward(x,levelR); 
wB.getLayer(x,0); 
L1.getTFmap()->Forward(x,S,levelD);
L1.getTFmap()->setlow(64.);
L1.getTFmap()->sethigh(512.);
L1.getTFmap()->lprFilter(1,0,120.,4.);
L1.white(100.,w_mode,8.,20.);
vx=L1.getTFmap()->variability();


// H1

sprintf(file,"/home/klimenko/wat/wat-4.0.6/plots/lho_h1.lst");
readframes(file,"H1:LSC-STRAIN",x);

start=x.start();
rate=x.rate();
end=start+x.size()/rate;
duration=end-start-2*waveoffset;

wB.Forward(x,levelR);
wB.getLayer(x,0);
H1.getTFmap()->Forward(x,S,levelD);
H1.getTFmap()->setlow(64.);
H1.getTFmap()->sethigh(512.);
H1.getTFmap()->lprFilter(1,0,120.,4.);
H1.white(100.,w_mode,8.,20.);
vy=H1.getTFmap()->variability();


// H2

sprintf(file,"/home/klimenko/wat/wat-4.0.6/plots/lho_h2.lst");
readframes(file,"H2:LSC-STRAIN",x);

start=x.start();
rate=x.rate();
end=start+x.size()/rate;
duration=end-start-2*waveoffset;

wB.Forward(x,levelR);
wB.getLayer(x,0);
H2.getTFmap()->Forward(x,S,levelD);
H2.getTFmap()->setlow(64.);
H2.getTFmap()->sethigh(512.);
H2.getTFmap()->lprFilter(1.,0,120.,4.);
H2.white(100.,w_mode,8.,20.);
vz=H2.getTFmap()->variability();

printf("L1: %f, H1: %f, H2: %f\n",nX.start(),nY.start(),nZ.start());
printf("L1: %f, H1: %f, H2: %f\n",vx.start(),vy.start(),vz.start());

cout<<"live time: "<<NET.setVeto(16.)<<endl;  // set veto array for injections

double R = x.rate();                         // original data rate
double scale;
double* pLN;


wB.resize(1);
 x.resize(1);


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// low pass filtering
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int low = (int(R+0.5)>>(levelD))/2; // finest frequency resolution
n = lpfcut/low;                     // number of layers to zero

for(i=0; i<n; i++){
   L1.getTFmap()->getLayer(x,i); x = 0.; L1.getTFmap()->putLayer(x,i);  // zero level i
   H1.getTFmap()->getLayer(x,i); x = 0.; H1.getTFmap()->putLayer(x,i);  // zero level i
   H2.getTFmap()->getLayer(x,i); x = 0.; H2.getTFmap()->putLayer(x,i);  // zero level i
}

sprintf(filtername,"/home/klimenko/wat/wat-4.2.0/data/Meyer1024_L8.dat");
cout<<filtername<<endl;
L1.readFilter(filtername); H1.setFilter(L1); H2.setFilter(L1); 
NET.setFilter(&H1,0.011); NET.setDelay(&H1); NET.setDelay(&H2);

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// loop over time windows
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char name[64];

for(i=18; i<19; i++) {
  NET.initwc(12.+i*0.05,0.1);   
  NET.setRMS();
  sprintf(name,"lsm_%d.dat",i);
  cout<<"  pixels: "<<NET.likelihood3('L',false,0.,1,0);
  NET.nLikelihood.DumpBinary(name,0);
}

}






