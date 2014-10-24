{
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// set parameters
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char* suffix = "sim2x";

int runID         = 0;      // run number

double bpp        = 0.10;   // black pixel probability
double cWindow    = 0.025;  // councidence window [sec]
double cThreshold = 0.5;    // coincidence threshold
double onePixelT  = 0.0;    // threshold on one-pixel clusters
bool halo         = true;   // processing of halo pixels

double start = 0;           // start time of MC
//double start = 751974320  // start time of 4000 sec
//double start = 729337293; // start time: first PG set
double duration = 60*4;    // duration [sec]
double waveoffset = 4.;     // wavelet boundary offset [sec]
double skip  = 0.;          // skip duration

int first=9;                // first level
int last =9;                // last  level
int levelR=3;               // resampling level
int levelD=9;               // decomposition level
int lpfcut=64;              // low pass filter cut-off [Hz]

int lags=1;                // number of time lags including zero lag 
double step=4.25;           // time interval between lags [sec]

double offset;
double shift;               // time shift between series
int N,n;                    // number of samples in duration seconds
int M,m;                    // number of samples in offset seconds
int lag;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//Biorthogonal<double> B(30); // set wavelet for decimation
Symlet<double>     B(60);   // set wavelet for production
Symlet<double>     S(60,1); // set wavelet for production

wavecluster cX[lags];       // array of cluster structures for L1
wavecluster cY[lags];       // array of cluster structures for H1 
wavecluster cZ[lags];       // array of cluster structures for H2
wavecluster CX,CY,CZ;
wavecluster* p[3];

wavearray<float> vx,vy,vz;  // noise variability
WSeries<double>  nX,nY,nZ;  // noise rms

wavearray<double> x;        // temporary time series
wavearray<double> y(16394); // time series for injections
WSeries<double> wB(B);      // original WSeries
WSeries<double> wX(S);      // L1 wavelet series
WSeries<double> wY(S);      // H1 wavelet series
WSeries<double> wZ(S);      // H2 wavelet series
WSeries<double> WX,WY,WZ;   // rank amplitudes
WSeries<double> VX,VY,VZ;   // rank amplitudes


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// initialization of output root file 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char outFile[512];
sprintf(outFile,"wave_%d_%d_%d_%d%s.root",start,int(100.*bpp),int(10.*cThreshold),int(10*onePixelT),suffix);
cout<<"output file name: "<<outFile<<endl;
TFile *froot = new TFile(outFile, "RECREATE");          // root file
wbsingle waveburst;
TTree* wave_tree = waveburst.setTree();
variability wavevar;
TTree* var_tree = wavevar.setTree();
wavenoise noiserms;
TTree* noise_tree = noiserms.setTree();


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input data
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

y.rate(1024*16);
y=0.;
addWGNoise(y,0.3,0.1);

char file[512];

// L1

//sprintf(file,"/cdf/tmp1/ligosoft/sazonov/work2004/frames/L-AS_Q1-%d-16.gwf",int(start));
//ReadFrFile(x,duration+2*waveoffset,skip,"L1:LSC-AS_Q",file);

//sprintf(file,"/cdf/tmp1/ligosoft/S2/v5/frames/HL-SG5-%d-640.gwf",int(start));
//ReadFrFile(x,duration+2*waveoffset,skip,"L1:LSC-AS_Q",file,false,"proc");

x.resize(int(duration+2*waveoffset+0.1)*1024*16);
x=0.; x.rate(1024*16);
addGauss(x,1.,0.);
x.add(y,y.size(),0,(x.size()-y.size())/2);

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wX.Forward(x,levelD);
wX.median(8.,true);
nX=wX.white(60.);
vx=wX.variability(1.);

// H1

//sprintf(file,"/cdf/tmp1/ligosoft/sazonov/work2004/frames/H-AS_Q1-%d-16.gwf",int(start));
//ReadFrFile(x,duration+2*waveoffset,skip,"H1:LSC-AS_Q",file);

//sprintf(file,"/cdf/tmp1/ligosoft/S2/v5/frames/HL-SG5-%d-640.gwf",int(start));
//ReadFrFile(x,duration+2*waveoffset,skip,"H1:LSC-AS_Q",file,false,"proc");

x.resize(int(duration+2*waveoffset+0.1)*1024*16);
x=0.; x.rate(1024*16);
addGauss(x,1.,0.);
x.add(y,y.size(),0,(x.size()-y.size())/2);

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wY.Forward(x,levelD);
wY.median(8.,true);
nY=wY.white(60.);
vy=wY.variability(1.);

// H2

//sprintf(file,"/cdf/tmp1/ligosoft/sazonov/work2004/frames/H2-AS_Q1-%d-16.gwf",int(start));
//ReadFrFile(x,duration+2*waveoffset,skip,"H2:LSC-AS_Q",file);

//sprintf(file,"/cdf/tmp1/ligosoft/S2/v5/frames/HL-SG5-%d-640.gwf",int(start));
//ReadFrFile(x,duration+2*waveoffset,skip,"H2:LSC-AS_Q",file,false,"proc");

x.resize(int(duration+2*waveoffset+0.1)*1024*16);
x=0.; x.rate(1024*16);
addGauss(x,1.,0.);
x.add(y,y.size(),0,(x.size()-y.size())/2);

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wZ.Forward(x,levelD);
wZ.median(8.,true);
nZ=wZ.white(60.);
vz=wZ.variability(1.);

printf("L1: %f, H1: %f, H2: %f\n",nX.start(),nY.start(),nZ.start());
printf("L1: %f, H1: %f, H2: %f\n",vx.start(),vy.start(),vz.start());

double R = x.rate();              // original data rate
double f_res = R/2./(1<<levelD);  // frequency resolution

N = int(R*duration+0.5);      // zero lag duration
M = int(R*waveoffset+0.5);    // zero lag offset

wB.resize(1);
 x.resize(1);

cout<<wX.size()<<" "<<wY.size()<<" "<<wZ.size()<<endl<<endl;

int i,j,k,l;
int low = (int(R+0.5)>>(levelD))/2; // finest frequency resolution

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// level and lag loop
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

slice S0;

for(i=1; i<=last; i++){

   // correct first layer when get to respective resolution
   if(lpfcut == int(f_res+0.5)) {
      cout<<"apply lpf filter: "<<f_res<<"  i="<<i<<endl;
      wX.getLayer(x,0); x = 0.; wX.putLayer(x,0);  // zero level 0
      wY.getLayer(x,0); x = 0.; wY.putLayer(x,0);  // zero level 0
      wZ.getLayer(x,0); x = 0.; wZ.putLayer(x,0);  // zero level 0
      wX.getLayer(x,1); x*=1.41; wX.putLayer(x,1);
      wY.getLayer(x,1); x*=1.41; wY.putLayer(x,1);
      wZ.getLayer(x,1); x*=1.41; wZ.putLayer(x,1);
      x.resize(1);
   }

   wX.Inverse(1); 
   wY.Inverse(1); 
   wZ.Inverse(1); 
   S0=wX.getSlice(0);                    // 0 layer slice

   f_res *= 2.;                          // resolution at this step
   cout<<"resolution: "<<f_res<<"  i="<<i<<endl;
   if(i<first) continue;                 // skip first levels

//+++++++++++ lag definitions +++++++++++++++++++++++++
// o - start of zero lag interval
// lag>0:   o----------          L1
//              o-----------     H1,H2
//
// lag<0:       o----------      L1
//          o-----------         H1,H2
//+++++++++++++++++++++++++++++++++++++++++++++++++++++

   for(j=0; j<lags; j++){                // L1 time shifts

      lag = j - lags/2;
      shift = fabs(lag*step);            // time shift due to lag

      cout<<"process: level="<<i;
      cout<<",  lag="<<lag<<",  shift="<<lag*step<<endl;

      k = int(R*shift+0.5);              // offset due to lag
      n = N-k;
      m = lag>=0 ? M : M+k;              // lag<0 - shift L1 (first channel) 

      if((m/S0.stride())*S0.stride() != m) {
	 cout<<"wb3: time shift error";
	 cout<<" stride="<<S0.stride()<<"  offset="<<m<<endl;
      }

      WX=wX[slice(m,n,1)]; 

      m = lag>=0 ? M+k : M;               // lag>0 shift H1,h2 (second channel)
      if((m/S0.stride())*S0.stride() != m) cout<<"wb3: time shift error\n";

      WY=wY[slice(m,n,1)]; 
      WZ=wZ[slice(m,n,1)]; 


//      WX.percentile(bpp,1);
//      WY.percentile(bpp,1);
//      WZ.percentile(bpp,1);

/*
      cout<<WX.fraction(60.,bpp,1)<<"  ";
      cout<<WY.fraction(60.,bpp,1)<<"  ";
      cout<<WZ.fraction(60.,bpp,1)<<endl;

      cout<<WX.significance((1<<i)*32)<<"  ";
      cout<<WY.significance((1<<i)*32)<<"  ";
      cout<<WZ.significance((1<<i)*32)<<endl;

      cout<<"  pixel occupancy: ";
      printf("%5.2e  ",WX.Coincidence(WY,cWindow,cThreshold));
      printf("%5.2e  ",WY.Coincidence(WX,cWindow,cThreshold));
      printf("%5.2e  ",WZ.Coincidence(WY,0.,cThreshold)); cout<<endl;
      cout<<"  pixel occupancy: ";
      printf("%5.2e  ",WX.Coincidence(WZ,cWindow,cThreshold));
      printf("%5.2e  ",WY.Coincidence(WZ,0.,cThreshold));
      printf("%5.2e  ",WZ.Coincidence(WX,cWindow,cThreshold)); cout<<endl;
      cout<<"  pixel occupancy: ";
      printf("%5.2e  ",WX.Coincidence(WY,cWindow,cThreshold));
      printf("%5.2e  ",WY.Coincidence(WX,cWindow,cThreshold));
      printf("%5.2e  ",WZ.Coincidence(WY,0.,cThreshold)); cout<<endl;

      cout<<"  final occupancy: ";
      printf("%5.2e  ",WX.pixclean(onePixelT));
      printf("%5.2e  ",WY.pixclean(onePixelT));
      printf("%5.2e  ",WZ.pixclean(onePixelT)); cout<<endl;

      offset = lag>=0 ? waveoffset : waveoffset+shift;  

      CX.init(WX,halo);
      CX.ifo = 1; CX.shift = lag*step; CX.run = runID;
      CX.apush(wX,offset); 
      CX.cluster(); CX.cleanhalo();

      offset = lag>=0 ? waveoffset+shift : waveoffset;  

      CY.init(WY,halo); 
      CY.ifo = 2; CY.shift = lag*step; CX.run = runID;
      CY.apush(wY,offset);
      CY.cluster(); CY.cleanhalo();

      CZ.init(WZ,halo); 
      CZ.ifo = 3; CZ.shift = lag*step; CX.run = runID;
      CZ.apush(wZ,offset);
      CZ.cluster(); CZ.cleanhalo();
      
      cout<<"cluster list size: ";
      printf(" L1: %5d",CX.size());
      printf(" H1: %5d",CY.size());
      printf(" H2: %5d",CZ.size()); cout<<endl;

      cout<<"  total list size: ";
      printf(" L1: %5d",cX[j].append(CX));
      printf(" H1: %5d",cY[j].append(CY));
      printf(" H2: %5d",cZ[j].append(CZ)); 
      cout<<endl<<endl;
*/
   }
}

/*
int counts = 0;

for(j=0; j<lags; j++){

   cout<<"lag="<<j<<"  ";
   cout<<cX[j].merge()<<" ";
   cout<<cY[j].merge()<<" ";
   cout<<cZ[j].merge()<<endl;

//   printf("L1: start=%f, stop=%f\n",cX[j].start,cX[j].stop);
//   printf("H1: start=%f, stop=%f\n",cY[j].start,cY[j].stop);
//   printf("H2: start=%f, stop=%f\n",cZ[j].start,cZ[j].stop);

   p[0] = &cX[j];
   p[1] = &cY[j];
   p[2] = &cZ[j];

   waveburst.output(wave_tree,p);

   cX[j].coincidence(cY[j],2*cWindow);
   cY[j].coincidence(cZ[j],2*cWindow);
   cZ[j].coincidence(cX[j],2*cWindow);
   cout<<cX[j].coincidence(cZ[j],2*cWindow)<<" ";
   cout<<cY[j].coincidence(cX[j],2*cWindow)<<" ";
   cout<<cZ[j].coincidence(cY[j],2*cWindow)<<endl;

}

wavevar.output(var_tree,&vx,1,waveoffset);
wavevar.output(var_tree,&vy,2,waveoffset);
wavevar.output(var_tree,&vz,3,waveoffset);
noiserms.output(noise_tree,&nX,1,R/2);
noiserms.output(noise_tree,&nY,2,R/2);
noiserms.output(noise_tree,&nZ,3,R/2);
froot->Write();
froot->Close();

*/
}







