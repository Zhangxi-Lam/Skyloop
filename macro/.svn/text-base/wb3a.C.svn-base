{
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// set parameters
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char* suffix = "5";

int runID         = 0;      // run number

// WB thresholds

double bpp        = 0.10;   // black pixel probability
double cWindow    = 0.032;  // councidence window [sec]
double cThreshold = 1.0;    // coincidence threshold
double onePixelT  = 0.0;    // threshold on one-pixel clusters
bool halo         = true;   // processing of halo pixels

// input data settings

double start = 751974320;   // start time of 4000 sec
//double start = 729337293; // start time: first PG set
double duration = 60*2;     // duration [sec]
double waveoffset = 4.;     // wavelet boundary offset [sec]
double skip  = 0.;          // skip duration

// wavelet transformation settings

int levelR= 2;              // resampling level
int levelD= 10;              // decomposition level
int l_low = 7;              // low frequency resolution level
int l_high= 2;              // high frequency resolution level
int lpfcut=64;              // low pass filter cut-off [Hz]

// data conditioning settings


// time shift analysis settings

int lags= 1;                // number of time lags including zero lag 
double step=4.25;           // time interval between lags [sec]

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

double offset;
double shift;               // time shift between series
int N,n;                    // number of samples in duration seconds
int M,m;                    // number of samples in offset seconds
int lag;

//Biorthogonal<double> B(30); // set wavelet for decimation
Symlet<double>     B(60);   // set wavelet for production
Symlet<double>     S(60,1); // set wavelet for production
Biorthogonal<double>     BB(60,1); // set wavelet for production

wavecluster cX[lags];       // array of cluster structures for L1
wavecluster cY[lags];       // array of cluster structures for H1 
wavecluster cZ[lags];       // array of cluster structures for H2
wavecluster CX,CY,CZ;
wavecluster* p[3];

wavearray<float> vx,vy,vz;  // noise variability
WSeries<double>  nX,nY,nZ;  // noise rms

wavearray<double> x;        // temporary time series
wavearray<double> y(16394); // temporary time series
WSeries<double> wB(B);      // original WSeries
WSeries<double> wX(S);      // L1 wavelet series
WSeries<double> wY(S);      // H1 wavelet series
WSeries<double> wZ(S);      // H2 wavelet series
WSeries<double> WX,WY,WZ;   // rank amplitudes

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// initialization of output root file 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char outFile[512];
sprintf(outFile,"wave_%d_%d_%d_%d%s.root",
	start,int(100.*bpp),int(10.*cThreshold),int(10*onePixelT),suffix);
cout<<"output file name: "<<outFile<<endl;

TFile *froot = new TFile(outFile, "RECREATE");          // root file
wbsingle waveburst;
TTree* wave_tree = waveburst.setTree();
variability wavevar;
TTree* var_tree = wavevar.setTree();
wavenoise noiserms;
TTree* noise_tree = noiserms.setTree();
xcsample xcor;
TTree* xcor_tree = xcor.setTree();


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input data
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

y.rate(1024*16);
y=0.;
//addWGNoise(y,200.5,0.1);
addSGBurst(y, 50., 100., 0.1);
double avr,rms;
y.getStatistics(avr,rms);
cout<<"signal hrss "<<sqrt(rms*rms*y.size()/y.rate())<<endl;

char file[512];

// L1

sprintf(file,"/cdf/tmp1/ligosoft/sazonov/work2004/frames/L-AS_Q1-%d-16.gwf",int(start));
ReadFrFile(x,duration+2*waveoffset,skip,"L1:LSC-AS_Q",file);

//sprintf(file,"/cdf/tmp1/ligosoft/S2/v5/frames/HL-SG5-%d-640.gwf",int(start));
//ReadFrFile(x,duration+2*waveoffset,skip,"L1:LSC-AS_Q",file,false,"proc");

x.add(y,y.size(),0,(x.size()-y.size())/2);

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wX.Forward(x,levelD);
nX=wX.white(60.);
wX.setlow(64.);
vx=wX.variability(1.);

//wX.Inverse();
//wX.lprFilter(0.25,4.);
//wX.Forward(levelD-l_high+1);

// H1

sprintf(file,"/cdf/tmp1/ligosoft/sazonov/work2004/frames/H-AS_Q1-%d-16.gwf",int(start));
ReadFrFile(x,duration+2*waveoffset,skip,"H1:LSC-AS_Q",file);

//sprintf(file,"/cdf/tmp1/ligosoft/S2/v5/frames/HL-SG5-%d-640.gwf",int(start));
//ReadFrFile(x,duration+2*waveoffset,skip,"H1:LSC-AS_Q",file,false,"proc");

x.add(y,y.size(),0,(x.size()-y.size())/2);

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wY.Forward(x,levelD);
//wY.lprFilter(2.,4.);
nY=wY.white(60.);
wY.setlow(64.);
vy=wY.variability(1.);
//wY.sethigh(wY.rate()/2.);
//wY.Inverse();
//wY.lprFilter(0.25,4.);
//wY.Forward(levelD-l_high+1);

// H2

sprintf(file,"/cdf/tmp1/ligosoft/sazonov/work2004/frames/H2-AS_Q1-%d-16.gwf",int(start));
ReadFrFile(x,duration+2*waveoffset,skip,"H2:LSC-AS_Q",file);

//sprintf(file,"/cdf/tmp1/ligosoft/S2/v5/frames/HL-SG5-%d-640.gwf",int(start));
//ReadFrFile(x,duration+2*waveoffset,skip,"H2:LSC-AS_Q",file,false,"proc");

x.add(y,y.size(),0,(x.size()-y.size())/2);

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wZ.Forward(x,levelD);
//wZ.lprFilter(2.,4.);
nZ=wZ.white(60.);
wZ.setlow(64.);
vz=wZ.variability(1.);
//wZ.sethigh(wY.rate()/2.);
//wZ.Inverse();
//wZ.lprFilter(0.25,4.);
//wZ.Forward(levelD-l_high+1);

/*
wavearray<float>* pxy;
wavearray<float>* pyz;
wavearray<float>* pzx;

wavecor xcXY;
wavecor xcZX;
wavecor xcYZ;

pxy = &(xcXY.xcor);
pyz = &(xcYZ.xcor);
pzx = &(xcZX.xcor);

wX.Inverse();
wY.Inverse();
wZ.Inverse();

WX=wX[slice(2048*4,2048*200,1)];
WY=wY[slice(2048*4,2048*200,1)];
WZ=wZ[slice(2048*4,2048*200,1)];

WX.exponential(0.25);
WY.exponential(0.25);
WZ.exponential(0.25);

xcXY.init(WX,WY,0.05,0.02,8);
xcYZ.init(WY,WZ,0.05,0.02,8);
xcZX.init(WZ,WX,0.05,0.02,8);

for(int i=0; i<pxy->size(); i++){if(pxy->data[i]<0) pxy->data[i]*=-1;}
for(int i=0; i<pyz->size(); i++){if(pyz->data[i]<0) pyz->data[i]*=-1;}
for(int i=0; i<pzx->size(); i++){if(pzx->data[i]<0) pzx->data[i]*=-1;}
*/


printf("L1: %f, H1: %f, H2: %f\n",nX.start(),nY.start(),nZ.start());
printf("L1: %f, H1: %f, H2: %f\n",vx.start(),vy.start(),vz.start());

double R = x.rate();              // original data rate
double f_res = R/2./(1<<(levelD-l_high+1));  // frequency resolution

N = int(R*duration+0.5);      // zero lag duration
M = int(R*waveoffset+0.5);    // zero lag offset

wB.resize(1);
 x.resize(1);

cout<<wX.size()<<" "<<wY.size()<<" "<<wZ.size()<<endl<<endl;



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// low pass filtering
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int i,j,k,l;
int low = (int(R+0.5)>>(levelD-l_high+1))/2; // finest frequency resolution

cout<<low<<endl;
k = lpfcut/low;                     // number of layers to zero
for(i=0; i<k; i++){
   wX.getLayer(x,i); x = 0.; wX.putLayer(x,i);  // zero level i
   wY.getLayer(x,i); x = 0.; wY.putLayer(x,i);  // zero level i
   wZ.getLayer(x,i); x = 0.; wZ.putLayer(x,i);  // zero level i
}



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// level and lag loop
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

slice S0;

for(i=1; i<=(l_low-l_high+1); i++){

// correct first layer when get to respective resolution
   if(lpfcut == int(f_res+0.5)) {
      cout<<"correct first layesr at: "<<f_res<<"  i="<<i<<endl;
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
//   if(i<l_high) continue;                // skip first levels

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

      WX.fraction(60.,bpp,1);
      WY.fraction(60.,bpp,1);
      WZ.fraction(60.,bpp,1);

      cout<<"  pixel occupancy: "<<endl;
      printf("%5.2e  ",WX.significance(8.,bpp));
      printf("%5.2e  ",WY.significance(8.,bpp));
      printf("%5.2e  ",WZ.significance(8.,bpp)); cout<<endl;

      cout<<"  pixel occupancy: ";
      printf("%5.2e  ",WX.Coincidence(WY,cWindow,cThreshold));
      printf("%5.2e  ",WY.Coincidence(WX,cWindow,cThreshold));
      printf("%5.2e  ",WZ.Coincidence(WY,cWindow,cThreshold)); cout<<endl;
      cout<<"  pixel occupancy: ";
      printf("%5.2e  ",WX.Coincidence(WZ,cWindow,cThreshold));
      printf("%5.2e  ",WY.Coincidence(WZ,cWindow,cThreshold));
      printf("%5.2e  ",WZ.Coincidence(WX,cWindow,cThreshold)); cout<<endl;
      cout<<"  pixel occupancy: ";
      printf("%5.2e  ",WX.Coincidence(WY,cWindow,cThreshold));
      printf("%5.2e  ",WY.Coincidence(WX,cWindow,cThreshold));
      printf("%5.2e  ",WZ.Coincidence(WY,cWindow,cThreshold)); cout<<endl;


      offset = lag>=0 ? waveoffset : waveoffset+shift;  

      CX.init(WX,halo);
      CX.ifo = 1; CX.shift = lag*step; CX.run = runID;
      CX.apush(wX,offset); 
      CX.cluster(); CX.cleanhalo();

      offset = lag>=0 ? waveoffset+shift : waveoffset;  

      CY.init(WY,halo); 
      CY.ifo = 2; CY.shift = lag*step; CY.run = runID;
      CY.apush(wY,offset);
      CY.cluster(); CY.cleanhalo();

      CZ.init(WZ,halo); 
      CZ.ifo = 3; CZ.shift = lag*step; CZ.run = runID;
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

   }
}

for(j=0; j<lags; j++){

   cout<<cX[j].merge()<<" ";
   cout<<cY[j].merge()<<" ";
   cout<<cZ[j].merge()<<endl;

   cX[j].setrms(nX); cY[j].setrms(nY); cZ[j].setrms(nZ);
   cX[j].setvar(vx); cY[j].setvar(vy); cZ[j].setvar(vz);

   p[0] = &cX[j];
   p[1] = &cY[j];
   p[2] = &cZ[j];

   cX[j].coincidence(cY[j],5*cWindow);
   cY[j].coincidence(cZ[j],5*cWindow);
   cZ[j].coincidence(cX[j],5*cWindow);
   cout<<cX[j].coincidence(cZ[j],5*cWindow)<<" ";
   cout<<cY[j].coincidence(cX[j],5*cWindow)<<" ";
   cout<<cZ[j].coincidence(cY[j],5*cWindow)<<endl;

   waveburst.output(wave_tree,p);

}
/*
// calculate x-correlation

wavecor xcXY;
wavecor xcZX;
wavecor xcYZ;

wX.Inverse();
wY.Inverse();
wZ.Inverse();

wX.exponential(0.25);
wY.exponential(0.25);
wZ.exponential(0.25);

   for(j=0; j<lags; j++){                // L1 time shifts

      lag = j - lags/2;
      shift = fabs(lag*step);            // time shift due to lag

      cout<<"process wavecor";
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

      xcXY.init(WX,WY,0.05,0.022,8);
      xcYZ.init(WY,WZ,0.05,0.022,8);
      xcZX.init(WZ,WX,0.05,0.022,8);

      xcXY.ifo = 1; xcXY.shift = lag*step; xcXY.run = runID;
      xcZX.ifo = 3; xcZX.shift = lag*step; xcZX.run = runID;
      xcYZ.ifo = 2; xcYZ.shift = lag*step; xcYZ.run = runID;

      cout<<"              occupancy: ";
      printf(" L1H1: %6f",xcXY.select(3.2));
      printf(" H1H2: %6f",xcYZ.select(3.2));
      printf(" H2L1: %6f",xcZX.select(3.2)); cout<<endl;

      cout<<"  coincidence occupancy: ";
      printf(" L1H1: %6f",xcXY.coincidence(0.05,&xcYZ));
      printf(" H1H2: %6f",xcYZ.coincidence(0.05,&xcXY));
      printf(" H2L1: %6f",xcZX.coincidence(0.05,&xcYZ)); cout<<endl;
      cout<<"  coincidence occupancy: ";
      printf(" L1H1: %6f",xcXY.coincidence(0.05,&xcZX));
      printf(" H1H2: %6f",xcYZ.coincidence(0.05,&xcZX));
      printf(" H2L1: %6f",xcZX.coincidence(0.05,&xcXY)); cout<<endl;

      xcor.output(xcor_tree, xcXY);
      xcor.output(xcor_tree, xcZX);
      xcor.output(xcor_tree, xcYZ);
}
*/

wavevar.output(var_tree,&vx,1,waveoffset);
wavevar.output(var_tree,&vy,2,waveoffset);
wavevar.output(var_tree,&vz,3,waveoffset);
noiserms.output(noise_tree,&nX,1,R/2);
noiserms.output(noise_tree,&nY,2,R/2);
noiserms.output(noise_tree,&nZ,3,R/2);


froot->Write();
froot->Close();

}







