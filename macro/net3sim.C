{
cout<<"Starting the job"<<endl;
gSystem->Exec("date");
gSystem->Exec("hostname");


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// set parameters
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char* suffix = "net5";

int runID         = 0;      // run number

// WB thresholds

double bpp        = 0.10;   // black pixel probability

// wavelet transform settings

int levelR= 2;              // resampling level
int levelD= 10;             // decomposition level
int l_low = 7;              // low frequency resolution level
int l_high= 2;              // high frequency resolution level
int lpfcut=64;              // low pass filter cut-off [Hz]

// time shift analysis settings

int lags= 101;              // number of time lags including zero lag 
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
//Meyer<double>     S(1); // set wavelet for production

wavecluster cA[lags];       // array of cluster structures

wavearray<float> vx,vy,vz;  // noise variability
WSeries<double>  nX,nY,nZ;  // noise rms

wavearray<double> x;        // temporary time series
wavearray<double> yp(16384); // time series for injections
wavearray<double> yx(16384); // time series for injections

WSeries<double> wB(B);      // original WSeries
WSeries<double> wX(S);      // L1 wavelet series
WSeries<double> wY(S);      // H1 wavelet series
WSeries<double> wZ(S);      // H2 wavelet series

detector        L1("L1");   // detector
detector        H1("H1");   // detector
detector        H2("V1");   // detector
network         NET;        // network

NET.add(&L1); NET.add(&H1); NET.add(&H2); // add detectors to network
NET.setSkyMaps(2.);                       // set network skymaps

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
netevent netburst(3);
TTree* net_tree = netburst.setTree();


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input data
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
wavearray<double> xw(16384*2); // time series for injections
wavearray<double> yw(16384*2); // time series for injections
wavearray<double> zw(16384*2); // time series for injections
Symlet<double>     SS(16,1); // set wavelet for production
WSeries<double> ww(SS);         // wavelet series
xw.rate(1024*16);
yw.rate(1024*16);
zw.rate(1024*16);

yp.rate(1024*16);
yx.rate(1024*16);
yp=0.; yx=0.;

//addWGNoise(yp,2.e-20,0.01);
//addWGNoise(yx,2.e-20,0.01);
addSGBurst(yp, 1.e-21, 280., 0.1);
addCGBurst(yx, 1.e-21, 280., 0.1);
double avr,rms;
yp.getStatistics(avr,rms);
cout<<"plus signal hrss "<<sqrt(rms*rms*yp.size()/yp.rate())<<endl;
yx.getStatistics(avr,rms);
cout<<"cross signal hrss "<<sqrt(rms*rms*yx.size()/yx.rate())<<endl;

d_complex An;
double theta=20.;
double phi = 150.;
double delay;
int iTheta = L1.tau.indexTheta(theta);
int iPhi   = L1.tau.indexPhi(phi);
int ii;

char file[512];

// L1

An=L1.antenna(theta,phi);
yp *= An.real();
yx *= An.imag();
delay = L1.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"L1 delay = "<<delay<<"  "<<ii<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.1/plots/lv_l1.lst");
//readframes(file,"L1:LSC-STRAIN",x);
readframes(file,"L1:STRAIN",x);

xw.add(yp,yp.size(),0,(xw.size()-yp.size())/2-ii);
xw.add(yx,yx.size(),0,(xw.size()-yx.size())/2-ii);
ww.Forward(xw,6); ww.getLayer(xw,2); ww=0.; 
ww.putLayer(xw,2); ww.Inverse(); ww.getLayer(xw,0);
x.add(xw,xw.size(),0,(x.size()-xw.size())/2);
yp*=1./An.real();
yx*=1./An.imag();
xw.getStatistics(avr,rms);
cout<<"injected hrss "<<sqrt(rms*rms*xw.size()/xw.rate())<<endl;


double start=x.start();
double rate=x.rate();
double end=start+x.size()/rate;
double duration=end-start-2*waveoffset;

fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);

//x.add(y,y.size(),0,(x.size()-y.size())/2);

wB.Forward(x,levelR);
wB.getLayer(x,0);
wX.Forward(x,levelD);
nX=wX.white(60.);
wX.setlow(64.);
vx=wX.variability(2.);
L1 = wX;
L1.getCList()->set(nX);
L1.getCList()->set(vx);

//********************
// H1
//********************

An=H1.antenna(theta,phi);
yp *= An.real();
yx *= An.imag();
delay = H1.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"H1 delay = "<<delay<<"  "<<ii<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.1/plots/lv_h1.lst");
//readframes(file,"H1:LSC-STRAIN",x);
readframes(file,"H1:STRAIN",x);

yw.add(yp,yp.size(),0,(yw.size()-yp.size())/2-ii);
yw.add(yx,yx.size(),0,(yw.size()-yx.size())/2-ii);
ww.Forward(yw,6); ww.getLayer(yw,2); ww=0.; 
ww.putLayer(yw,2); ww.Inverse(); ww.getLayer(yw,0);
x.add(yw,yw.size(),0,(x.size()-yw.size())/2);
yp*=1./An.real();
yx*=1./An.imag();
yw.getStatistics(avr,rms);
cout<<"injected hrss "<<sqrt(rms*rms*yw.size()/yw.rate())<<endl;

start=x.start();
rate=x.rate();
end=start+x.size()/rate;
duration=end-start-2*waveoffset;

fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);

wB.Forward(x,levelR);
wB.getLayer(x,0);
wY.Forward(x,levelD);
nY=wY.white(60.);
wY.setlow(64.);
vy=wY.variability(2.);
H1 = wY;
H1.getCList()->set(nY);
H1.getCList()->set(vy);


// V1

An=H2.antenna(theta,phi);
yp *= An.real();
yx *= An.imag();
delay = H2.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"H2 delay = "<<delay<<"  "<<ii<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.1/plots/lv_v1.lst");
//readframes(file,"V1:LSC-STRAIN",x);
readframes(file,"V1:noise",x);
//readframes(file,"V1:DFM_A1B2G1",x);

zw.add(yp,yp.size(),0,(zw.size()-yp.size())/2-ii);
zw.add(yx,yx.size(),0,(zw.size()-yx.size())/2-ii);
ww.Forward(zw,6); ww.getLayer(zw,2); ww=0.; 
ww.putLayer(zw,2); ww.Inverse(); ww.getLayer(zw,0);
x.add(zw,zw.size(),0,(x.size()-zw.size())/2);
yp*=1./An.real();
yx*=1./An.imag();
zw.getStatistics(avr,rms);
cout<<"injected hrss "<<sqrt(rms*rms*zw.size()/zw.rate())<<endl;

start=x.start();
rate=x.rate();
end=start+x.size()/rate;
duration=end-start-2*waveoffset;

fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);

wB.Forward(x,levelR);
wB.getLayer(x,0);
wZ.Forward(x,levelD);
nZ=wZ.white(60.);
wZ.setlow(64.);
vz=wZ.variability(2.);
H2 = wZ;
H2.getCList()->set(nZ);
H2.getCList()->set(vz);

NET.setIndex(&H1);
NET.setAntenna(&L1);
NET.setAntenna(&H1);
NET.setAntenna(&H2);
NET.setDelay(&H1);
NET.setDelay(&H2);
NET.Edge=waveoffset;

printf("L1: %f, H1: %f, H2: %f\n",nX.start(),nY.start(),nZ.start());
printf("L1: %f, H1: %f, H2: %f\n",vx.start(),vy.start(),vz.start());

double R = x.rate();              // original data rate
double f_res = R/2./(1<<levelD);  // frequency resolution

N = int(R*duration+0.5);      // zero lag duration
M = int(R*waveoffset+0.5);    // zero lag offset

wB.resize(1);
 x.resize(1);

//cout<<wX.size()<<" "<<wY.size()<<" "<<wZ.size()<<endl<<endl;



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// low pass filtering
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int i,j,k,l;
int low = (int(R+0.5)>>(levelD))/2; // finest frequency resolution
double scale = 1.;

k = lpfcut/low;                     // number of layers to zero
for(i=0; i<k; i++){
   L1.getTFmap()->getLayer(x,i); x = 0.; L1.getTFmap()->putLayer(x,i);  // zero level i
   H1.getTFmap()->getLayer(x,i); x = 0.; H1.getTFmap()->putLayer(x,i);  // zero level i
   H2.getTFmap()->getLayer(x,i); x = 0.; H2.getTFmap()->putLayer(x,i);  // zero level i
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// level and lag loop
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

slice S0;

for(i=1; i<=l_low; i++){

   L1.getTFmap()->Inverse(1); 
   H1.getTFmap()->Inverse(1); 
   H2.getTFmap()->Inverse(1); 
   S0=L1.getTFmap()->getSlice(0);                    // 0 layer slice

   f_res *= 2.;                          // resolution at this step
   if(i<l_high) continue;                // skip first levels

/*
   WX = *L1.getTFmap(); 
   WY = *H1.getTFmap(); 
   WZ = *H2.getTFmap();

// correct noise RMS in zero layer when get to respective resolution

   if(lpfcut < int(f_res+0.5)) {
     scale = f_res/lpfcut; scale = sqrt(scale/(scale-1.));
     cout<<"correct noise at zero layer: "<<f_res<<"Hz,  scale="<<scale<<endl;
     WX.getLayer(x,0); x*=scale; WX.putLayer(x,0);
     WY.getLayer(x,0); x*=scale; WY.putLayer(x,0);
     WZ.getLayer(x,0); x*=scale; WZ.putLayer(x,0);
     x.resize(1);
   }
   cout<<"  pixel occupancy: "<<bpp<<endl;
   
   printf("%5.2e  ",WX.gSignificance(5.,bpp,0.5));
   printf("%5.2e  ",WY.gSignificance(5.,bpp,0.5));
   printf("%5.2e  ",WZ.gSignificance(5.,bpp,0.5)); cout<<endl;
*/

   L1.readFilter("Sym60_L5.dat"); 
   H1.setFilter(L1); 
   H2.setFilter(L1); 

   cout<<NET.coherence3(0.05,12.,2,pln+3*3)<<endl;
   cout<<NET.cluster(4)<<endl;
         NET.likelihood3();

   for(j=0; j<lags; j++){               
      cout<<"  total list size: ";
      printf(" L1: %5d\n",cA[j].append(NET.getwc(j)));
   }

}

/*

for(j=0; j<lags; j++){

   cout<<cX[j].merge()<<" ";
   cout<<cY[j].merge()<<" ";
   cout<<cZ[j].merge()<<endl;

   cX[j].setrms(nX); cY[j].setrms(nY); cZ[j].setrms(nZ);
   cX[j].setvar(vx); cY[j].setvar(vy); cZ[j].setvar(vz);

   p[0] = L1.getCList(); *p[0] = cX[j];
   p[1] = H1.getCList(); *p[1] = cY[j];
   p[2] = H2.getCList(); *p[2] = cZ[j];

   cout<<NET.cidfill(0.1)<<endl;
   netburst.output(net_tree,&NET,2);

   cX[j].coincidence(cY[j],4*cWindow);
   cY[j].coincidence(cZ[j],4*cWindow);
   cZ[j].coincidence(cX[j],4*cWindow);
   cout<<cX[j].coincidence(cZ[j],4*cWindow)<<" ";
   cout<<cY[j].coincidence(cX[j],4*cWindow)<<" ";
   cout<<cZ[j].coincidence(cY[j],4*cWindow)<<endl;

   waveburst.output(wave_tree,p);

}


cout<<"WaveBurst done\n";

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







