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
double cWindow    = 0.064;  // councidence window [sec]
double cThreshold = 1.0;    // coincidence threshold
double onePixelT  = 0.0;    // threshold on one-pixel clusters
bool halo         = true;   // processing of halo pixels

// input data settings

double start = 0;           // start time of 4000 sec
double duration = 60*10;    // duration [sec]
double waveoffset = 8.;     // wavelet boundary offset [sec]
double skip  = 0.;          // skip duration

// wavelet transform settings

int levelR= 2;              // resampling level
int levelD= 10;             // decomposition level
int l_low = 7;              // low frequency resolution level
int l_high= 2;              // high frequency resolution level
int lpfcut=64;              // low pass filter cut-off [Hz]

// time shift analysis settings

int lags= 101;              // number of time lags including zero lag 
double step=3.25;           // time interval between lags [sec]

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

double offset;
double shift;               // time shift between series
int N,n;                    // number of samples in duration seconds
int M,m;                    // number of samples in offset seconds
int lag;

//Biorthogonal<double> B(30); // set wavelet for decimation
Meyer<double>     B(60);   // set wavelet for production
Meyer<double>     S(60,1); // set wavelet for production

wavecluster cX[lags];       // array of cluster structures for L1
wavecluster cY[lags];       // array of cluster structures for H1 
wavecluster cZ[lags];       // array of cluster structures for H2
wavecluster CX,CY,CZ;
wavecluster* p[3];

wavearray<float> vx,vy,vz;  // noise variability
WSeries<double>  nX,nY,nZ;  // noise rms
WSeries<double>* pL1;
WSeries<double>* pH1;
WSeries<double>* pH2;

wavearray<double> x;        // temporary time series
wavearray<double> y(16384); // time series for injections
WSeries<double> wB(B);      // original WSeries
WSeries<double> wX(S);      // L1 wavelet series
WSeries<double> wY(S);      // H1 wavelet series
WSeries<double> wZ(S);      // H2 wavelet series
WSeries<double> WX,WY,WZ;   // significance
detector        L1("L1");   // detector
detector        H1("H1");   // detector
detector        H2("H2");   // detector
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

y.rate(1024*16);
y=0.;
//addWGNoise(y,200.5,0.1);
addSGBurst(y, 0.0, 300., 0.1);
double avr,rms;
y.getStatistics(avr,rms);
cout<<"signal hrss "<<sqrt(rms*rms*y.size()/y.rate())<<endl;

char file[512];

// L1

sprintf(file,"/home/klimenko/wat/wat-4.0.1/plots/h2.lst");
//readframes(file,"L1:LSC-STRAIN",x);
readframes(file,"L1:STRAIN",x);

start=x.start();
double rate=x.rate();
double end=start+x.size()/rate;
duration=end-start-2*waveoffset;

fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);

//x.add(y,y.size(),0,(x.size()-y.size())/2);

wB.Forward(x,levelR);
wB.getLayer(x,0);
wX.Forward(x,levelD);
nX=wX.white(60.);
wX.setlow(64.);
vx=wX.variability(2.);

// H1

sprintf(file,"/home/klimenko/wat/wat-4.0.1/plots/h2.lst");
//readframes(file,"H1:LSC-STRAIN",x);
readframes(file,"H1:STRAIN",x);

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


// H2

sprintf(file,"/home/klimenko/wat/wat-4.0.1/plots/h2.lst");
readframes(file,"H2:STRAIN",x);
//readframes(file,"V1:noise",x);

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
   wX.getLayer(x,i); x = 0.; wX.putLayer(x,i);  // zero level i
   wY.getLayer(x,i); x = 0.; wY.putLayer(x,i);  // zero level i
   wZ.getLayer(x,i); x = 0.; wZ.putLayer(x,i);  // zero level i
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// level and lag loop
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

slice S0;

for(i=1; i<=l_low; i++){

   wX.Inverse(1); 
   wY.Inverse(1); 
   wZ.Inverse(1); 
   S0=wX.getSlice(0);                    // 0 layer slice

   f_res *= 2.;                          // resolution at this step
   if(i<l_high) continue;                // skip first levels

   WX = wX; WY = wY; WZ = wZ;

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

      pL1 = L1.getwseries();
      L1 = WX[slice(m,n,1)]; 

      m = lag>=0 ? M+k : M;               // lag>0 shift H1,h2 (second channel)
      if((m/S0.stride())*S0.stride() != m) cout<<"wb3: time shift error\n";

      pH1 = H1.getwseries();
      H1 = WY[slice(m,n,1)]; 
      pH2 = H2.getwseries();
      H2 = WZ[slice(m,n,1)]; 

//   cout<<"  pixel occupancy: "<<bpp<<endl;
//   printf("%5.2e  ",pL1->rSignificance(5.,bpp,0.5));
//   printf("%5.2e  ",pH1->rSignificance(5.,bpp,0.5));
//   printf("%5.2e  ",pH2->rSignificance(5.,bpp,0.5)); cout<<endl;

      NET.coincidence(cWindow,cThreshold);
      
//      cout<<"  pixel occupancy: ";
//      printf("%5.2e  ",pL1->Coincidence(*pH1,cWindow,cThreshold));
//      printf("%5.2e  ",pH1->Coincidence(*pL1,cWindow,cThreshold));
//      printf("%5.2e  ",pH2->Coincidence(*pH1,cWindow,cThreshold)); cout<<endl;
//      cout<<"  pixel occupancy: ";
//      printf("%5.2e  ",pL1->Coincidence(*pH2,cWindow,cThreshold));
//      printf("%5.2e  ",pH1->Coincidence(*pH2,cWindow,cThreshold));
//      printf("%5.2e  ",pH2->Coincidence(*pL1,cWindow,cThreshold)); cout<<endl;
//      cout<<"  pixel occupancy: ";
//      printf("%5.2e  ",pL1->Coincidence(*pH1,cWindow,cThreshold));
//      printf("%5.2e  ",pH1->Coincidence(*pL1,cWindow,cThreshold));
//      printf("%5.2e  ",pH2->Coincidence(*pH1,cWindow,cThreshold)); cout<<endl;
      

      offset = lag>=0 ? waveoffset : waveoffset+shift;  

      pws = L1.getwseries();
      CX.init(*pws,halo);
      CX.ifo = 1; CX.shift = lag*step; CX.run = runID;
      CX.apush(wX,offset); 
      CX.cluster(); CX.cleanhalo();

      offset = lag>=0 ? waveoffset+shift : waveoffset;  

      pws = H1.getwseries();
      CY.init(*pws,halo); 
      CY.ifo = 2; CY.shift = lag*step; CY.run = runID;
      CY.apush(wY,offset);
      CY.cluster(); CY.cleanhalo();

      pws = H2.getwseries();
      CZ.init(*pws,halo); 
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

   p[0] = L1.getwavecluster(); *p[0] = cX[j];
   p[1] = H1.getwavecluster(); *p[1] = cY[j];
   p[2] = H2.getwavecluster(); *p[2] = cZ[j];

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
}







