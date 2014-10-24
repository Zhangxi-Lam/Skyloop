{
int jid;
cin>>jid;
cout<<"Starting the job"<<endl;
gSystem->Exec("date");
gSystem->Exec("hostname");

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// set parameters
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char* suffix = "a5";

double bpp        = 0.10;   // black pixel probability
double cWindow    = 0.025;  // councidence window [sec]
double cThreshold = 1.0;    // coincidence threshold
double onePixelT  = 3.0;    // threshold on one-pixel clusters
bool halo         = true;   // processing of halo pixels

double start;
double end;
double duration;            // duration [sec]
double rate;
double waveoffset = 4.;     // wavelet boundary offset [sec]
double skip  = 0.;          // skip duration

int first=2;                // first level
int last =7;                // last  level
int levelR=2;               // resampling level
int levelD=9;               // decomposition level
int lpfcut=64;              // low pass filter cut-off [Hz]

int lags=1;                 // number of time lags including zero lag 
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

wavearray<float> ax,ay,az;  // cluster parameters
wavearray<float> vx,vy,vz;  // noise variability
WSeries<double>  nX,nY,nZ;  // noise rms

wavearray<double> x,y;      // temporary time series
WSeries<double> wB(B);      // original WSeries
WSeries<double> wX(S);      // L1 wavelet series
WSeries<double> wY(S);      // H1 wavelet series
WSeries<double> wZ(S);      // H2 wavelet series
WSeries<double> WX,WY,WZ;   // rank amplitudes



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input data
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char file[512];

// L1

sprintf(file,"INPUT/llo.lst.%d",jid);
readframes(file,"L1:LSC-AS_Q",x);

start=x.start();
rate=x.rate();
end=start+x.size()/rate;
duration=end-start-2*waveoffset;

fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);

sprintf(file,"INPUT/mdc.lst.%d",jid);
readframes(file,"L1:GW",y);
x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
//x.add(y,y.size(),0,int((y.start()-x.start())*x.rate()));
y.resize(1);

cout<<"After L1 input"<<endl;
gSystem->Exec("date");

//x.resize(int(duration+2*waveoffset+0.1)*1024*16);
//x=0.; x.rate(1024*16);
//AddGauss(x,1.,0);

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wX.Forward(x,levelD);
nX=wX.white(60.);
vx=wX.variability(1.);

// H1
sprintf(file,"INPUT/lho.lst.%d",jid);
readframes(file,"H1:LSC-AS_Q",x);

if(start!=x.start() || rate!=x.rate())
{
  fprintf(stderr."H1 and L1 are inconsistent: start L1=%f, H1 start=%f, L1 rate=%f, H1 rate=%f\n",
	  start,x.start(),rate,x.rate());
  exit(1);
}

sprintf(file,"INPUT/mdc.lst.%d",jid);
readframes(file,"H1:GW",y);
x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
//x.add(y,y.size(),0,int((y.start()-x.start())*x.rate()));
y.resize(1);

cout<<"After H1 input"<<endl;
gSystem->Exec("date");

//x.resize(int(duration+2*waveoffset+0.1)*1024*16);
//x=0.; x.rate(1024*16);
//AddGauss(x,1.,0);

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wY.Forward(x,levelD);
nY=wY.white(60.);
vy=wY.variability(1.);

// H2
sprintf(file,"INPUT/lho.lst.%d",jid);
readframes(file,"H2:LSC-AS_Q",x);

if(start!=x.start() || rate!=x.rate())
{
  fprintf(stderr."H2 and L1 are inconsistent: start L1=%f, H2 start=%f, L1 rate=%f, H2 rate=%f\n",
	  start,x.start(),rate,x.rate());
  exit(1);
}
sprintf(file,"INPUT/mdc.lst.%d",jid);
readframes(file,"H2:GW",y);
x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
//x.add(y,y.size(),0,int((y.start()-x.start())*x.rate()));
y.resize(1);

cout<<"After H2 input"<<endl;
gSystem->Exec("date");

//x.resize(int(duration+2*waveoffset+0.1)*1024*16);
//x=0.; x.rate(1024*16);
//AddGauss(x,1.,0);

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wZ.Forward(x,levelD);
nZ=wZ.white(60.);
vz=wZ.variability(1.);

printf("L1: %f, H1: %f, H2: %f\n",nX.start(),nY.start(),nZ.start());
printf("L1: %f, H1: %f, H2: %f\n",vx.start(),vy.start(),vz.start());

double R = x.rate();         // original data rate
double f_res = R/2./(1<<levelD);  // frequency resolution

N = int(R*duration+0.5);      // zero lag duration
M = int(R*waveoffset+0.5);    // zero lag offset

wB.resize(1);
 x.resize(1);

cout<<wX.size()<<" "<<wY.size()<<" "<<wZ.size()<<endl<<endl;




//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// initialization of output root file 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


char outFile[512];
sprintf(outFile,"/usr1/igor/waveburst/wave_%d_%d_%d_%d%s.root",int(start),int(duration),int(100.*bpp),int(10.*cThreshold),suffix);
cout<<"output file name: "<<outFile<<endl;
TFile *froot = new TFile(outFile, "RECREATE");          // root file
wbsingle waveburst;
TTree* wave_tree = waveburst.setTree();
variability wavevar;
TTree* var_tree = wavevar.setTree();
wavenoise noiserms;
TTree* noise_tree = noiserms.setTree();


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// low pass filtering
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int i,j,k,l;
int low = (int(R+0.5)>>(levelD))/2; // finest frequency resolution
int inv = 0;

k = lpfcut/low;
for(i=0; i<k; i++){
   wX.getLayer(x,i); x = 0.; wX.putLayer(x,i);  // zero level i
   wY.getLayer(x,i); x = 0.; wY.putLayer(x,i);  // zero level i
   wZ.getLayer(x,i); x = 0.; wZ.putLayer(x,i);  // zero level i
}

while(k>(1<<++inv));       
if(inv < first) {            // correct noise rms
   f_res *= double(1<<(inv+1));
   wX.Inverse(inv); wX.getLayer(x,1); x*=1.41; wX.putLayer(x,1);
   wY.Inverse(inv); wY.getLayer(x,1); x*=1.41; wY.putLayer(x,1);
   wZ.Inverse(inv); wZ.getLayer(x,1); x*=1.41; wZ.putLayer(x,1);
   i = inv+1;                // level index where start the analysis
   inv = first-inv;          // next number of inversion steps
}
else { i = first; inv = i; } // set parameters for next loop

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// level and lag loop
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

slice S0;

for(; i<=last; i++){
   // correct first layer when get to respective resolution
   if(lpfcut == int(f_res+0.5)) {
      cout<<"second: inv="<<inv<<"  i="<<i<<endl;
      wX.getLayer(x,1); x*=1.41; wX.putLayer(x,1);
      wY.getLayer(x,1); x*=1.41; wY.putLayer(x,1);
      wZ.getLayer(x,1); x*=1.41; wZ.putLayer(x,1);
      x.resize(1);
   }

   wX.Inverse(inv); 
   wY.Inverse(inv); 
   wZ.Inverse(inv); 
   S0=wX.getSlice(0);                    // 0 layer slice

   for(j=0; j<lags; j++){                // L1 time shifts

      lag = j - lags/2;
      shift = fabs(lag*step);            // time shift due to lag

      cout<<"process: level="<<i;
      cout<<",  lag="<<lag<<",  shift="<<lag*step<<endl;

      k = int(R*shift+0.5);              // offset due to lag
      n = N-k;
      m = lag>=0 ? M : M+k;
      if((m/S0.stride())*S0.stride() != m) {
	 cout<<"wb3: time shift error";
	 cout<<" stride="<<S0.stride()<<"  offset="<<m<<endl;
      }

      WX=wX[slice(m,n,1)]; 

      m = lag>=0 ? M+k : M;
      if((m/S0.stride())*S0.stride() != m) cout<<"wb3: time shift error\n";

      WY=wY[slice(m,n,1)]; 
      WZ=wZ[slice(m,n,1)]; 


//   WX.percentile(bpp,2);
//   WY.percentile(bpp,2);
//   WZ.percentile(bpp,2);


      WX.fraction(60.,bpp,1);
      WY.fraction(60.,bpp,1);
      WZ.fraction(60.,bpp,1);

      WX.significance((1<<i)*32);
      WY.significance((1<<i)*32);
      WZ.significance((1<<i)*32);

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
      CX.ifo = 1; CX.shift = lag*step;
      CX.apush(wX,offset); 
      CX.cluster(); CX.cleanhalo();

      offset = lag>=0 ? waveoffset+shift : waveoffset;  

      CY.init(WY,halo); 
      CY.ifo = 2; CY.shift = lag*step;
      CY.apush(wY,offset);
      CY.cluster(); CY.cleanhalo();

      CZ.init(WZ,halo); 
      CZ.ifo = 3; CZ.shift = lag*step;
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
   if(inv>1) inv=1;     
}


for(j=0; j<lags; j++){
   cout<<cX[j].merge()<<endl;
   cout<<cY[j].merge()<<endl;
   cout<<cZ[j].merge()<<endl;

   cX[j].coincidence(cY[j],2*cWindow);
   cY[j].coincidence(cZ[j],2*cWindow);
   cZ[j].coincidence(cX[j],2*cWindow);
   cout<<cX[j].coincidence(cZ[j],2*cWindow)<<" ";
   cout<<cY[j].coincidence(cX[j],2*cWindow)<<" ";
   cout<<cZ[j].coincidence(cY[j],2*cWindow)<<endl;

   printf("L1: start=%f, stop=%f\n",cX[j].start,cX[j].stop);
   printf("H1: start=%f, stop=%f\n",cY[j].start,cY[j].stop);
   printf("H2: start=%f, stop=%f\n",cZ[j].start,cZ[j].stop);

   p[0] = &cX[j];
   p[1] = &cY[j];
   p[2] = &cZ[j];

   waveburst.output(wave_tree,p);
}

wavevar.output(var_tree,&vx,1,waveoffset);
wavevar.output(var_tree,&vy,2,waveoffset);
wavevar.output(var_tree,&vz,3,waveoffset);
noiserms.output(noise_tree,&nX,1,R/2);
noiserms.output(noise_tree,&nY,2,R/2);
noiserms.output(noise_tree,&nZ,3,R/2);
froot->Write();
froot->Close();

cout<<"Stopping the job"<<endl;
gSystem->Exec("date");

return 0;
}







