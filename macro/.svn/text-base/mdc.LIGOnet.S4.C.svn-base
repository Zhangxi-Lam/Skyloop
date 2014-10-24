{
int jid;
char input_dir[500],output_dir[500], input_label[500], output_label[500];
cin>>jid>>input_dir>>input_label>>output_dir>>output_label;
cout<<"Starting the job "<<jid<<" "<<input_dir<<" "<<" "<<input_label<<" "<<output_dir<<" "<<output_label<<endl;
gSystem->Exec("date");
gSystem->Exec("hostname");
double factor=1.0;

int nfactor=21;

double factors[]={0.0133, 0.0188, 0.0266, 0.0376, 0.0531, 0.0750, 0.106, 0.150, 0.211, 0.298,
                  0.422, 0.596, 0.841, 1.19, 1.68, 2.37, 3.35, 4.73, 6.68, 9.44, 13.3};



for(int iii=0;iii<nfactor;iii++)
{
  factor=factors[iii];
  cout<<"factor="<<factor<<endl;


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// set parameters
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


int runID         = 0;      // run number

// WB thresholds

double bpp        = 0.10;   // black pixel probability
double cWindow    = 0.032;  // councidence window [sec]
double cThreshold = 1.0;    // coincidence threshold
double onePixelT  = 0.0;    // threshold on one-pixel clusters
bool halo         = true;   // processing of halo pixels

// input data settings

double start, end, rate;
double duration;            // duration [sec]
double waveoffset = 8.;     // wavelet boundary offset [sec]
double skip  = 0.;          // skip duration

// wavelet transform settings

int levelR= 2;              // resampling level
int levelD= 10;              // decomposition level
int l_low = 7;              // low frequency resolution level
int l_high= 2;              // high frequency resolution level
int lpfcut=64;              // low pass filter cut-off [Hz]

// time shift analysis settings

int lags= 1;                // number of time lags including zero lag 
double step=3.125;           // time interval between lags [sec]


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

wavecluster cX[lags];       // array of cluster structures for L1
wavecluster cY[lags];       // array of cluster structures for H1 
wavecluster cZ[lags];       // array of cluster structures for H2
wavecluster CX,CY,CZ;
wavecluster* p[3];

wavearray<float> vx,vy,vz;  // noise variability
WSeries<double>  nX,nY,nZ;  // noise rms
WSeries<double>  calX, calY, calZ; //calibration coefficients

WSeries<double>* pL1;
WSeries<double>* pH1;
WSeries<double>* pH2;

wavearray<double> x,y;      // temporary time series
WSeries<double> wB(B);      // original WSeries
WSeries<double> wX(S);      // L1 wavelet series
WSeries<double> wY(S);      // H1 wavelet series
WSeries<double> wZ(S);      // H2 wavelet series
WSeries<double> WX,WY,WZ;   // rank amplitudes
detector        L1("L1");   // detector
detector        H1("H1");   // detector
detector        H2("H2");   // detector
network         NET;        // network

NET.add(&L1); NET.add(&H1); NET.add(&H2); // add detectors to network
NET.setSkyMaps(4.);                       // set network skymaps



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input data
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char file[512];

// L1

 sprintf(file,"%s/llo1.lst.%d",input_dir,jid);
 printf("file=%s\n",file);
 readframes(file,"L1:LSC-STRAIN",x);

start=x.start();
rate=x.rate();
end=start+x.size()/rate;
duration=end-start-2*waveoffset;

fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);

sprintf(file,"%s/%s.lst.%d",input_dir,input_label,jid);
readframes(file,"L1:GW-H",y);
y*=factor;
x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
y.resize(1);

cout<<"After L1 input"<<endl;
gSystem->Exec("date");

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wX.Forward(x,levelD);
nX=wX.white(60.);
wX.setlow(double(lpfcut));
vx=wX.variability(2.);

// H1

 sprintf(file,"%s/lho1.lst.%d",input_dir,jid);
 readframes(file,"H1:LSC-STRAIN",x);

if(start!=x.start() || rate!=x.rate())
{
  fprintf(stderr,"H1 and L1 are inconsistent: start L1=%f, H1 start=%f, L1 rate=%f, H1 rate=%f\n",
	  start,x.start(),rate,x.rate());
  exit(1);
}

sprintf(file,"%s/%s.lst.%d",input_dir,input_label,jid);
readframes(file,"H1:GW-H",y);
y*=factor;
x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
y.resize(1);

cout<<"After H1 input"<<endl;
gSystem->Exec("date");

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wY.Forward(x,levelD);
nY=wY.white(60.);
wY.setlow(double(lpfcut));
vy=wY.variability(2.);

 cout<<"wX.rate()="<<wX.rate()<<endl;

// H2

 sprintf(file,"%s/lho2.lst.%d",input_dir,jid);
 readframes(file,"H2:LSC-STRAIN",x);

if(start!=x.start() || rate!=x.rate())
{
  fprintf(stderr,"H2 and L1 are inconsistent: start L1=%f, H2 start=%f, L1 rate=%f, H2 rate=%f\n",
	  start,x.start(),rate,x.rate());
  exit(1);
}

sprintf(file,"%s/%s.lst.%d",input_dir,input_label,jid);
readframes(file,"H2:GW-H",y);
y*=factor;
x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
y.resize(1);

cout<<"After H2 input"<<endl;
gSystem->Exec("date");

cout<<x.rate()<<"  "<<file<<endl;
wB.Forward(x,levelR);
wB.getLayer(x,0);
wZ.Forward(x,levelD);
nZ=wZ.white(60.);
wZ.setlow(double(lpfcut));
vz=wZ.variability(2.);


printf("L1: %f, H1: %f, H2: %f\n",nX.start(),nY.start(),nZ.start());
printf("L1: %f, H1: %f, H2: %f\n",vx.start(),vy.start(),vz.start());

double R = x.rate();              // original data rate
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
char outFileTmp[512];
sprintf(outFile,"/usr1/igor/waveburst/wave_%d_%d_%s_%g_id%d_.root",int(start+waveoffset),int(duration),output_label,factor,jid);
sprintf(outFileTmp,"/usr1/igor/waveburst/wave_%d_%d_%s_%g_id%d_.root.tmp",int(start+waveoffset),int(duration),output_label,factor,jid);
cout<<"output file name: "<<outFile<<endl;
cout<<"output file tmp: "<<outFileTmp<<endl;


TFile *froot = new TFile(outFileTmp, "RECREATE");          // root file
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

      NET.coincidence(cWindow,cThreshold);

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

   waveburst.output(wave_tree,p,3);

}



wavevar.output(var_tree,&vx,1,waveoffset);
wavevar.output(var_tree,&vy,2,waveoffset);
wavevar.output(var_tree,&vz,3,waveoffset);
noiserms.output(noise_tree,&nX,1,R/2);
noiserms.output(noise_tree,&nY,2,R/2);
noiserms.output(noise_tree,&nZ,3,R/2);


froot->Write();
froot->Close();

char command[512];
sprintf(command,"mv %s %s", outFileTmp, outFile);
gSystem->Exec(command);
sprintf(command,"cp %s %s",outFile,output_dir);
gSystem->Exec(command);

}

cout<<"Stopping the job "<<jid<<endl;
gSystem->Exec("date");

return 0;

}
