{

TF1 *chi = new TF1("chi",ChiSquare,0.,50,2);
chi->SetParameters(4000,2.);
chi->SetParNames ("max","degree"); 
//chi->SetParNames ("max","scale","degree"); 

TF1 *logn = new TF1("logn",LogNorm,0.,10,4);
logn->SetParameters(4000,2,0.4,0.001);
logn->SetParNames ("scale","peak","sigma","as"); 


cout<<"Starting the job"<<endl;
gSystem->Exec("date");
gSystem->Exec("hostname");

int i,j,k,l;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// set parameters
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char* suffix = "tst06";
char filtername[128];

int runID         = 0;       // run number

// WB threshold settings

double bpp        = 0.10;    // black pixel probability
double Lo         = 15.0;    // Likelihood threshold
double gap        = 0.04;    // time gap between clusters

// wavelet transform settings

int levelR= 2;               // resampling level
int levelD= 8;               // decomposition level
int l_low = 3;               // low frequency resolution level
int l_high= 7;               // high frequency resolution level
int lpfcut=64;               // low pass filter cut-off [Hz]
int w_mode=1;                // whitening mode
double waveoffset = 8.;      // wavelet boundary offset [sec]

// time shift analysis settings

int lags= 5;              // number of time lags including zero lag 
double step=3.;             // time interval between lags [sec]

// logNormal parameters

//double pln[18]={7.28,2.20,0.35,   /* level 3 */
//		6.21,2.36,0.28,   /* level 4 */
//		5.06,2.36,0.31,   /* level 5 */
//		4.01,2.22,0.37,   /* level 6 */
//		3.00,1.99,0.46,   /* level 7 */
//		2.42,1.76,0.53};  /* level 8 */

//double pln[18]={6.86,2.22,0.24,   /* level 3 */
//		5.88,2.43,0.25,   /* level 4 */
//		4.58,2.10,0.37,   /* level 5 */
//		3.74,2.01,0.42,   /* level 6 */
//		2.42,1.68,0.53,   /* level 7 */
//		2.42,1.76,0.53};  /* level 8 */

// sqrt((a1*a1+a2*a2+a3*a3)/3)
double pln[27]={1.903,0.274,0.220,   /* level 1 */
		1.903,0.274,0.220,   /* level 2 */
                2.304,0.298,0.171,   /* level 3 */
		2.144,0.322,0.137,   /* level 4 */
		1.965,0.336,0.145,   /* level 5 */
		1.772,0.356,0.142,   /* level 6 */
		1.587,0.371,0.146,   /* level 7 */
		1.411,0.351,0.173};  /* level 8 */

// sqrt((a2*a2+a3*a3)/2)
//double pln[18]={2.231,0.312,0.215,   /* level 3 */
//		2.089,0.340,0.202,   /* level 4 */
//		1.929,0.364,0.194,   /* level 5 */
//		1.755,0.392,0.187,   /* level 6 */
//		1.581,0.414,0.190,   /* level 7 */
//		1.581,0.414,0.190};  /* level 8 */

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//Symlet<double>         B(60);   // set wavelet for production
//Symlet<double>         S(60,1); // set wavelet for production
Meyer<double>         B(512*2);   // set wavelet for production
Meyer<double>         S(512*2,2); // set wavelet for production
B.parity(false); 
S.parity(false);

wavecluster WC[lags];       // array of cluster structures
wavecluster wc;           

WSeries<float>   vx,vy,vz;  // noise variability
WSeries<double>  nX,nY,nZ;  // noise rms

wavearray<double> yp(16384); // time series for injections
wavearray<double> yx(16384); // time series for injections
wavearray<double> x,y;       // temporary time series
WSeries<double> wB(B);       // original WSeries


detector        L1("L1");   // detector
detector        H1("H1");   // detector
detector        H2("V1");   // detector

network         NET;        // network

NET.add(&L1); NET.add(&H1); NET.add(&H2); // add detectors to network
NET.setSkyMaps(1.);                       // set network skymaps
NET.setTimeShifts(lags,step);
NET.setIndex(&H1);
NET.setAntenna(&L1); 
NET.setAntenna(&H1); 
NET.setAntenna(&H2);
NET.setbpp(bpp);
NET.atleast(3);
NET.Edge = waveoffset;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// initialization of output root file 
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char outFile[512];
sprintf(outFile,"wave_%d%s.root",int(100.*bpp),suffix);
cout<<"output file name: "<<outFile<<endl;


TFile *froot = new TFile(outFile, "RECREATE");          // root file
variability wavevar;
TTree* var_tree = wavevar.setTree();
wavenoise noiserms;
TTree* noise_tree = noiserms.setTree();
netevent netburst(3);
TTree* net_tree = netburst.setTree();


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input data
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

wavearray<double> xw(16384*2); // time series for injections
wavearray<double> yw(16384*2); // time series for injections
wavearray<double> zw(16384*2); // time series for injections
Meyer<double>     SS(32,1);    // set wavelet for production
WSeries<double>   ww(SS);      // wavelet series
xw.rate(1024*16); xw=0.;
yw.rate(1024*16); yw=0.;
zw.rate(1024*16); zw=0.;

yp.rate(1024*16);
yx.rate(1024*16);
yp=0.; yx=0.;

//addWGNoise(yp,5.e-21,0.1);
//addWGNoise(yx,5.e-21,0.1);
addSGBurst(yp, 15.e-22, 290., 0.05);
addCGBurst(yx, 15.e-22, 280., 0.02);
//yx=0;
double avr,rms;
yp.getStatistics(avr,rms);
cout<<"plus signal hrss "<<sqrt(rms*rms*yp.size()/yp.rate())<<endl;
yx.getStatistics(avr,rms);
cout<<"cross signal hrss "<<sqrt(rms*rms*yx.size()/yx.rate())<<endl;

d_complex An;
double theta=20.;
double phi = 150.;
double psi = 0.0*180./PI;
//double psi = 0.39270*180./PI;
double Theta=20.;
double Phi = 150.;
double delay;
int iTheta = L1.tau.indexTheta(theta);
int iPhi   = L1.tau.indexPhi(phi);
int ii=0;

char file[512];

// L1

An=L1.antenna(theta,phi,psi);
yp *= An.real();
yx *= An.imag();
delay = L1.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"L1 delay = "<<delay<<"  "<<ii<<" F+ = "<< An.real()<<" Fx = "<< An.imag()<<endl;


sprintf(file,"/home/klimenko/wat/wat-4.0.RH/plots/llo_l1.lst");
readframes(file,"L1:LSC-STRAIN",x);
//sprintf(file,"/home/klimenko/wat/wat-4.0.RH/plots/lh_b1.lst");
//readframes(file,"L1:STRAIN",x);
//sprintf(file,"/home/klimenko/wat/wat-4.0.RH/plots/lh_ib1.lst");
//ii=0; readframes(file,"L1:DFM_A1B2G1",y); yp.cpf(y,16384,int(89.5*16384)); yp*=3;
//ii=0; readframes(file,"L1:SG235",y); yp.cpf(y,16384,int(89.5*16384)); yp*=0.e-22;

xw.add(yp,yp.size(),0,(xw.size()-yp.size())/2-ii);
xw.add(yx,yx.size(),0,(xw.size()-yx.size())/2-ii);
//ww.Forward(xw,6); ww.getLayer(xw,2); ww=0.; 
//ww.putLayer(xw,2); ww.Inverse(); ww.getLayer(xw,0);

   for(i=1; i<11; i++){ 
      x.add(xw,xw.size(),0,i*4096*30*4);
   }

yp*=1./An.real();
yx*=1./An.imag();
xw.getStatistics(avr,rms);
cout<<"injected hrss "<<sqrt(rms*rms*xw.size()/xw.rate())<<" "<<xw.rate()<<endl;


double start=x.start();
double rate=x.rate();
double end=start+x.size()/rate;
double duration=end-start-2*waveoffset;

ii = int(yp.rate()*L1.tau.get(L1.tau.indexTheta(Theta),L1.tau.indexPhi(Phi))); 
wB.rate(x.rate());
wB.start(x.start());
wB.resize(x.size());
if(ii>0) wB.cpf(x,x.size()-ii,0,ii);
else     wB.cpf(x,x.size()+ii,ii);


wB.Forward(x,levelR); 
wB.getLayer(x,0); 
L1.getTFmap()->Forward(x,S,levelD);
L1.getTFmap()->setlow(64.);
L1.getTFmap()->lprFilter(1,0,120.,4.);
L1.white(60.,w_mode,8.,30.);
vx=L1.getTFmap()->variability();



// H1

An=H1.antenna(theta,phi,psi);
yp *= An.real();
yx *= An.imag();
delay = H1.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"H1 delay = "<<delay<<"  "<<ii<<" F+ = "<< An.real()<<" Fx = "<< An.imag()<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.RH/plots/lho_h1.lst");
readframes(file,"H1:LSC-STRAIN",x);
//sprintf(file,"/home/klimenko/wat/wat-4.0.RH/plots/lh_b1.lst");
//readframes(file,"H1:STRAIN",x);
//sprintf(file,"/home/klimenko/wat/wat-4.0.RH/plots/lh_ib1.lst");
//ii=0; readframes(file,"H1:DFM_A1B2G1",y); yp.cpf(y,16384,int(89.5*16384)); yp*=3;
//ii=0; readframes(file,"H1:SG235",y); yp.cpf(y,16384,int(89.5*16384)); yp*=0.e-22;

yw.add(yp,yp.size(),0,(yw.size()-yp.size())/2-ii);
yw.add(yx,yx.size(),0,(yw.size()-yx.size())/2-ii);
//ww.Forward(yw,6); ww.getLayer(yw,2); ww=0.; 
//ww.putLayer(yw,2); ww.Inverse(); ww.getLayer(yw,0);

   for(i=1; i<11; i++){ 
      x.add(yw,yw.size(),0,i*4096*30*4);
   }

yp*=1./An.real();
yx*=1./An.imag();
yw.getStatistics(avr,rms);
cout<<"injected hrss "<<sqrt(rms*rms*yw.size()/yw.rate())<<endl;

start=x.start();
rate=x.rate();
end=start+x.size()/rate;
duration=end-start-2*waveoffset;

//fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);


ii = int(yp.rate()*H1.tau.get(H1.tau.indexTheta(Theta),H1.tau.indexPhi(Phi))); 
wB.rate(x.rate());
wB.start(x.start());
wB.resize(x.size());
if(ii>0) wB.cpf(x,x.size()-ii,0,ii);
else     wB.cpf(x,x.size()+ii,ii);


wB.Forward(x,levelR);
wB.getLayer(x,0);
H1.getTFmap()->Forward(x,S,levelD);
H1.getTFmap()->setlow(64.);
H1.getTFmap()->lprFilter(1,0,120.,4.);
H1.white(60.,w_mode,8.,30.);
vy=H1.getTFmap()->variability();


// V1

An=H2.antenna(theta,phi,psi);
yp *= An.real();
yx *= An.imag();
delay = H2.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"H2 delay = "<<delay<<"  "<<ii<<" F+ = "<< An.real()<<" Fx = "<< An.imag()<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.RH/plots/lho_h2.lst");
readframes(file,"H2:LSC-STRAIN",x);
//sprintf(file,"/home/klimenko/wat/wat-4.0.RH/plots/v1_b1.lst");
//readframes(file,"V1:LSC-STRAIN",x);
//readframes(file,"V1:noise",x);
//ii=0; readframes(file,"V1:DFM_A1B2G1",y); yp.cpf(y,16384,int(89.5*16384)); yp*=3;
//ii=0; readframes(file,"V1:SG235",y); yp.cpf(y,16384,int(89.5*16384)); yp*=0.e-22;

zw.add(yp,yp.size(),0,(zw.size()-yp.size())/2-ii);
zw.add(yx,yx.size(),0,(zw.size()-yx.size())/2-ii);
//ww.Forward(zw,6); ww.getLayer(zw,2); ww=0.; 
//ww.putLayer(zw,2); ww.Inverse(); ww.getLayer(zw,0);

   for(i=1; i<11; i++){ 
      x.add(zw,zw.size(),0,i*4096*30*4);
   }

yp*=1./An.real();
yx*=1./An.imag();
zw.getStatistics(avr,rms);
cout<<"injected hrss "<<sqrt(rms*rms*zw.size()/zw.rate())<<endl;

start=x.start();
rate=x.rate();
end=start+x.size()/rate;
duration=end-start-2*waveoffset;

//fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);


ii = int(yp.rate()*H2.tau.get(H2.tau.indexTheta(Theta),H2.tau.indexPhi(Phi))); 
wB.rate(x.rate());
wB.start(x.start());
wB.resize(x.size());
if(ii>0) wB.cpf(x,x.size()-ii,0,ii);
else     wB.cpf(x,x.size()+ii,ii);


wB.Forward(x,levelR);
wB.getLayer(x,0);
H2.getTFmap()->Forward(x,S,levelD);
H2.getTFmap()->setlow(64.);
H2.getTFmap()->lprFilter(1.,0,120.,4.);
H2.white(60.,w_mode,8.,30.);
vz=H2.getTFmap()->variability();

printf("L1: %f, H1: %f, H2: %f\n",nX.start(),nY.start(),nZ.start());
printf("L1: %f, H1: %f, H2: %f\n",vx.start(),vy.start(),vz.start());

double R = x.rate();              // original data rate
double scale;
double* pLN;


wB.resize(1);
 x.resize(1);


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// low pass filtering
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int low = (int(R+0.5)>>(levelD))/2; // finest frequency resolution
k = lpfcut/low;                     // number of layers to zero

for(i=0; i<k; i++){
   L1.getTFmap()->getLayer(x,i); x = 0.; L1.getTFmap()->putLayer(x,i);  // zero level i
   H1.getTFmap()->getLayer(x,i); x = 0.; H1.getTFmap()->putLayer(x,i);  // zero level i
   H2.getTFmap()->getLayer(x,i); x = 0.; H2.getTFmap()->putLayer(x,i);  // zero level i
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// loop over TF resolutions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for(i=levelD; i>=l_low; i--)
{
  if(i<=l_high) {

    pLN = pln+3*(i-1); // pointer to lognormal parameters
   
    sprintf(filtername,"/home/klimenko/wat/wat-4.0.5/data/Meyer1024_L%1d.dat",i);
    cout<<filtername<<endl;

//   cout<<L1.setFilter(32)<<endl; 
//   L1.writeFilter("/home/klimenko/wat/wat-4.0.RH/data/Meyer1024_L7.dat");

    L1.readFilter(filtername); H1.setFilter(L1); H2.setFilter(L1); 
    NET.setFilter(&H1); NET.setDelay(&H1); NET.setDelay(&H2);
    
    Lo=logNormArg(0.0005,pLN[0],pLN[1],pLN[2]); 
    Lo=3*Lo*Lo/2.; cout<<"Lo="<<Lo<<endl;
    cout<<"coherence: "<<NET.coherence3(Lo)<<"  "<<endl;

    k = size_t(2.*gap*L1.getTFmap()->resolution(0));
    if(k<2) k = 2;
    
    cout<<"cluster: "<<NET.cluster(k,4)<<endl;
    
    for(j=0; j<NET.nLag; j++) {
      NET.getwc(j)->select("significance",0.1,3,'s');     // detector coincidence
      wc = *(NET.getwc(j));
      cout<<wc.csize()<<"|";
      cout<<WC[j].append(wc)<<" "; 
    }
    cout<<endl;
//    i=l_low-1;
  }

  if(i>l_low) { 
    NET.Inverse(1); 

    if(lpfcut == low) {  // correct noise RMS in zero layer when get to respective resolution
      cout<<"correct SNR for zero layer at "<<low*2<<" Hz"<<endl;
      L1.getTFmap()->getLayer(x,0); x*=sqrt(2); L1.getTFmap()->putLayer(x,0);
      H1.getTFmap()->getLayer(x,0); x*=sqrt(2); H1.getTFmap()->putLayer(x,0);
      H2.getTFmap()->getLayer(x,0); x*=sqrt(2); H2.getTFmap()->putLayer(x,0);
      x.resize(1);
    }

    low *= 2; 
  }

}

wavecluster* pwc;
for(j=0; j<lags; j++){
   cout<<WC[j].supercluster('L',0,false)<<"|";
   pwc = NET.getwc(j);
   *pwc = WC[j];
   cout<<pwc->size()<<" ";
}
cout<<endl;
cout<<"Search done\n";

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// finalize likelihood and coordinate reconstruction
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// roll back - include only core pixels into likelihood calculation

NET.setRMS();
NET.delink();
NET.printwc(0);
NET.likelihood3(10.,0,0,true);
for(i=l_low; i<l_high; i++)
{
   low /= 2;            // resolution to be

   if(lpfcut == low) {  // correct noise RMS in zero layer when get to respective resolution
     cout<<"correct SNR for zero layer at "<<low<<" Hz"<<endl;
     L1.getTFmap()->getLayer(x,0); x*=1/sqrt(2); L1.getTFmap()->putLayer(x,0);
     H1.getTFmap()->getLayer(x,0); x*=1/sqrt(2); H1.getTFmap()->putLayer(x,0);
     H2.getTFmap()->getLayer(x,0); x*=1/sqrt(2); H2.getTFmap()->putLayer(x,0);
     x.resize(1);
   }
  
   NET.Forward(1);
 
   sprintf(filtername,"/home/klimenko/wat/wat-4.0.5/data/Meyer1024_L%1d.dat",i+1);
   cout<<filtername<<endl;
   L1.readFilter(filtername); H1.setFilter(L1); H2.setFilter(L1); 
   NET.setFilter(&H1); NET.setDelay(&H1); NET.setDelay(&H2);
   NET.likelihood3(10.,0,0,true);
}


size_t nevent=0;
for(j=0; j<lags; j++) {
   pwc = NET.getwc(j);
   pwc->select("significance",0.01,3,'S');      // gSF>0.1 in all detectors 
   WC[j] = *pwc;                               // copy selected superclusters
   *pwc  = WC[j];
   nevent += pwc->supercluster('L',20,true);   // reconstruct superclusters
   WC[j] = *pwc;                               // copy selected superclusters
   *pwc  = WC[j];
   pwc->setcore(true);                         // set all pixels to be core
}
cout<<"number of events: "<<nevent<<endl;

// roll forward - include all pixels into likelihood calculation

NET.delink();
NET.printwc(0);
NET.likelihood3(10.);
NET.likelihood3(10.,0,0,true);
//NET.setRank(8.);
for(i=l_high; i>l_low; i--)
{
   NET.Inverse(1);

   if(lpfcut == low) {  // correct noise RMS in zero layer when get to respective resolution
     cout<<"correct SNR for zero layer at "<<low<<" Hz"<<endl;
     L1.getTFmap()->getLayer(x,0); x*=sqrt(2); L1.getTFmap()->putLayer(x,0);
     H1.getTFmap()->getLayer(x,0); x*=sqrt(2); H1.getTFmap()->putLayer(x,0);
     H2.getTFmap()->getLayer(x,0); x*=sqrt(2); H2.getTFmap()->putLayer(x,0);
     x.resize(1);
   }
   low *= 2;            // resolution
 
   sprintf(filtername,"/home/klimenko/wat/wat-4.0.5/data/Meyer1024_L%1d.dat",i-1);
   cout<<filtername<<endl;
   L1.readFilter(filtername); H1.setFilter(L1); H2.setFilter(L1); 
   NET.setFilter(&H1); NET.setDelay(&H1); NET.setDelay(&H2);
   NET.likelihood3(10.);
   NET.likelihood3(10.,0,0,true);
   //   NET.setRank(8.);
}
/*
size_t nevent=0;
for(j=0; j<lags; j++) {
   pwc = NET.getwc(j);
   pwc->select("significance",0.1,3,'s');      // gSF>0.1 for any 2 detectors 
   WC[j] = *pwc;                               // copy selected superclusters
   *pwc  = WC[j];
   nevent += pwc->supercluster('L',20,true);   // reconstruct superclusters
}
cout<<"number of events: "<<nevent<<endl;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// save data in root file
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

netburst.output(net_tree,&NET);
wavevar.output(var_tree,&vx,1,waveoffset);
wavevar.output(var_tree,&vy,2,waveoffset);
wavevar.output(var_tree,&vz,3,waveoffset);
noiserms.output(noise_tree,&L1.nRMS,1,R/2);
noiserms.output(noise_tree,&H1.nRMS,2,R/2);
noiserms.output(noise_tree,&H2.nRMS,3,R/2);
froot->Write();
froot->Close();
*/
}








