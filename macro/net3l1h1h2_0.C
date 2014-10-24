{
TF1 *chi = new TF1("chi",ChiSquare,0.,10,4);
chi->SetParameters(4000,2.,5.,0.);
chi->SetParNames ("max","scale","degree"); 

TF1 *logn = new TF1("logn",LogNorm,0.,10,4);
logn->SetParameters(4000,5,2.,0.001);
logn->SetParNames ("scale","peak","sigma","as"); 



cout<<"Starting the job"<<endl;
gSystem->Exec("date");
gSystem->Exec("hostname");

int i,j,k,l;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// set parameters
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char* suffix = "net3l1h1h2_h0_s20";
char filtername[128];

int runID         = 0;       // run number

// WB threshold settings

double bpp        = 0.05;    // black pixel probability
double Lo         = 8.0;     // Likelihood threshold
double gap        = 0.05;    // time gap between clusters

// wavelet transform settings

int levelR= 2;               // resampling level
int levelD= 9;               // decomposition level
int l_low = 3;               // low frequency resolution level
int l_high= 7;               // high frequency resolution level
int lpfcut=64;               // low pass filter cut-off [Hz]
double waveoffset = 8.;      // wavelet boundary offset [sec]

// time shift analysis settings

int lags= 101;              // number of time lags including zero lag 
double step=3.25;           // time interval between lags [sec]

// logNormal parameters

double pln[18]={6.86,2.22,0.24,   /* level 3 */
		5.88,2.43,0.25,   /* level 4 */
		4.58,2.10,0.37,   /* level 5 */
		3.74,2.01,0.42,   /* level 6 */
		2.42,1.68,0.53,   /* level 7 */
		2.42,1.76,0.53};  /* level 8 */

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Meyer<double>         B(512);   // set wavelet for production
Meyer<double>         S(512,2); // set wavelet for production

wavecluster WC[lags];       // array of cluster structures

WSeries<float>   vx,vy,vz;  // noise variability
WSeries<double>  nX,nY,nZ;  // noise rms

wavearray<double> yp(16384); // time series for injections
wavearray<double> yx(16384); // time series for injections
wavearray<double> x;         // temporary time series
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
addSGBurst(yp, 0.e-22, 250., 0.02);
addCGBurst(yx, 0.e-22, 250., 0.02);
//yx=0;
double avr,rms;
yp.getStatistics(avr,rms);
cout<<"plus signal hrss "<<sqrt(rms*rms*yp.size()/yp.rate())<<endl;
yx.getStatistics(avr,rms);
cout<<"cross signal hrss "<<sqrt(rms*rms*yx.size()/yx.rate())<<endl;

d_complex An;
double theta=20.;
double phi = 150.;
double Theta=20;
double Phi = 150;
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
cout<<"L1 delay = "<<delay<<"  "<<ii<<" F+ = "<< An.real()<<" Fx = "<< An.imag()<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.3/plots/llo_l1.lst");
readframes(file,"L1:LSC-STRAIN",x);

xw.add(yp,yp.size(),0,(xw.size()-yp.size())/2-ii);
xw.add(yx,yx.size(),0,(xw.size()-yx.size())/2-ii);
//ww.Forward(xw,6); ww.getLayer(xw,2); ww=0.; 
//ww.putLayer(xw,2); ww.Inverse(); ww.getLayer(xw,0);

//   for(i=1; i<171; i++){ 
//      x.add(xw,xw.size(),0,i*4096*11*4);
//   }

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
L1.getTFmap()->lprFilter(4.,8.);
nX=L1.getTFmap()->white(60.,2);
vx=L1.getTFmap()->variability();
L1.getCList()->set(nX);
//L1.getCList()->set(vx);

// H1

An=H1.antenna(theta,phi);
yp *= An.real();
yx *= An.imag();
delay = H1.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"H1 delay = "<<delay<<"  "<<ii<<" F+ = "<< An.real()<<" Fx = "<< An.imag()<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.3/plots/lho_h1.lst");
readframes(file,"H1:LSC-STRAIN",x);

yw.add(yp,yp.size(),0,(yw.size()-yp.size())/2-ii);
yw.add(yx,yx.size(),0,(yw.size()-yx.size())/2-ii);
//ww.Forward(yw,6); ww.getLayer(yw,2); ww=0.; 
//ww.putLayer(yw,2); ww.Inverse(); ww.getLayer(yw,0);

//   for(i=1; i<171; i++){ 
//      x.add(yw,yw.size(),0,i*4096*11*4);
//   }

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
H1.getTFmap()->lprFilter(4.,8.);
nY=H1.getTFmap()->white(60.,2);
vy=H1.getTFmap()->variability();
H1.getCList()->set(nY);
//H1.getCList()->set(vy);


// V1

An=H2.antenna(theta,phi);
yp *= An.real();
yx *= An.imag();
delay = H2.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"H2 delay = "<<delay<<"  "<<ii<<" F+ = "<< An.real()<<" Fx = "<< An.imag()<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.3/plots/lho_h2.lst");
readframes(file,"H2:LSC-STRAIN",x);

zw.add(yp,yp.size(),0,(zw.size()-yp.size())/2-ii);
zw.add(yx,yx.size(),0,(zw.size()-yx.size())/2-ii);
//ww.Forward(zw,6); ww.getLayer(zw,2); ww=0.; 
//ww.putLayer(zw,2); ww.Inverse(); ww.getLayer(zw,0);

//   for(i=1; i<171; i++){ 
//      x.add(zw,zw.size(),0,i*4096*11*4);
//   }

yp*=1./An.real();
yx*=1./An.imag();
zw.getStatistics(avr,rms);
cout<<"injected hrss "<<sqrt(rms*rms*zw.size()/zw.rate())<<endl;

start=x.start();
rate=x.rate();
end=start+x.size()/rate;
duration=end-start-2*waveoffset;

fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);


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
H2.getTFmap()->lprFilter(4.,8.);
nZ=H2.getTFmap()->white(60.,2);
vz=H2.getTFmap()->variability();
H2.getCList()->set(nZ);
//H2.getCList()->set(vz);


printf("L1: %f, H1: %f, H2: %f\n",nX.start(),nY.start(),nZ.start());
printf("L1: %f, H1: %f, H2: %f\n",vx.start(),vy.start(),vz.start());

double R = x.rate();              // original data rate
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

k = levelD-l_high-1;
if(k > 0) NET.Inverse(k);

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// level loop
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for(i=l_high; i>=l_low; i--)
{
   pLN = pln+3*(i-l_low); // pointer to lognormal parameters
   
   NET.Inverse(1); 
   cout<<"max layer "<<L1.getTFmap()->maxLayer()<<endl;

   sprintf(filtername,"/home/klimenko/wat/wat-4.0.3/data/Meyer512_L%1d.dat",i);
   cout<<filtername<<endl;

//   cout<<L1.setFilter(32)<<endl; 

   L1.readFilter(filtername); 
   H1.setFilter(L1); H2.setFilter(L1); 
   NET.setFilter(&H1); NET.setDelay(&H1); NET.setDelay(&H2);

   cout<<"setdelays"<<endl;

   NET.setLogNorm(pLN[0],pLN[1],pLN[2]);
   cout<<NET.coherence3(bpp,Lo,3,2)<<"  ";

   k = size_t(2.*gap*L1.getTFmap()->resolution(0));
   if(k<2) k = 2;
   
   cout<<NET.cluster(k,4)<<endl;
//         NET.likelihood3(0,0);
         NET.printwc(0);
//
   for(j=0; j<NET.nLag; j++)
     cout<<WC[j].append(*(NET.getwc(j)))<<" ";
   
   cout<<endl;
   
//   WSeries<double> a = L1.TFmap;
//   WSeries<double> a = L1.TFmap; a=0;
//   a.cpf(L1.TFmap,L1.TFmap.size()-256*256,256*128,256*128);

//   NET.likelihood(Theta,Phi,0);
//   WSeries<double> a = NET.pixeLHood;
//   a = 0.;
//   a.cpf(NET.pixeLHood,NET.pixeLHood.size()-256*256,256*128,256*128);

//   WSeries<double> b = wY;
//   b = 0.;
//   b.cpf(wY,wX.size()-256*256,256*128,256*128);
//   for(int l=0; l<b.size(); l++) {b.data[l] *= b.data[l]*0.5; }

//   i=l_low-1;
}

cout<<"Search done\n";


wavecluster* pwc;
for(j=0; j<lags; j++){
   cout<<WC[j].supercluster('L')<<" ";
   pwc = NET.getwc(j);
   *pwc = WC[j];
   pwc->setrms(); pwc->setrms(&nX); pwc->setrms(&nY); pwc->setrms(&nZ);
//   pwc->setvar(vx); pwc->setvar(vy); pwc->setvar(vz);
}
cout<<endl;

// finalize likelihood and coordinate reconstruction

NET.likelihood3(0,0,2);
for(i=l_low; i<l_high; i++)  // roll back
{
   NET.Forward(1); 
   sprintf(filtername,"/home/klimenko/wat/wat-4.0.3/data/Meyer512_L%1d.dat",i+1);
   cout<<filtername<<endl;
   L1.readFilter(filtername); H1.setFilter(L1); H2.setFilter(L1); 
   NET.setFilter(&H1); NET.setDelay(&H1); NET.setDelay(&H2);
   NET.likelihood3(0,0,2);
}

size_t nevent=0;
for(j=0; j<lags; j++) nevent+=NET.getwc(j)->supercluster('L',20.);
cout<<"number of events: "<<nevent<<endl;

netburst.output(net_tree,&NET);
wavevar.output(var_tree,&vx,1,waveoffset);
wavevar.output(var_tree,&vy,2,waveoffset);
wavevar.output(var_tree,&vz,3,waveoffset);
noiserms.output(noise_tree,&nX,1,R/2);
noiserms.output(noise_tree,&nY,2,R/2);
noiserms.output(noise_tree,&nZ,3,R/2);
froot->Write();
froot->Close();


}








