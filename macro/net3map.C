{
TF1 *chi = new TF1("chi",ChiSquare,0.,10,4);
chi->SetParameters(4000,2.,5.,0.);
chi->SetParNames ("max","scale","degree"); 

TF1 *logn = new TF1("logn",LogNorm,0.,10,4);
logn->SetParameters(4000,3.7,5.,0.001);
logn->SetParNames ("scale","peak","sigma","as"); 



cout<<"Starting the job"<<endl;
gSystem->Exec("date");
gSystem->Exec("hostname");


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// set parameters
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char* suffix = "net0";
char filtername[128];

int runID         = 0;       // run number

// WB threshold settings

double bpp        = 0.05;    // black pixel probability
double Lo         = 14.5;    // Likelihood threshold
double gap        = 0.1;     // time gap between clusters

// wavelet transform settings

int levelR= 2;               // resampling level
int levelD= 10;              // decomposition level
int l_low = 3;               // low frequency resolution level
int l_high= 6;               // high frequency resolution level
int lpfcut=64;               // low pass filter cut-off [Hz]
double waveoffset = 8.;      // wavelet boundary offset [sec]

// time shift analysis settings

int lags= 101;              // number of time lags including zero lag 
double step=1.;             // time interval between lags [sec]

// logNormal parameters

double pln[18]={7.05,2.26,0.23,   /* level 3 */
		6.19,2.46,0.20,   /* level 4 */
		5.06,2.19,0.35,   /* level 5 */
		4.01,2.08,0.41,   /* level 6 */
		3.00,1.90,0.48,   /* level 7 */
		2.14,1.71,0.53};  /* level 8 */

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Symlet<double>     B(60);   // set wavelet for production
Symlet<double>     S(60,1); // set wavelet for production

wavecluster WC[lags];       // array of cluster structures

wavearray<float> vx,vy,vz;  // noise variability
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

/*
TFile *froot = new TFile(outFile, "RECREATE");          // root file
variability wavevar;
TTree* var_tree = wavevar.setTree();
wavenoise noiserms;
TTree* noise_tree = noiserms.setTree();
netevent netburst(3);
TTree* net_tree = netburst.setTree();
*/

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input data
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

wavearray<double> xw(16384*2); // time series for injections
wavearray<double> yw(16384*2); // time series for injections
wavearray<double> zw(16384*2); // time series for injections
Symlet<double>    SS(16,1);    // set wavelet for production
WSeries<double>   ww(SS);      // wavelet series
xw.rate(1024*16); xw=0.;
yw.rate(1024*16); yw=0.;
zw.rate(1024*16); zw=0.;

yp.rate(1024*16);
yx.rate(1024*16);
yp=0.; yx=0.;

//addWGNoise(yp,5.e-21,0.1);
//addWGNoise(yx,5.e-21,0.1);
addSGBurst(yp, 5.e-22, 250., 0.02);
addCGBurst(yx, 5.e-22, 250., 0.02);
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

sprintf(file,"/home/klimenko/wat/wat-4.0.1/plots/lv_l1.lst");
//readframes(file,"L1:LSC-STRAIN",x);
readframes(file,"L1:STRAIN",x);

xw.add(yp,yp.size(),0,(xw.size()-yp.size())/2-ii);
xw.add(yx,yx.size(),0,(xw.size()-yx.size())/2-ii);
//ww.Forward(xw,6); ww.getLayer(xw,2); ww=0.; 
//ww.putLayer(xw,2); ww.Inverse(); ww.getLayer(xw,0);
x.add(xw,xw.size(),0,(x.size()-xw.size())/2);
yp*=1./An.real();
yx*=1./An.imag();
xw.getStatistics(avr,rms);
cout<<"injected hrss "<<sqrt(rms*rms*xw.size()/xw.rate())<<" "<<xw.rate()<<endl;


double start=x.start();
double rate=x.rate();
double end=start+x.size()/rate;
double duration=end-start-2*waveoffset;

//fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);

//x.add(y,y.size(),0,(x.size()-y.size())/2);


ii = int(yp.rate()*L1.tau.get(L1.tau.indexTheta(Theta),L1.tau.indexPhi(Phi))); 
wB.rate(x.rate());
wB.start(x.start());
wB.resize(x.size());
if(ii>0) wB.cpf(x,x.size()-ii,0,ii);
else     wB.cpf(x,x.size()+ii,ii);


wB.Forward(x,levelR);
wB.getLayer(x,0);
L1.getTFmap()->Forward(x,S,levelD);
nX=L1.getTFmap()->white(60.);
L1.getTFmap()->setlow(64.);
vx=L1.getTFmap()->variability(2.);
L1.getCList()->set(nX);
L1.getCList()->set(vx);

// H1

An=H1.antenna(theta,phi);
yp *= An.real();
yx *= An.imag();
delay = H1.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"H1 delay = "<<delay<<"  "<<ii<<" F+ = "<< An.real()<<" Fx = "<< An.imag()<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.1/plots/lv_h1.lst");
//readframes(file,"H1:LSC-STRAIN",x);
readframes(file,"H1:STRAIN",x);

yw.add(yp,yp.size(),0,(yw.size()-yp.size())/2-ii);
yw.add(yx,yx.size(),0,(yw.size()-yx.size())/2-ii);
//ww.Forward(yw,6); ww.getLayer(yw,2); ww=0.; 
//ww.putLayer(yw,2); ww.Inverse(); ww.getLayer(yw,0);
x.add(yw,yw.size(),0,(x.size()-yw.size())/2);
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
nY=H1.getTFmap()->white(60.);
vy=H1.getTFmap()->variability(2.);
H1.getCList()->set(nY);
H1.getCList()->set(vy);


// V1

An=H2.antenna(theta,phi);
yp *= An.real();
yx *= An.imag();
delay = H2.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"H2 delay = "<<delay<<"  "<<ii<<" F+ = "<< An.real()<<" Fx = "<< An.imag()<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.1/plots/lv_v1.lst");
//readframes(file,"V1:LSC-STRAIN",x);
readframes(file,"V1:noise",x);
//readframes(file,"V1:DFM_A1B2G1",x);

zw.add(yp,yp.size(),0,(zw.size()-yp.size())/2-ii);
zw.add(yx,yx.size(),0,(zw.size()-yx.size())/2-ii);
//ww.Forward(zw,6); ww.getLayer(zw,2); ww=0.; 
//ww.putLayer(zw,2); ww.Inverse(); ww.getLayer(zw,0);
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


ii = int(yp.rate()*H2.tau.get(H2.tau.indexTheta(Theta),H2.tau.indexPhi(Phi))); 
wB.rate(x.rate());
wB.start(x.start());
wB.resize(x.size());
if(ii>0) wB.cpf(x,x.size()-ii,0,ii);
else     wB.cpf(x,x.size()+ii,ii);


wB.Forward(x,levelR);
wB.getLayer(x,0);
H2.getTFmap()->Forward(x,S,levelD);
nZ=H2.getTFmap()->white(60.);
H2.getTFmap()->setlow(64.);
vz=H2.getTFmap()->variability(2.);
H2.getCList()->set(nZ);
H2.getCList()->set(vz);


printf("L1: %f, H1: %f, H2: %f\n",nX.start(),nY.start(),nZ.start());
printf("L1: %f, H1: %f, H2: %f\n",vx.start(),vy.start(),vz.start());

double R = x.rate();              // original data rate
wavecluster* pwc;

wB.resize(1);
 x.resize(1);


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// low pass filtering
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int i,j,k,l;
int low = (int(R+0.5)>>(levelD))/2; // finest frequency resolution

k = lpfcut/low;                     // number of layers to zero
for(i=0; i<k; i++){
   L1.getTFmap()->getLayer(x,i); x = 0.; L1.getTFmap()->putLayer(x,i);  // zero level i
   H1.getTFmap()->getLayer(x,i); x = 0.; H1.getTFmap()->putLayer(x,i);  // zero level i
   H2.getTFmap()->getLayer(x,i); x = 0.; H2.getTFmap()->putLayer(x,i);  // zero level i
}

k = levelD-l_high-1;
if(k > 0)
{
   L1.getTFmap()->Inverse(k); 
   H1.getTFmap()->Inverse(k); 
   H2.getTFmap()->Inverse(k); 
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// level loop
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for(i=l_high; i>=l_low; i--)
{
   L1.getTFmap()->Inverse(1); 
   H1.getTFmap()->Inverse(1); 
   H2.getTFmap()->Inverse(1); 

   sprintf(filtername,"/home/klimenko/wat/wat-4.0.2/data/Sym60_L%1d.dat",i);
   cout<<filtername<<endl;

//   L1.setFilter(32); 

   L1.readFilter(filtername); 
   H1.setFilter(L1); 
   H2.setFilter(L1); 

   NET.setFilter(&H1);
   NET.setDelay(&H1);
   NET.setDelay(&H2);

   cout<<NET.coherence3(bpp,Lo,2,pln+3*(i-l_low))<<"  ";

   k = size_t(2.*gap*L1.getTFmap()->resolution(0));
   if(k<2) k = 2;
   
   cout<<NET.cluster(k)<<endl;
         NET.likelihood3();
         NET.printwc(0);
//
   for(j=0; j<NET.nLag; j++)
   {
//     NET.getwc(j)->cleanhalo();
//     cout<<WC[j].append(*(NET.getwc(j)))<<" ";
      pwc = NET.getwc(j);
      pwc->setrms(nX); pwc->setrms(nY); pwc->setrms(nZ);
      pwc->setvar(vx); pwc->setvar(vy); pwc->setvar(vz);
   }
   cout<<endl;
   
//   WSeries<double> a = L1.TFmap;
//   WSeries<double> a = NET.pixeLHood; a=0;
//   a.cpf(NET.pixeLHood,L1.TFmap.size()-256*256,256*128,256*128);

//   NET.likelihood(Theta,Phi,0);
//   WSeries<double> a = NET.pixeLHood;
//   a = 0.;
//   a.cpf(NET.pixeLHood,NET.pixeLHood.size()-256*256,256*128,256*128);

//   WSeries<double> b = wY;
//   b = 0.;
//   b.cpf(wY,wX.size()-256*256,256*128,256*128);
//   for(int l=0; l<b.size(); l++) {b.data[l] *= b.data[l]*0.5; }

i=l_low-1;
}

cout<<"WaveBurst done\n";

/*

for(j=0; j<lags; j++){
   cout<<WC[j].merge('L')<<" ";
   pwc = NET.getwc(j);
   *pwc = WC[j];
   pwc->setrms(nX); pwc->setrms(nY); pwc->setrms(nZ);
   pwc->setvar(vx); pwc->setvar(vy); pwc->setvar(vz);
}
cout<<endl;

netburst.output(net_tree,&NET);
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








