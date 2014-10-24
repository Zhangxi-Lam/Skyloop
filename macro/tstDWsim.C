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

int i,j,n,m;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// set parameters
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char* suffix = "tst2d0_Feb21";
char filtername[128];

int runID         = 0;       // run number

// WB threshold settings

double bpp        = 0.0001;  // black pixel probability (for rank statistic)
double Lo         = 2.5;     // Likelihood threshold (in units of noise rms)
double Tgap       = 0.05;    // time gap between clusters
double Fgap       = 256.;    // frequency gap between clusters

// wavelet transform settings

int levelR= 2;               // resampling level
int levelD= 8;               // decomposition level
int l_low = 3;               // low frequency resolution level
int l_high= 8;               // high frequency resolution level
int lpfcut=64;               // low pass filter cut-off [Hz]
int w_mode=1;                // whitening mode
double waveoffset = 8.;      // wavelet boundary offset [sec]

// time shift analysis settings

int lags= 1;                 // number of time lags including zero lag 
double step=3.25;            // time interval between lags [sec]

// logNormal parameters

// sqrt((a1*a1+a2*a2+a3*a3)/3)
double Ao;
double pln[27]={2.700,0.254,0.200,   /* level 1 */
		2.500,0.274,0.200,   /* level 2 */
                2.304,0.298,0.171,   /* level 3 */
		2.144,0.322,0.137,   /* level 4 */
		1.965,0.336,0.145,   /* level 5 */
		1.772,0.356,0.142,   /* level 6 */
		1.587,0.371,0.146,   /* level 7 */
		1.430,0.380,0.156};  /* level 8 */

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Meyer<double>         B(1024);   // set wavelet for production
Meyer<double>         S(1024,2); // set wavelet for production
B.parity(true); 
S.parity(true);

netcluster WC[lags];       // array of cluster structures
netcluster wc;           

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
NET.setbpp(0.1);
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
addSGBurst(yp, 0.e-22, 290., 0.05);
addCGBurst(yx, 0.e-22, 280., 0.02);
//yx=0;
double avr,rms;
yp.getStatistics(avr,rms);
cout<<"plus signal hrss "<<sqrt(rms*rms*yp.size()/yp.rate())<<endl;
yx.getStatistics(avr,rms);
cout<<"cross signal hrss "<<sqrt(rms*rms*yx.size()/yx.rate())<<endl;

wavecomplex An;
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


// H2

An=H2.antenna(theta,phi,psi);
yp *= An.real();
yx *= An.imag();
delay = H2.tau.get(iTheta,iPhi);
ii = int(delay*yp.rate());
cout<<"H2 delay = "<<delay<<"  "<<ii<<" F+ = "<< An.real()<<" Fx = "<< An.imag()<<endl;

sprintf(file,"/home/klimenko/wat/wat-4.0.RH/plots/lho_h2.lst");
readframes(file,"H2:LSC-STRAIN",x);
//sprintf(file,"/home/klimenko/wat/wat-4.0.RH/plots/v1_b1.lst");
//readframes(file,"H2:LSC-STRAIN",x);
//readframes(file,"H2:noise",x);
//ii=0; readframes(file,"H2:DFM_A1B2G1",y); yp.cpf(y,16384,int(89.5*16384)); yp*=3;
//ii=0; readframes(file,"H2:SG235",y); yp.cpf(y,16384,int(89.5*16384)); yp*=0.e-22;

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
n = lpfcut/low;                     // number of layers to zero

for(i=0; i<n; i++){
   L1.getTFmap()->getLayer(x,i); x = 0.; L1.getTFmap()->putLayer(x,i);  // zero level i
   H1.getTFmap()->getLayer(x,i); x = 0.; H1.getTFmap()->putLayer(x,i);  // zero level i
   H2.getTFmap()->getLayer(x,i); x = 0.; H2.getTFmap()->putLayer(x,i);  // zero level i
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// loop over TF resolutions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for(i=levelD; i>=l_low; i--)
{
  gSystem->Exec("date");

  if(i<=l_high) {

    pLN = pln+3*(i-1); // pointer to lognormal parameters
   
    sprintf(filtername,"/home/klimenko/wat/wat-4.2.0/data/Meyer1024_L%1d.dat",i);
    cout<<filtername<<endl;

//   cout<<L1.setFilter(32)<<endl; 
//   L1.writeFilter("/home/klimenko/wat/wat-4.0.RH/data/Meyer1024_L7.dat");

    L1.readFilter(filtername); H1.setFilter(L1); H2.setFilter(L1); 
    NET.setFilter(&H1); NET.setDelay(&H1); NET.setDelay(&H2);
    
    Ao=logNormArg(0.0001,pLN[0],pLN[1],pLN[2]);
    cout<<"pixel threshold in units of noise rms: "<<Ao<<endl;

    gSystem->Exec("date");
    cout<<"core pixels: "<<NET.coherence3(Ao,2.,3.)<<"  "<<endl;

    n = size_t(2.*Tgap*L1.getTFmap()->resolution(0)+0.1);
    m = size_t(Fgap/L1.getTFmap()->resolution(0)+0.1);
    if(n<2) n = 2;
    if(m<2) m = 2;
    
    gSystem->Exec("date");
    cout<<"clusters: "<<NET.cluster(n,m)<<"  "<<endl;
    
    for(j=0; j<NET.nLag; j++) {
      NET.getwc(j)->select("significance",0.1,3,'S');     // detector coincidence
    }

    gSystem->Exec("date");
    cout<<"core pixels: "<<NET.likelihood3('E',true)<<endl;

    for(j=0; j<NET.nLag; j++) {
      NET.getwc(j)->select("significance",0.1,3,'S');     // detector coincidence
      wc = *(NET.getwc(j));
      cout<<wc.csize()<<"|"<<wc.size()<<"|";
      cout<<WC[j].append(wc)<<" "; 
    }
    cout<<endl;
    //    i=l_low-1;
  }

  if(i>l_low) NET.Inverse(1); 

}


netcluster* pwc;
for(j=0; j<lags; j++){
   cout<<WC[j].supercluster('L',0,true)<<"|";
   pwc = NET.getwc(j);
   *pwc = WC[j];
   cout<<pwc->size()<<" ";
}
cout<<endl;
cout<<"Search done\n";
gSystem->Exec("date");

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// finalize likelihood and coordinate reconstruction
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

NET.setRMS();
NET.delink();

cout<<"pixels: "<<NET.likelihood3('L',false,2.0)<<" \n";
NET.setRank(8.);
gSystem->Exec("date");

for(i=l_low; i<l_high; i++)
{
  
   NET.Forward(1);
 
   sprintf(filtername,"/home/klimenko/wat/wat-4.1.0/data/Meyer1024_L%1d.dat",i+1);
   cout<<filtername<<endl;

   L1.readFilter(filtername); H1.setFilter(L1); H2.setFilter(L1); 
   NET.setFilter(&H1); NET.setDelay(&H1); NET.setDelay(&H2);

   cout<<"pixels: "<<NET.likelihood3('L',false,2.0)<<" \n";
   NET.setRank(8.);
   gSystem->Exec("date");
}

size_t nevent=0;
for(j=0; j<lags; j++) {
   pwc = NET.getwc(j);
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

}








