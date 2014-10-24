{
int jid;
 int i,j,m;
 char input_dir[500],output_dir[500], input_label[500], output_label[500], filtername[500];
 double *pLN;
cin>>jid>>input_dir>>input_label>>output_dir>>output_label;
cout<<"Starting the job "<<jid<<endl;
cout<<"Input directory  "<<input_dir<<",  label: "<<input_label<<endl;
cout<<"Output directory "<<output_dir<<",  label: "<<output_label<<endl;
gSystem->Exec("date");
gSystem->Exec("hostname");

long f_id;long f_size;long f_flags;long f_modtime;

double factor=1.0;


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// declarations
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Meyer<double> B(1024);       // set wavelet for production
Meyer<double> S(1024,2);     // set wavelet for production

netcluster WC[lags];         // array of cluster structures
netcluster wc;           
netcluster* pwc;           

WSeries<float>   vx,vy,vz;   // noise variability
WSeries<double>  nX,nY,nZ;   // noise rms

 wavearray<double> x,y;         // temporary time series
WSeries<double> wB(B);       // original WSeries

detector        L1("L1");    // detector
detector        H1("H1");    // detector
detector        H2("H2");    // detector

network         NET;         // network
NET.setRunID(jid);
NET.readMDClog("BurstMDC-SG20_S4-Log.txt");
NET.readSEGlist("segment.lst",2);
NET.add(&L1); 
NET.add(&H1); 
NET.add(&H2);
NET.setSkyMaps(1.);          // set network skymaps
NET.setTimeShifts(lags,step);
NET.setIndex(&H1);
NET.setAntenna(&L1); 
NET.setAntenna(&H1); 
NET.setAntenna(&H2);
NET.setbpp(0.1);
NET.Edge = waveoffset;


for(int iii=0;iii<nfactor;iii++)
{
  factor=factors[iii];
  cout<<"factor="<<factor<<endl;
  for(int j=0; j<NET.nLag; j++) WC[j].clear();
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input data
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

char file[512];

// L1

sprintf(file,"%s/llo.lst.%d",input_dir,jid);
readframes(file,"L1:LSC-STRAIN",x);
printf("file=%s\n",file);

start    = x.start(); 
rate     = x.rate();
end      = start+x.size()/rate;
duration = end-start-2*waveoffset;

fprintf(stdout,"start=%f end=%f duration=%f rate=%f\n",start,end,duration,rate);

 sprintf(file,"%s/%s.lst.%d",input_dir,input_label,jid);
 readframes(file,"L1:GW-H",y);
 y*=factor;
 x[slice(int((y.start()-x.start())*x.rate()),y.size(),1)]+=y;
 y.resize(1);

wB.Forward(x,levelR); 
wB.getLayer(x,0); 
L1.getTFmap()->Forward(x,S,levelD);

// ----- L1 calibration start ------
/*
 readCalibration("L1",refname_L1,facname_L1, refstart_L1, reflength_L1, start, end-start, 
		 nrc_L1, fstep_L1, &R_L1, &C_L1, alpha_L1, gamma_L1);
 cout<<"Before calibrate"<<endl;

 cout<<"nrc="<<nrc_L1<<" fstep="<<fstep_L1<<" R[1]=("<<R_L1[1]->real()<<","<<
   R_L1[1]->imag()<<") C[1]=("<<C_L1[1]->real()<<","<<C_L1[1]->imag()<<") alpha[1]="<<alpha_L1.data[1]<<
   " gamma[1]="<<gamma_L1.data[1]<<" wX.rate="<<wX.rate()<<endl;

 printf("alpha.start=%f alpha.rate=%f gamma.start=%f gamma.rate=%f\n",
	alpha_L1.start(),alpha_L1.rate(),gamma_L1.start(),gamma_L1.rate());

 calX=wX.calibrate(nrc_L1,fstep_L1,R_L1,C_L1,alpha_L1,gamma_L1,1);

 delete [] R_L1;
 delete [] C_L1;
 alpha_L1.resize(1);
 gamma_L1.resize(1);
*/
// ----- L1 calibration end ------


L1.getTFmap()->setlow(64.);
L1.getTFmap()->lprFilter(1,0,120.,4.);
L1.white(60.,w_mode,8.,30.);
vx=L1.getTFmap()->variability();

cout<<"After L1 data conditioning"<<endl; gSystem->Exec("date");


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

wB.Forward(x,levelR);
wB.getLayer(x,0);
H1.getTFmap()->Forward(x,S,levelD);


// ----- H1 calibration start ------
/*
 readCalibration("H1",refname_H1,facname_H1, refstart_H1, reflength_H1, start, 
		 end-start,nrc_H1, fstep_H1, &R_H1, &C_H1, alpha_H1, gamma_H1);
 calY=wY.calibrate(nrc_H1,fstep_H1,R_H1,C_H1,alpha_H1,gamma_H1,1);
 delete [] R_H1;
 delete [] C_H1;
 alpha_H1.resize(1);
 gamma_H1.resize(1);
*/
// ----- H1 calibration end ------


H1.getTFmap()->setlow(64.);
H1.getTFmap()->lprFilter(1,0,120.,4.);
H1.white(60.,w_mode,8.,30.);
vy=H1.getTFmap()->variability();

cout<<"After H1 data conditioning"<<endl; gSystem->Exec("date");


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

wB.Forward(x,levelR);
wB.getLayer(x,0);
H2.getTFmap()->Forward(x,S,levelD);


// ----- calibrate start ------
/*
readCalibration("H2",refname_H2, facname_H2, refstart_H2, reflength_H2, start, 
		end-start, nrc_H2, fstep_H2, &R_H2, &C_H2, alpha_H2, gamma_H2);
 calZ=wZ.calibrate(nrc_H2,fstep_H2,R_H2,C_H2,alpha_H2,gamma_H2,1);
 delete [] R_H2;
 delete [] C_H2;
 alpha_H2.resize(1);
 gamma_H2.resize(1);
*/
// ----- calibrate end ------


H2.getTFmap()->setlow(64.);
H2.getTFmap()->lprFilter(1.,0,120.,4.);
H2.white(60.,w_mode,8.,30.);
vz=H2.getTFmap()->variability();

cout<<"After H2 data conditioning"<<endl; gSystem->Exec("date");

double R = x.rate();          // original data rate

wB.resize(1);
 x.resize(1);

cout<<L1.getTFmap()->size()<<" "<<H1.getTFmap()->size()<<" "<<H2.getTFmap()->size()<<endl<<endl;


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
netevent netburst(3);
TTree* net_tree = netburst.setTree();
variability wavevar;
TTree* var_tree = wavevar.setTree();
wavenoise noiserms;
TTree* noise_tree = noiserms.setTree();
injection mdc(3);
TTree* mdc_tree = mdc.setTree();


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// low pass filtering
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 cout<<"live time: "<<NET.setVeto(4.)<<endl;  // set veto array for injections

int low = (int(R+0.5)>>(levelD))/2; // finest frequency resolution
int n   = lpfcut/low;               // number of layers to zero

for(i=0; i<n; i++){
   L1.getTFmap()->getLayer(x,i); x = 0.; L1.getTFmap()->putLayer(x,i);  // zero level i
   H1.getTFmap()->getLayer(x,i); x = 0.; H1.getTFmap()->putLayer(x,i);  // zero level i
   H2.getTFmap()->getLayer(x,i); x = 0.; H2.getTFmap()->putLayer(x,i);  // zero level i
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// start of the coherent search
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// loop over TF resolutions

for(i=levelD; i>=l_low; i--)
{
  gSystem->Exec("date");

  if(i<=l_high) {

    pLN = pln+3*(i-1); // pointer to lognormal parameters
   
    sprintf(filtername,"%s/Meyer1024_L%1d.dat",data_dir,i);
    cout<<filtername<<endl;

    L1.readFilter(filtername); H1.setFilter(L1); H2.setFilter(L1); 
    NET.setFilter(&H1,0.011); NET.setDelay(&H1); NET.setDelay(&H2);
    
    Ao=logNormArg(bpp,pLN[0],pLN[1],pLN[2]);
    cout<<"pixel threshold in units of noise rms: "<<Ao<<endl;

    cout<<"core pixels: "<<NET.coherence3(Ao,4.,0.)<<"  ";

    n = size_t(2.*Tgap*L1.getTFmap()->resolution(0)+0.1);
    m = size_t(Fgap/L1.getTFmap()->resolution(0)+0.1);
    if(n<1) n = 1;
    
    cout<<"clusters: "<<NET.cluster(n,m)<<"  ";
    cout<<"l pixels: "<<NET.likelihood3('l',true,Acore)<<" ";
    NET.corrcut(0.1,4,0);

    for(j=0; j<NET.nLag; j++) {
      pwc = NET.getwc(j);
      wc = *pwc;
      cout<<wc.csize()<<"|"<<wc.size()<<"|"<<WC[j].append(wc)<<" "; 
    }

    cout<<endl;
  }

  if(i>l_low) NET.Inverse(1); 

}

size_t nevent=0;
for(j=0; j<lags; j++){
  m = WC[j].supercluster('L',0,false);
  cout<<m<<"|";
  pwc = NET.getwc(j);
  *pwc = WC[j];
  cout<<pwc->size()<<" ";
  nevent += m;
}
cout<<endl;
cout<<"Search done\n";
cout<<"number of events: "<<nevent<<endl;

gSystem->Exec("date");

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// finalize likelihood and coordinate reconstruction
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

NET.setRMS();
NET.delink();

cout<<"l pixels: "<<NET.likelihood3('l',false,Acore)<<" ";
NET.corrcut(0.2,0,-int(R/(L1.getTFmap()->maxLayer()+1)+0.5));
cout<<"L pixels: "<<NET.likelihood3('L',false,Acore)<<" \n";
NET.setRank(8.);
gSystem->Exec("date");

for(i=l_low; i<l_high; i++)
{
  
   NET.Forward(1);
 
   //   sprintf(filtername,"/home/klimenko/wat/wat-4.1.0/data/Meyer1024_L%1d.dat",i+1);
   sprintf(filtername,"%s/Meyer1024_L%1d.dat",data_dir,i+1);
   cout<<filtername<<endl;

   L1.readFilter(filtername); H1.setFilter(L1); H2.setFilter(L1); 
   NET.setFilter(&H1,0.011); NET.setDelay(&H1); NET.setDelay(&H2);

   cout<<"l pixels: "<<NET.likelihood3('l',false,Acore)<<" ";
   NET.corrcut(0.2,0,-int(R/(L1.getTFmap()->maxLayer()+1)+0.5));
   cout<<"L pixels: "<<NET.likelihood3('L',false,Acore)<<" \n";
   NET.setRank(8.);
   gSystem->Exec("date");
}


nevent = 0;
for(j=0; j<lags; j++) {
   pwc = NET.getwc(j);
   wc = *pwc; *pwc  = wc;                      // copy selected superclusters
   nevent += pwc->supercluster('L',20,true);   // reconstruct superclusters
}
cout<<"number of events: "<<nevent<<endl;


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// save data in root file
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 netburst.output(net_tree,&NET,factor);
 mdc.output(mdc_tree,&NET,factor);
 /*
wavevar.output(var_tree,&vx,1,waveoffset);
wavevar.output(var_tree,&vy,2,waveoffset);
wavevar.output(var_tree,&vz,3,waveoffset);
noiserms.output(noise_tree,&L1.nRMS,1,R/2);
noiserms.output(noise_tree,&H1.nRMS,2,R/2);
noiserms.output(noise_tree,&H2.nRMS,3,R/2);
 */
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







