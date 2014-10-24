{

char* ifo  = "H1";
char* ifo1 = "H2";
char* ifo2 = "H1";
char* site = "Hanford";
int   I    = 0;           // channel index-1 (0 or 1)


FILE* in;
char s[512];
char channel[256];
char fname[512];
int m,k;

gStyle->SetPalette(1,0);

TChain wave("waveburst");
TChain rms("noise");
TChain var("variability");
TChain net("netwaveburst");

sprintf(fname,"/home/klimenko/wat/wat-4.1.0/plots/rootFiles.lst");
//sprintf(fname,"/home/klimenko/analysis/S3/v5/ligo-geo/rootFiles.lst");
//sprintf(fname,"/home/klimenko/analysis/E11/e11h2/rootFiles.lst");
//sprintf(fname,"/home/klimenko/analysis/E12/rootFiles.lst");
//sprintf(fname,"/home/klimenko/analysis/S3/v5/ligo-3.7.5/rootFiles.lst");
//sprintf(fname,"/home/klimenko/analysis/S3/v5/pgMDCnew1/rootFiles.lst");
//sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/SG4_S3_PB1/rootFiles.lst");
//sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/GA1_S3_PB1/rootFiles.lst");
//sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/pgMDC/SG4_S3_Pb/rootFiles.lst");
//sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/SG4_S3_P/rootFiles.lst");
//sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/WNB2A_S3_P/rootFiles.lst");

if( (in=fopen(fname,"r"))!=NULL ){ 
  while(fgets(s,512,in) != NULL){
    if(m++ > 10000) break;
    for(int i=strlen(s)-1; i<512; i++) s[i]='\0';
//    wave.Add(s);
//    var.Add(s);
//    rms.Add(s);
    net.Add(s);
  }
  fclose(in);
}

netevent W(&net,3);

/*
sprintf(fname,"/home/klimenko/analysis/S3/v5/ligo-3.7.5/rootFiles1.lst");
//sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/SG4_S3_P/wxrootFiles.lst");
//sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/playground-3.6/xcrootFiles.lst");
//sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/pgMDC/SG5_S3_Pb/rootFiles.lst");
TChain WAVE("waveburst");

if( (in=fopen(fname,"r"))!=NULL ){ 
  while(fgets(s,512,in) != NULL){
    if(m++ > 1000) break;
    for(int i=strlen(s)-1; i<512; i++) s[i]='\0';
    WAVE.Add(s);
//    rms.Add(s);
//    cout<<s<<endl;
//    xc3.Add(s);
//    var.Add(s);
  }
  fclose(in);
}


//wbtriple W(&sg5);

//sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/pgLG-3.7.5/rootFiles.lst");
sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/pgMDClast/sgrootFiles.lst");
//sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/playgroundN/rootFiles.lst");
TChain pgn("waveburst");
cout<<fname<<endl;

m=0;
if( (in=fopen(fname,"r"))!=NULL ){ 
  while(fgets(s,512,in) != NULL){
    if(m++ > 1000) break;
    for(int i=strlen(s)-1; i<512; i++) s[i]='\0';
    cout<<s<<endl;
    pgn.Add(s);
  }
  fclose(in);
}

sprintf(fname,"/cdf/tmp1/ligosoft/S3/v5/pgMDCnew1/S3P6Bnov.f1/rootFiles.lst");
TChain pg5("waveburst");

if( (in=fopen(fname,"r"))!=NULL ){ 
  while(fgets(s,512,in) != NULL){
    if(m++ > 1000) break;
    for(int i=strlen(s)-1; i<512; i++) s[i]='\0';
    pg5.Add(s);
  }
  fclose(in);
}

//wbtriple W(&pg5);

//triple W(&wave);
//triple WB(&wave);


TFile *f0 = new TFile("/cdf/tmp1/ligosoft/mrf/plots/veto_asi.root", "READ");
TTree* v0  = (TTree*)f0->Get("netwaveburst");
*/

//TF1 *sig = new TF1("fit",sfit,-22,-17.8,3);
//sig->SetParameters(4000,-20,2.);
//sig->SetParNames ("max","h50","width"); 

TCanvas *c1 = new TCanvas("c","C",0,0,800,600);
gPad->SetBorderMode(0);
gPad->SetFillColor(0);



}



