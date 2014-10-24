{
  size_t nIFO = 4;
  char ifoname[5][4] = {"L1", "H1", "H2", "V1", "V1"};
  char netdir[256] = "plot/L1H1H2V1bII";   // output directory for netplot.C
  char simdir[256] = "plot/SG1";           // output directory for simplot.C

// thresholds (used in netplot.C and simplot.C)

double T_cor = 0.6;    // network correlation coefficient
double T_cut = 3.4;    // network effective SNR f>200Hz
double T_CUT = 4.5;    // network effective SNR f<200Hz
    
  TChain var("variability");
  TChain rms("wavenoise");
  TChain net("waveburst");
  TChain sim("waveburst");
  TChain mdc("mdc");
  TChain liv("liveTime");
  
  TChain VAR("variability");
  TChain RMS("wavenoise");
  TChain NET("waveburst");
  TChain SIM("waveburst");
  TChain MDC("mdc");
  TChain LIV("liveTime");

// Add files

//sim.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_G1H1H2L1V1_run25.merged/w_SG1_S5_LV_G1H1H2L1V1_run25.root");
//mdc.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_G1H1H2L1V1_run25.merged/mdc_SG1_S5_LV_G1H1H2L1V1_run25.root");
//net.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_G1H1H2L1V1_run25.merged/w_S5_LV_G1H1H2L1V1_run25.root");
//liv.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_G1H1H2L1V1_run25.merged/lt_S5_LV_G1H1H2L1V1_run25.root");

//sim.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_G1H1H2L1_run25.merged/w_SG1_S5_LV_G1H1H2L1_run25.root");
//mdc.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_G1H1H2L1_run25.merged/mdc_SG1_S5_LV_G1H1H2L1_run25.root");
//net.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_G1H1H2L1_run25.merged/w_S5_LV_G1H1H2L1_run25.root");
//liv.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_G1H1H2L1_run25.merged/lt_S5_LV_G1H1H2L1_run25.root");

sim.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_H1H2L1V1_run25.merged/w_SG1_S5_LV_H1H2L1V1_run25.root");
mdc.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_H1H2L1V1_run25.merged/mdc_SG1_S5_LV_H1H2L1V1_run25.root");
net.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_H1H2L1V1_run25.merged/w_S5_LV_H1H2L1V1_run25.root");
liv.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_H1H2L1V1_run25.merged/lt_S5_LV_H1H2L1V1_run25.root");

//sim.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_H1H2L1_run25.merged/w_SG1_S5_LV_H1H2L1_run25.root");
//mdc.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_H1H2L1_run25.merged/mdc_SG1_S5_LV_H1H2L1_run25.root");
//net.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_H1H2L1_run25.merged/w_S5_LV_H1H2L1_run25.root");
//liv.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_H1H2L1_run25.merged/lt_S5_LV_H1H2L1_run25.root");

//sim.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_H1L1_run25.merged/w_SG1_S5_LV_H1L1_run25.root");
//mdc.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_H1L1_run25.merged/mdc_SG1_S5_LV_H1L1_run25.root");
//net.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_H1L1_run25.merged/w_S5_LV_H1L1_run25.root");
//liv.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_H1L1_run25.merged/lt_S5_LV_H1L1_run25.root");

//sim.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_H1H2_run25.merged/w_SG1_S5_LV_H1H2_run25.root");
//mdc.Add("/archive/home/igor/public_html/LV1/OUTPUT_SG1_S5_LV_H1H2_run25.merged/mdc_SG1_S5_LV_H1H2_run25.root");
//net.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_H1H2_run25.merged/w_S5_LV_H1H2_run25.root");
//liv.Add("/archive/home/igor/public_html/LV1/OUTPUT_S5_LV_H1H2_run25.merged/lt_S5_LV_H1H2_run25.root");

netevent  W(&net,nIFO);
netevent  ww(&sim,nIFO);
injection mm(&mdc,nIFO);

//string CUT = "lag[0]>0&&run<=7244&&run<24450&&run!=23218&&"+RSN+">0";

 if(nIFO==2) {

   string L0  = "(snr[0]-null[0])";
   string L1  = "(snr[1]-null[1])";
   
   string snr  = "(snr[0]+snr[1])";
   string nil  = "(nill[0]+nill[1])";
   string nul  = "(null[0]+null[1])";

   string aSNR = "sqrt(min(min("+L0+","+L1+"),4*ecor)/2)";
   
   string L09  = L0+">9&&"+L1+">9";
   string L16  = L0+">16&&"+L1+">16";
   string L25  = L0+">25&&"+L1+">25";
   string L36  = L0+">36&&"+L1+">36";
   
   string rsn  = "sqrt((rSNR[0]+rSNR[1])/2)";
   string RSN  = "pow(rSNR[0]*rSNR[1],1./4)";
   string rsf  = "log10((rSF[0]+rSF[1])/2.)";
   string RSF  = "log10(rSF[0]*rSF[1])/2.";
   
   string cutT = "(abs(1000*(time[0]-time[2]))/1000.<0.1)";
 }

 if(nIFO==3) {

   string L01  = "(snr[0]-null[0]+snr[1]-null[1])";
   string L02  = "(snr[0]-null[0]+snr[2]-null[2])";
   string L12  = "(snr[1]-null[1]+snr[2]-null[2])";
   
   string snr  = "(snr[0]+snr[1]+snr[2])";
   string nil  = "(nill[0]+nill[1]+nill[2])";
   string nul  = "(null[0]+null[1]+null[2])";

   string aSNR = "sqrt(min(min("+L01+","+L02+"),min("+L12+",8*ecor/3))/2)";
   
   string L10  = L01+">10&&"+L02+">10&&"+L12+">10";
   string L16  = L01+">16&&"+L02+">16&&"+L12+">16";
   string L25  = L01+">25&&"+L02+">25&&"+L12+">25";
   string L36  = L01+">36&&"+L02+">36&&"+L12+">36";
   string L50  = L01+">50&&"+L02+">50&&"+L12+">50";
   string L70  = L01+">70&&"+L02+">70&&"+L12+">70";
   
   string rsn  = "sqrt((rSNR[0]+rSNR[1]+rSNR[2])/3)";
   string RSN  = "pow(rSNR[0]*rSNR[1]*rSNR[2],1./6)";
   string rsf  = "log10((rSF[0]+rSF[1]+rSF[2])/3.)";
   string RSF  = "log10(rSF[0]*rSF[1]*rSF[2])/3.";
   
   string cutT = "(abs(1000*(time[0]-time[3]))/1000.<0.1)";
 }


 if(nIFO==4) {

   string L012 = "(snr[0]-null[0]+snr[1]-null[1]+snr[2]-null[2])";
   string L013 = "(snr[0]-null[0]+snr[1]-null[1]+snr[3]-null[3])";
   string L023 = "(snr[0]-null[0]+snr[2]-null[2]+snr[3]-null[3])";
   string L123 = "(snr[1]-null[1]+snr[2]-null[2]+snr[3]-null[3])";
   
   string snr  = "(snr[0]+snr[1]+snr[2]+snr[3])";
   string nil  = "(nill[0]+nill[1]+nill[2]+nill[3])";
   string nul  = "(null[0]+null[1]+null[2]+null[3])";

   string aSNR = "sqrt(min(min(min("+L012+","+L013+"),min("+L023+","+L123+")),2*ecor)/2)";
   
   string L10  = L012+">10&&"+L013+">10&&"+L023+">10&&"+L123+">10";
   string L16  = L012+">16&&"+L013+">16&&"+L023+">16&&"+L123+">16";
   string L25  = L012+">25&&"+L013+">25&&"+L023+">25&&"+L123+">25";
   string L36  = L012+">36&&"+L013+">36&&"+L023+">36&&"+L123+">36";
   string L50  = L012+">50&&"+L013+">50&&"+L023+">50&&"+L123+">50";
   string L70  = L012+">70&&"+L013+">70&&"+L023+">70&&"+L123+">70";
   
   string rsn  = "sqrt((rSNR[0]+rSNR[1]+rSNR[2]+rSNR[3])/4)";
   string RSN  = "pow(rSNR[0]*rSNR[1]*rSNR[2]*rSNR[3],1./8)";
   string rsf  = "log10((rSF[0]+rSF[1]+rSF[2]+rSF[3])/4.)";
   string RSF  = "log10(rSF[0]*rSF[1]*rSF[2]*rSF[3])/4.";
   
   string cutT = "(abs(1000*(time[0]-time[4]))/1000.<0.1)";
 }

 if(nIFO==5) {

   string L012 = "(snr[0]-null[0]+snr[1]-null[1]+snr[2]-null[2])";
   string L013 = "(snr[0]-null[0]+snr[1]-null[1]+snr[3]-null[3])";
   string L023 = "(snr[0]-null[0]+snr[2]-null[2]+snr[3]-null[3])";
   string L123 = "(snr[1]-null[1]+snr[2]-null[2]+snr[3]-null[3])";
   
   string snr  = "(snr[0]+snr[1]+snr[2]+snr[3])";
   string nil  = "(nill[0]+nill[1]+nill[2]+nill[3])";
   string nul  = "(null[0]+null[1]+null[2]+null[3])";

   string aSNR = "sqrt(min(min(min("+L012+","+L013+"),min("+L023+","+L123+")),2*ecor)/2)";
   
   string L10  = L012+">10&&"+L013+">10&&"+L023+">10&&"+L123+">10";
   string L16  = L012+">16&&"+L013+">16&&"+L023+">16&&"+L123+">16";
   string L25  = L012+">25&&"+L013+">25&&"+L023+">25&&"+L123+">25";
   string L36  = L012+">36&&"+L013+">36&&"+L023+">36&&"+L123+">36";
   string L50  = L012+">50&&"+L013+">50&&"+L023+">50&&"+L123+">50";
   string L70  = L012+">70&&"+L013+">70&&"+L023+">70&&"+L123+">70";
   
   string rsn  = "sqrt((rSNR[0]+rSNR[1]+rSNR[2]+rSNR[3])/4)";
   string RSN  = "pow(rSNR[0]*rSNR[1]*rSNR[2]*rSNR[3],1./8)";
   string rsf  = "log10((rSF[0]+rSF[1]+rSF[2]+rSF[3])/4.)";
   string RSF  = "log10(rSF[0]*rSF[1]*rSF[2]*rSF[3])/4.";
   
   string cutT = "(abs(1000*(time[0]-time[4]))/1000.<0.1)";
 }

 string efrsnr  = "pow("+rsn+",netcor)"; 
 string efsnr1  = "pow("+aSNR+",netcor)"; 
 string efsnr2  = "pow("+aSNR+",sqrt(netcor))"; 


 gStyle->SetPalette(1,0);

 TCanvas *c1 = new TCanvas("c","C",0,0,600,600);
 c1->SetBorderMode(0);
 c1->SetFillColor(0);
 c1->SetBorderSize(2);
 c1->SetLogx(kFALSE);
 c1->SetGridx();
 c1->SetGridy();
 c1->SetRightMargin(0.1517039);
 c1->SetTopMargin(0.0772727);
 c1->SetBottomMargin(0.103939);


}



