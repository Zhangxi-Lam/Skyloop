/*
#include "TCanvas.h"
#include "TStyle.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TRandom.h"
#include "TMath.h"
#include "/Users/klimenko/wat/trunk/wat/detector.hh"
#include "/Users/klimenko/wat/trunk/wat/network.hh"
#include "/Users/klimenko/wat/trunk/wat/skymap.hh"
#include "/Users/klimenko/wat/trunk/wat/watfun.hh"
#include "/Users/klimenko/wat/trunk/wat/watplot.hh"
#include "/Users/klimenko/wat/trunk/wat/wat.hh"

#define XIFO 4

void dualcorr()
*/
{    
  size_t m=0;
  gStyle->SetPalette(1,0);
  char fname[128];

  TH1F* g0 = new TH1F("g0","g0",200,-0.,1.1);
  TH1F* g1 = new TH1F("g1","g1",200,-1.,1.1);
  TH1F* g2 = new TH1F("g2","g2",200,-1.,1.1);
  TH1F* g3 = new TH1F("g3","g3",200,-1.,1.);
  TH1F* g4 = new TH1F("g4","g4",200,-1.,1.);

  TH2F* q0 = new TH2F("q0","q0",100,-0.,1.1,100,-0.,1.);
  TH2F* q1 = new TH2F("q1","q1",100,-0.,1.1,100,-1.,2.);
  TH2F* q2 = new TH2F("q2","q2",100,-0.,1.1,100,-0.,1.);

  int nIFO = 2;
  double SNR = 0;
  double gamma = 0.05; //gamma*=gamma;
  double delta = 1.;
  double deLTa = 1.;
  //double delta = 0.3;
  //double deLTa = 0;
    
//  if(!m){ 

  network         NET;         // network
  detector L1("L1"); NET.add(&L1);
  detector H1("V1"); NET.add(&H1);
  detector V1("V1"); NET.add(&V1);
  detector A2("J1"); NET.add(&A2);
  
  NET.setSkyMaps(0.5,0.,180.,0.,360.);
  NET.setAntenna(); 
  size_t N = L1.tau.size();
  size_t K = 1;
  size_t M = 512;
  size_t R = 16384;
  size_t i,n,k,kk,nn,mm;
  double a,b,ao,AO;

  WDM<double> M1(M, M, 4,6);
  WSeries<double> HP(M1);
  WSeries<double> HX(M1);
  
  skymap sm=NET.nLikelihood; sm = 0.;
  skymap Sm=NET.nLikelihood; Sm = 0.;
  skymap SM=NET.nLikelihood; SM = 0.;
  
  wavearray<double> hpts(R*32);
  wavearray<double> hxts(R*32);
  wavearray<double> sas(240*2);

  for (i=0; i<hpts.size(); i++){
     hpts.data[i] = gRandom->Gaus(0.,1.); 
     hxts.data[i] = gRandom->Gaus(0.,1.); 
  }  
  
  HP.Forward(hpts);
  HX.Forward(hxts);
  int N2 = HP.size()/2;
    
  double hp,hx,EE,Lp,Xp,Xx,Ec,Et,W,nul,Hh,bo;
  double gr,gp,gx,gR,gI,gc,Lo,E,Lc,La,Cc,ff,FF;
  double C,c,uc,us,vc,vs,co,si,ee,fm,FM,Si,ll,LL;
  double cc,um,vm,hh,xx,xp,lp,et,ec,ss,SS,ed,ED;
  double CC,UM,VM,HH,XX,XP,LP,ET,EC,CO,SI,ni,NI,mi,MI;
  
  double  am[5];
  double  AM[5];
  double  aa[5];
  double  AA[5];
  double  Fp[5];
  double  Fx[5];
  double  fp[5];
  double  fx[5];
  double   s[5];
  double   S[5];
  double   x[5];
  double   X[5];
  double   e[5];
  double   u[5];
  double   v[5];
  double   f[5];
  double   p[5];
  double   q[5];
  double   P[5];
  double   Q[5];
  double   U[5];
  double   F[5];
  double   V[5];
  double   z[5]={1./sqrt(2),1./sqrt(2),0.,0.,0.};
  
  wavearray<double> h1(360);
  wavearray<double> h2(360);
  wavearray<double> h3(360);
  wavearray<double> h4(360);
  
  for(i=0; i<5; i++) { 
    Fp[i]=0.; Fx[i]=0.; aa[i]=0.; AA[i]=0.; am[i]=0.; AM[i]=0.; 
    u[i]=0.; v[i]=0.;  U[i]=0.; V[i]=0.; e[i]=0.;
    f[i]=0.; F[i]=0.; s[i]=0.; S[i]=0.; x[i]=0.; X[i]=0.;
    p[i]=0.; q[i]=0.; P[i]=0.; Q[i]=0.; z[i]=0.;
  }

  size_t mm = size_t(gRandom->Rndm(11.)*L1.tau.size());
  for(i=0; i<4; i++) { 
     p[i] = nIFO>i ? NET.getifo(i)->fp.get(mm) : 0.;
     q[i] = nIFO>i ? NET.getifo(i)->fx.get(mm) : 0.;
  }

//  cout<<mm<<endl;
//  for(nn=0; nn<10; nn++) {
  mm = 0;
  for(nn=0; nn<N; nn++) {
     
//     if(!nn) n = mm;
//     else    n = nn;
    n = nn;

//    n = size_t(gRandom->Rndm(11.)*L1.tau.size());

    for(i=0; i<4; i++) { 
       Fp[i] = nIFO>i ? NET.getifo(i)->fp.get(n) : 0.;
       Fx[i] = nIFO>i ? NET.getifo(i)->fx.get(n) : 0.;
    }
    //Fp[1]*=0.5; Fx[1]*=0.5;
    kk = 0;
    EE=E=Lo=Lc=La=C=W = 0.;
    
    //    if(!nn) {
    m = R*2+(N2-R*2-4)*gRandom->Rndm(11);
    if(!(m%(M+1))) m++;
    if(m%(M+1)==M) m+=2;
 
    double psi = 2.*PI*gRandom->Rndm(11);
    double sss = sin(psi);
    double ccc = cos(psi);

       for(i=0; i<nIFO; i++) {
//
//             ff = p[i]*ccc+q[i]*sss;
//             FF = q[i]*ccc-p[i]*sss;             
             ff = Fp[i]*ccc+Fx[i]*sss;
             FF = Fx[i]*ccc-Fp[i]*sss;
//
//          aa[i] = ff*HP.data[m]    + 1*FF*HP.data[m+N2];
//          AA[i] = ff*HP.data[m+N2] + 1*FF*HP.data[m];
          aa[i] = ff*HP.data[m]    + FF*HX.data[m];
          AA[i] = ff*HP.data[m+N2] + FF*HX.data[m+N2];
       } 
       ET = sqrt(NET.dot4(aa,aa)+NET.dot4(AA,AA));
        
       float noise = 0.;   //000000000000000000000000000000000//
       for(i=0; i<nIFO; i++) {
          m = R*2+(N2-R*2-4)*gRandom->Rndm(11);
          if(!(m%(M+1))) m++;
            if(m%(M+1)==M) m+=2;

            //am[i] = (1-noise)*aa[i]*SNR/ET + noise*HP.data[m];
            //AM[i] = (1-noise)*AA[i]*SNR/ET + noise*HP.data[m+N2];
            am[i] = (1-noise)*aa[i]*SNR/ET + HP.data[m];
            AM[i] = (1-noise)*AA[i]*SNR/ET + HP.data[m+N2];
       } 
       //    }
// transformation to DPF 

    gp = NET.dot4(Fp,Fp);                 // fp^2
    gx = NET.dot4(Fx,Fx);                 // fx^2
    gI = NET.dot4(Fp,Fx)*2;                 // fp*fx
    gR = (gp-gx); 
    gr = (gp+gx);
    gc = 2*sqrt(gR*gR+gI*gI);               // norm of complex antenna pattern
    a = sqrt(0.5+fabs(gR/gc));
    co=gR>0 ? a : fabs(gI)/gc/a; 
    si=gR>0 ? gI/gc/a : a*(gI>0?1:-1); 
    ff = NET.rot4(Fp,co,Fx,si,fp)+1.e-12;  // f+
    FF = NET.rot4(Fx,co,Fp,-si,fx)+1.e-12; // fx
    a = (gr-gc/2)/(gr+gc/2);    
    //NET.dot4(fp,fp,p); NET.dot4(fx,fx,q); NI=(NET.dot4(p,p)/ff+0*NET.dot4(q,q)/FF)/(ff+0*FF);
    

// projection in the plane

    xp = NET.dot4(fp,am);                 // (X*f+)
    xx = NET.dot4(fx,am);                 // (X*fx)       
    XP = NET.dot4(fp,AM);                 // (X*f+)
    XX = NET.dot4(fx,AM);                 // (X*fx)       

// find weak vector 00

    uc = xp/ff;                           // cos of rotation to PCF
    us = xx/FF;                           // sin of rotation to PCF
    um = NET.rot4(fp,uc,fx,us,aa);        // projected response

//    uc = (xp*gx - xx*gI);                 // u cos of rotation to PCF
//    us = (xx*gp - xp*gI);                 // u sin of rotation to PCF
//    um = NET.rot4(fp,uc,fx,us,u);        // calculate u and return its norm
//    et = NET.dot4(am,am);     
//    hh = NET.dot4(am,aa,e);
//    ec = (hh*hh - NET.dot4(e,e))/um;

// find weak vector 90

    uc = XP/ff;                           // cos of rotation to PCF
    us = XX/FF;                           // sin of rotation to PCF
    UM = NET.rot4(fp,uc,fx,us,AA);        // projected response

//    uc = (XP*gx - XX*gI);                 // u cos of rotation to PCF
//    us = (XX*gp - XP*gI);                 // u sin of rotation to PCF
//    UM = NET.rot4(fp,uc,fx,us,U);        // calculate u and return its norm
//    ET = NET.dot4(AM,AM);    
//    HH = NET.dot4(AM,AA,e);  
//    EC = (HH*HH - NET.dot4(e,e))/UM;

//  transformation to DDF

    gp = NET.dot4(aa,aa);          // fp^2
    gx = NET.dot4(AA,AA);          // fx^2
    gI = NET.dot4(aa,AA)*2;        // fp*fx
    gR = (gp-gx); 
    gr = (gp+gx);
    gc = 2*sqrt(gR*gR+gI*gI);               // norm of complex antenna pattern
    //b = (gr-gc/2)/(gr+gc/2); 

    //cc = sqrt((gc+gR)*(gc+gR)+gI*gI)+1.e-24;
    b = sqrt(0.5+fabs(gR/gc));
    co=gR>0 ? b : fabs(gI)/gc/b; 
    si=gR>0 ? gI/gc/b : b*(gI>0?1:-1); 
    ss = NET.rot4(aa,co,AA,si,s)+1.e-12;   // s[k] 
    et = NET.rot4(am,co,AM,si,x)+1.e-24;   // x[k] 
    SS = NET.rot4(AA,co,aa,-si,S)+1.e-24;  // S[k]
    ET = NET.rot4(AM,co,am,-si,X)+1.e-24;  // X[k]
    b = (gr-gc/2)/(gr+gc/2); 

    bo=1-NET.dot4(fp,s)*NET.dot4(fp,s)/(ss*ff);
    //bo=1-NET.dot4(fp,z)*NET.dot4(fp,z)/ff;
    ll = NET.dot4(x,s,p); LL = NET.dot4(X,S,q); c = ll-NET.dot4(p,p)/ll; ni = 1-c/ll;
    ee = (xp*xp+XP*XP)/ff; EE = ni*(xx*xx+XX*XX)/(FF+1.e-12)/(ll+LL); 
    //ao = (FF*(ff-FF)+gamma*(ff-2*sqrt(gamma)*ni)*ff)/nIFO/nIFO;  
    //ao = (FF+gamma*(ff-gamma*ni))/nIFO/2;  bo*=ni; co = (FF+0.01)*(ff+FF)/(nIFO*nIFO); 
    ao = (FF+ff*(1-ni)*(1-ni)-gamma*ni)/nIFO; //bo*=ni; //co = (FF+0.01)*(ff+FF)/(nIFO*nIFO); 
    //if(EE<delta-ni*deLTa) {q0->Fill(EE,1-ni); q1->Fill(sqrt((bo)),ao>0?sqrt(ao):-sqrt(-ao)); q2->Fill(sqrt(bo*ni),sqrt(FF/nIFO/2));}

    // hard constraint
    if(FF/nIFO<EE*gamma||EE>delta-ni*deLTa) {NET.mulx(fp,NET.dot4(x,fp)/ff,s); NET.mulx(fx,NET.dot4(X,fx)/FF,S);} 
    //if(ao<bo*gamma||EE>delta-ni*deLTa) {NET.mulx(fp,NET.dot4(x,fp)/ff,s); NET.mulx(fx,NET.dot4(X,fx)/FF,S);} 
    if(1-ni*deLTa<0.5) NET.mulx(S,0.,S);  //if(ao<0) NET.mulx(s,0.,s);
      
    ll = NET.dot4(x,s,p)+1.e-9; LL = NET.dot4(X,S,q)+1.e-9; cc=ll-NET.dot4(p,p)/ll; CC=LL-NET.dot4(q,q)/LL; 
    //ll = NET.dot4(s,s,p)+1.e-9; LL = NET.dot4(S,S,q)+1.e-9; c=ll-NET.dot4(p,p)/ll; C=LL-NET.dot4(q,q)/LL; 
       
    //if(ao<bo*ni*gamma||EE>delta-ni*deLTa) SM.set(n,(ff+FF)/nIFO);
    //co = (cc+CC)/(fabs(cc+CC)+et+ET-ll-LL);
    co = (ao<0&&1-ni*deLTa<0.5)?-1:(cc+CC)/(fabs(cc+CC)+et+ET-ll-LL);
    if(co>-2) sm.set(n,sqrt((FF+ff)/nIFO));
    //if((ao<gamma*bo||EE>delta-ni*deLTa)) Sm.set(n,1-sqrt(bo));
    if((FF/nIFO<gamma*bo||EE>delta-ni*deLTa)||ao<0) Sm.set(n,1-sqrt(bo));
    //if(FF/nIFO<gamma*EE||EE*0.333>(c+C)/(ll+LL)) Sm.set(n,1-sqrt(bo));
    /*if(FF/nIFO<gamma*EE||EE*0.4>(c+C)/(ll+LL))*/ q0->Fill(EE,(c+C)/(ll+LL));
    q2->Fill(EE,1-ni);
    SM.set(n,co);
    if(co>0.5) g0->Fill(sqrt(bo));
    g1->Fill(co);
    g2->Fill(1-ni-EE);

  }

q0->GetYaxis()->SetTitle("#sqrt{|f+|^2|fx|^2}");
q0->GetXaxis()->SetTitle("#sqrt{ni*Lx/L}");
q0->SetTitle("");
//q0->SetStats(kFALSE);
q0->SetStats(kTRUE);

q1->GetYaxis()->SetTitle("#sqrt{|f+|^2|fx|^2}");
q1->GetXaxis()->SetTitle("#sqrt{ni*Lx/L}");
q1->SetTitle("");
//q1->SetStats(kFALSE);
q1->SetStats(kTRUE);

q2->GetYaxis()->SetTitle("#sqrt{|f+|^2|fx|^2}");
q2->GetXaxis()->SetTitle("#sqrt{ni*Lx/L}");
q2->SetTitle("");
//q2->SetStats(kFALSE);
q2->SetStats(kTRUE);

g0->GetYaxis()->SetTitle("events");
g0->GetXaxis()->SetTitle("#sqrt{ni*Lx/L}");
g0->SetTitle("");
g0->SetStats(kTRUE);

g1->GetYaxis()->SetTitle("events");
g1->GetYaxis()->SetTitleOffset(1.6);
g1->GetXaxis()->SetTitle("cc");
g1->SetTitle("");
g1->SetStats(kTRUE);

g2->GetYaxis()->SetTitle("events");
g2->GetXaxis()->SetTitle("Ao");
g2->SetTitle("");
g2->SetStats(kTRUE);

g3->GetYaxis()->SetTitle("events");
g3->GetXaxis()->SetTitle("1p null");
g3->SetTitle("");
g3->SetStats(kTRUE);

g4->GetYaxis()->SetTitle("events");
g4->GetXaxis()->SetTitle("1p null");
g4->SetTitle("");
g4->SetStats(kTRUE);

g0->SetLineColor(1);
g1->SetLineColor(1);
g2->SetLineColor(2);
g3->SetLineColor(3);
g4->SetLineColor(4);
/*
watplot PP("plot",0,0,600,600);
//PP.canvas->SetLogy(kTRUE);
TCanvas *ca = PP.canvas;
//new TCanvas("c","C",0,0,600,600);
ca->SetBorderMode(0);
ca->SetFillColor(0);
ca->SetBorderSize(2);
ca->SetGridx();
ca->SetGridy();
ca->SetBottomMargin(0.143939);
ca->SetRightMargin(0.1517039);
ca->SetTopMargin(0.0772727);
*/
/*
 PP.plot(sm);
 sprintf(fname,"sm1corr.png");
 ca->Update(); ca->SaveAs(fname);

 PP.plot(Sm);
 sprintf(fname,"sm2corr.png");
 ca->Update(); ca->SaveAs(fname);


g0->Draw();
g2->Draw("same");
sprintf(fname,"g0corr.png");
ca->Update(); ca->SaveAs(fname);
g1->Draw();
g3->Draw("same");
sprintf(fname,"g1corr.png");
ca->Update(); ca->SaveAs(fname);
g2->Draw();
sprintf(fname,"g2corr.png");
ca->Update(); ca->SaveAs(fname);
g3->Draw();
sprintf(fname,"g3corr.png");
ca->Update(); ca->SaveAs(fname);
ca->SetLogy(kFALSE);
ca->SetLogz(kTRUE);
q0->Draw("colz");
sprintf(fname,"q0corr.png");
ca->Update(); ca->SaveAs(fname);
q1->Draw("colz");
sprintf(fname,"q1corr.png");
ca->Update(); ca->SaveAs(fname);
*/

/*
PP.canvas->SetLogz(kTRUE);

 PP.plot(h1,0,1);
 PP.plot(h4,1,2);
 sprintf(fname,"xcorr.png");
 PP.canvas->Update(); PP.canvas->SaveAs(fname);
 PP.plot(h4,1,2);
 sprintf(fname,"scalar.png");
 PP.canvas->Update(); PP.canvas->SaveAs(fname);
*/   

}
