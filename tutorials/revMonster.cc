#include "monster.hh"
#include "WDM.hh"

#include "TRandom3.h"
#include "TSystem.h"

const int nRES = 6;
WSeries<double> pTF[nRES];
WSeries<double> subTF;
monster ovlp;

void init()
{  
   // read catalog
   
   char name[1024];
   sprintf(name, "%s/wdmXTalk/OverlapCatalog_Lev_8_16_32_64_128_256_iNu_4_Prec_10.bin", 
      gSystem->Getenv("HOME_WAT_FILTERS") );
   ovlp.read(name);

   // create WDM objects corresponding to the catalog resolutions   
   int layers[nRES]  = {8, 16,32,64,128,256};  
   WDM<double>* wdm[nRES];
   for(int i=0; i<nRES; i++) wdm[i] = new WDM<double>(layers[i],layers[i], 4, 8);
      
  
   // create data: noise in Gaussian envelope to avoid boundary effects
   
   double Rate = 2048;
   double Duration = 10; 
   double Decay2 = 1.5*1.5; 
   TRandom3 rnd0(0);
   
   wavearray<double> ts(Rate*Duration);
   ts.rate(Rate);
   
   for(int i=0; i<Rate*Duration; ++i){
      double ii = i/Rate - Duration/2;
      ts[i] = 10*rnd0.Gaus()*exp(-ii*ii/2/Decay2);
   }
   
   // create TF maps   
   for(int i=0; i<nRES; i++) pTF[i].Forward(ts, *wdm[i]);
}

void testMonster(int r1, int r2)
{  
   if(r2>r1){ printf("Error: r2 must not be greater than r1!\n"); return;}
   
   
      
   int M1 = pTF[r1].getLevel();
   int size1 = pTF[r1].size()/2;
   double* map00_r1 = pTF[r1].data;
   //for(int i=0; i<size1; ++i) map00_r1[i] = 0.1;

   subTF = pTF[r2];
   int M2 = pTF[r2].getLevel();
   int size2 = pTF[r2].size()/2;
   double* map00_r2 = subTF.data;
  
   //printf("%d %d\n", M1, M2);
   
   int k = (M1/M2)*(M2+1);
   
   for(int m=0; m<=M1; ++m){
      int odd = 0;
      for(int i=m; i<size1; i+=M1+1){
         
         xtalkArray tmp = ovlp.catalog[r1][r2][m][odd];
         odd = 1- odd;
         
         if(m==0 || m==M1)if(!odd)continue; 
         
         for(int j=0; j<tmp.size; ++j){
            int indx = i/( 2*(M1+1) ) * (2*k) + tmp.data[j].index;
            if(indx<0 || indx>size2)continue;
            map00_r2[indx] -= map00_r1[i]*tmp.data[j].CC[0];
         }
      }
   }
   
   for(int i=0; i<size2; i+=M2+1)map00_r2[i] = map00_r2[i+M2] = 0;
   
   // watplot q;
   // q.plot(subTF, 3);
   // WDMPlotQuad(pTF[r2], 1, 0, 0., 0., 0);
   // Plot(ts);
   
}
