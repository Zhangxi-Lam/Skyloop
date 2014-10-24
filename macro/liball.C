#include <math.h>
#include <stdio.h>
#include <iostream.h>
#include <TRandom.h>
#include "wavelet.h"
#include "wavedata.h"


/**************************/
/* bit operation library  */
/**************************/

int bitSum(int n)
{
   long ns = long(n);
   int zero = 0;
   for(; ns!=0; ns>>=1){      
      zero += ns & 01;
   }
   return 16-2*zero;
}

int bitSum(double a)
{
   long ns = long(a);
   int zero = 0;
   for(; ns!=0; ns>>=1){      
      zero += ns & 01;
   }
   return 16-2*zero;
}


int bitSum(WaveData &a, int n=0)
{
   int minus=0;
   if(n<=0 || n>a.N) n=a.N;
   long ns;
   for(int i=0; i<n; i++){
      ns = long(a.data[i]);
      while(ns!=0){
	 minus += ns & 01;
	 ns>>=1;
      }
   }
   return n*16-2*minus;
}

double bitSum(WaveData &a, WaveData& bm, int n=0)
{
   double zero=0;
   if(n<=0 || n>a.N) n=a.N;
   long ns;
   for(int i=0; i<n; i++)
      zero += bm.data[long(a.data[i])];
   return zero;
}


WaveData* bitLook()
{
   int n=0177777+1;
   WaveData* p=new WaveData(n);
   for(int i=0; i<n; i++){
      p->data[i] = bitSum(i);
   }
   return p;
}



/*************************************************************** 
 * bitRMS averages correlation statistics for n*16 bits and    *
 * calculates variance of the n*16 bit correlation coefficient.* 
 ***************************************************************/
double bitRMS(WaveData &a, WaveData &bm, int n=1)
{
   int minus;
   if(a.Rate == 0.) a.Rate=1.;
   if(n<=0 || n>a.N) n=1;
   long ns;
   int i,j;
   double r,r2=0.;
   for(i=0; i<=(a.N-n); i+=n){
      minus=0;
      for(j=0; j<n; j++){
	 minus += int(bm.data[long(a.data[i+j])]);
      }
      r = minus/16./n;
      r2 += r*r;
   }
   return n*16*r2/(a.N/n)/a.Rate;
}


long* bitSign(WaveData &a, int &n)
{
   double x;
   n = a.N/16;
   long* b = new long[n];
   long ns;

   for(int i=0; i<n; i++){
      ns=0;
      for(int j=0; j<16; j++){
	 x=a.data[i*16+j];
         if(x<=0.) {ns++;}
	 ns<<=1;
      }
      ns>>=1;
      b[i]=ns;
   }
   return b;
}

WaveData bitSign(WaveData &a)
{
   double x;
   int n = a.N/16;
   WaveData b(n);
   b.Rate=a.Rate;
   long ns;

   for(int i=0; i<n; i++){
      ns=0;
      for(int j=0; j<16; j++){
	 x=a.data[i*16+j];
         if(x<=0.) {ns++;}
	 ns<<=1;
      }
      ns>>=1;
      b.data[i]=ns;
   }
   return b;
}

/* calculate correlation statistics for time shift lag */
WaveData* bitORex(WaveData &a, WaveData &b, int lag=0)
{
   double x;
   int n;

   if(a.N>b.N) n=b.N;
   else        n=a.N;
   n-=abs(lag);

   int la, lb;
   if(lag<0){la=-lag; lb=0;}
   else     {la=0; lb=lag;}

   double* pa = &(a.data[la]);
   double* pb = &(b.data[lb]);

   WaveData* p = new WaveData(n);
   p->Rate=a.Rate;
   for(int i=0; i<n; i++){
      p->data[i] = long(pa[i])^long(pb[i]);
   }
   return p;
}


/* calculate correlation coefficient for time shift lag */
double bitSum(WaveData &a, WaveData &b, WaveData &bm, int lag=0)
{
   int n;
   if(a.N>b.N) n=b.N;
   else        n=a.N;
   n-=abs(lag);

   int la, lb;
   if(lag<0){la=-lag; lb=0;}
   else     {la=0; lb=lag;}

   double* pa = &(a.data[la]);
   double* pb = &(b.data[lb]);

   double r=0;
   for(int i=0; i<n; i++){
      r += bm.data[int(pa[i])^int(pb[i])];
//      pa[i]=bm.data[int(pa[i])^int(pb[i])];
   }
   return r/16./n;
}

/* median of the distribution */
double median(WaveData &a, double range=1.)
{
   double avr,rms;
   int nx,ny,nz;
   int i;
   double data;
   bool error = false;
   a.getStatistics(avr,rms);
   double x,y,z;
   x = avr;

   for(int j=0; j<2; j++){
      y = avr + range*rms;
      z = avr - range*rms;

      nx = ny = nz = 0;
      for(i=0; i<a.N; i++){
	 data =a.data[i];
	 nx += (data<x) ? -1 : 1;
	 ny += (data<y) ? -1 : 1;
	 nz += (data<z) ? -1 : 1;
      }

      if(nx<0){
	 y = z;
	 ny = nz;
	 if(nz<0) error = true;
      }
      else{
	 if(ny>0) error = true;
      }
      if(!error) break;
      range = 1.;
   }

   do{
      z = (x+y)/2.;
      nz = 0;
      for(i=0; i<a.N; i++){
	 nz += (a.data[i]<z) ? -1 : 1;
      }

      if(nx*nz<ny*nz){
	 y = z;
	 ny = nz;
      }
      else{
	 x = z;
	 nx = nz;
      }

//   cout << nx <<"  "<< ny <<"  "<< nz << endl;

   } while(abs(nz)>1);
      
//   cout << nz <<"  "<< z<<endl;
   return z;
}


/***************************/
/*  general functions      */
/***************************/

void intw(WaveData &td)
{
   wavereal a;
   for(int i = 0; i<td.N; i++){
      a = td.data[i];
      if(a>0) a += 0.5;
      if(a<0) a -= 0.5;
      td.data[i] = int(a);
   }
   return;
}

void intw1(WaveData &td)
{
   wavereal a;
   for(int i = 0; i<td.N; i++){
      a = td.data[i];
      if(a>0) a += 1.;
      if(a<0) a -= 1.;
      td.data[i] = int(a);
   }
   return;
}

double haar(WaveData &x, int n){
   int N=x.N;
   WaveData d;
   WaveData a;
   double rate=x.Rate;
   double hsum=0.;
   int nzero=0;

   for(int i=1; i<=n; i++){
      N=N>>1;
      if(N<1) break;
      d.Resize(N);
      a.Resize(N);
      for(int j=0; j<N; j++){
	 a.data[j]=x.data[2*j];
	 d.data[j]=x.data[2*j+1];
      }
      d-=a;    // details
      d*=0.5;  // update
//      intw(d); // integer transform
      a+=d;    // approximation
//      intw(a);

      x=a;
      rate/=2.;
      x.Rate=rate;
   }
//   intw(x);

   for(int j=0; j<N; j++)
      hsum+=x.data[j];

   return hsum/N;
}

int sign(WaveData &a)
{
   double x;
   int flip = 0;
   double alast = 0; 
   alast=a.data[0];
   for(int i=0; i<a.N; i++){
      x=a.data[i];
      if(x>0.) a.data[i]=1.; 
      if(x<=0.) {a.data[i]=-1.; flip++;}
//      if(x==0.) a.data[i]=0.;
/*
      if(a.data[i]!=alast){
	 flip++;
	 alast=a.data[i];
      }
*/
   }
   return a.N-2*flip;
}


int nzero(WaveData &a)
{
   int zero=0;
   for(int i=0; i<a.N; i++)
      if(a.data[i]==1) zero++;

   return zero;
}

void x_test(WaveData &a, int n)
{
   WaveData b;
   double avr, rms;
   a.getStatistics(avr,rms);
   b=a;
   a.N = int(b.N/n);
   a.Resize(a.N);
   for(int i=0; i<a.N; i++){
      a.data[i] = 0.;
      for(int j=0; j<n; j++)
	 a.data[i] += b.data[i*n+j]*b.data[i*n+j];
      a.data[i] *= 1./n/rms/rms; 
   }
      return;
}

void t_test(WaveData &a, int n)
{
   WaveData b;
   double avr, rms;
   b=a;
   a.N = int(b.N/n);
   a.Resize(a.N);
   for(int i=0; i<a.N; i++){
      avr = 0.; rms = 0.;
      for(int j=0; j<n; j++){
	 avr += b.data[i*n+j];
	 rms += b.data[i*n+j]*b.data[i*n+j];
      }
      avr /= n;
      rms /= n-1;
      a.data[i] = avr*sqrt(n/rms); 
   }
      return;
}

int s2b(short a)
{
      int k;
      int d=a<0?-a:a;
      for(k=0;(1<<k)<d;k++);
      if(d!=0) k++;
      return k;
}

double diff(WaveData &a)
{
      int k;
      for(k=a.N-1; k>0; k--)
	 a.data[k] -= a.data[k-1];
      return a.data[0];
}

void wint(WaveData &a)
{
      int k;
      for(k=1; k<a.N; k++)
	 a.data[k] += a.data[k-1];
      return;
}

double wmin(WaveData &a)
{
      int k;
      double wm = a.data[0]; 
      for(k=1; k<a.N; k++)
	 if(a.data[k]<wm) wm = a.data[k];
      return wm;
}

double wmax(WaveData &a)
{
      int k;
      double wm = a.data[0]; 
      for(k=1; k<a.N; k++)
	 if(a.data[k]>wm) wm = a.data[k];
      return wm;
}

void winvert(WaveData &a)
{
      int k;
      double wmi = a.data[0]; 
      double wma = a.data[0]; 
      for(k=1; k<a.N; k++){
	 if(a.data[k]>wma) wma = a.data[k];
	 if(a.data[k]<wmi) wmi = a.data[k];
      }

      a-=(wma+wmi)/2.;
      double wm = (wma-wmi)/2.-0.5;

      for(k=0; k<a.N; k++)
	 a.data[k] += (a.data[k]>0) ? -wm : wm;
      return;
}

/**********************************/
/*  signals generation            */
/**********************************/

/*-------------------------------------------------------
 * Package:     Wavelet Analysis Tool
 * File name:   AddSignals.C
 *
 * This macro file is for ROOT interactive environment.
 * Adds white noise with Gauss distribution and RMS=v. 
 *-------------------------------------------------------
*/

void AddGauss(WaveData &td, double sigma, double mean=0., int mgen=0, double u=1)
{
  int n=td.N;
  int m=0;
  double g;
  unsigned int r;
  unsigned int m4=0;
  m4 = (~m4)-1;

  TRandom rnd;
  rnd.SetSeed((unsigned int)(m4*gRandom->Rndm()));

  for (int i=0; i < n; i++){
     if(mgen == 0) td.data[i] += sigma*rnd.Gaus(0,1.)+mean;
     if(mgen == 1) td.data[i] += sigma*2*(rnd.Rndm()-0.5)+mean;
     if(mgen == 2) {
	td.data[i] += sigma*rnd.Gaus(0,1);
	g = rnd.Rndm()-0.5;
	td.data[i] += g>0 ? mean : -mean;
     }
     if(mgen == 3) {
	td.data[i] += sigma*rnd.Gaus(0,1)+mean*rnd.Poisson(u);
     }
     if(mgen == 4) td.data[i] += rnd.Landau(mean,sigma);
     if(mgen >= 5) td.data[i] += rnd.Poisson(mean);

     if(u>1. && (rnd.Rndm())<0.05 && mgen != 3){
	td.data[i] += sigma*u*rnd.Gaus(mean,1.);
//	td.data[i] += v*u*2*(rnd.Rndm()-0.5);
     }	
  }
}




































