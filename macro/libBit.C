/* bit operation library */


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
   if(n<=0 || n>a.N) n=1;
//   cout<<n<<endl;
   long ns;
   int i,j;
   double r,r2=0.;
   for(i=0; i<=(a.N-n); i+=n){
      minus=0;
      for(j=0; j<n; j++){
	 minus += bm.data[long(a.data[i+j])];
      }
      r = minus/16./n;
      r2 += r*r;
   }
   return n*r2/(a.N/n);
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
double median(WaveData &a)
{
   double avr,rms;
   int n1=0;
   int n0=0;
   int i;
   a.getStatistics(avr,rms);
   double x = avr+0.01*rms;
//   cout << x <<"  "<< avr<<endl;

   for(i=0; i<a.N; i++){
      if(a.data[i]<x) n1--;
      else n1++;
      if(a.data[i]<avr) n0--;
      else n0++;
   }
//   cout << n0 <<"  "<< n1<<endl;
   avr += n0*0.01*rms/(n0-n1);
   return double(n0);
}











