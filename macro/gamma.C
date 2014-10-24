
double cl(double X, int n)
{
   double sum = 1.;
   double x,y;

   x=X; y=X;
   for(int k=n-1; k>0; k--){
      sum += y; y*=x/(n-k+1);
   }
   return x-log(sum);

}

double rsnr(double &z, int n, double s=1.645)
{
   double sum = 1.;
   double x,y;
   double X = z-log(n*100.); 

   cout<<n<<"  snr="<<z<<"  d="<<X<<endl;
   if(X < 0) return 0.;

   for(int i=0; i<100; i++){
      x=X+log(sum); y=x; sum=1.;
      for(int k=n-1; k>0; k--){
	 sum += y; y*=x/(n-k+1);
      }
   }
   z = z*sqrt(3.)+s*s;
   X = ((X+log(sum))*sqrt(3)+s*s*n);
   return sqrt(z/X);

}


wavearray<double> cl(wavearray<double>* pX, int n, double xp=1.645)
{
   int nn = pX->size();  

   int i,j,k,m;
   double sum = 0.;
   double x,y;

//   double xp = 1.03;
//   double xp = 1.645;
//   double xp = 1.96;
//   double xp = 2.575;
//   double xp = 3.0;

   wavearray<double> x2(nn);

   for(int i=0; i<nn; i++){
      for(j=0; j<n; j++){
	 m = int(nn*gRandom->Rndm(11));
	 sum += pX->data[m]*pX->data[m] - xp*xp;
      }
      sum *= 1./sqrt(3)/0.98;

      x=sum; y=sum; sum=1.;
      for(k=n-1; k>0; k--){
	 sum += y; y*=x/(n-k+1);
      }
      x2.data[i] = x-log(sum);
	 sum=0.;
   }

   return x2;

}

wavearray<double> clw(wavearray<double>* pX, int n)
{
   int nn = pX->size();  
   Symlet<double> S60(58,1);      
   Biorthogonal<double> B60(60,1);      
   WSeries<double> w; 
   WSeries<double> z(*pX,S60); 

   int m = 0;
   int k = 0;
   int j = 0;
   int l = 0;
   double sum = 0.;
   double x,y;

//   double xp = 1.03;
   double xp = 1.645;
//   double xp = 1.96;
//   double xp = 2.575;
//   double xp = 3.0;

   int N = n>0 ? n : -n;

   wavearray<double> x2(int(nn/N));
   wavearray<double> yy;

   z.Forward(6);
   z.getLayer(yy,0); yy=0; z.putLayer(yy,0);
   z.white(15.);
   w.percentile(0.1,1,&z); 

   for( ; ; ){
      l = int(nn*gRandom->Rndm(11));
      if(fabs(w[l])<=0.) continue;
      if(n>0) sum += z[l]*z[l]/sqrt(3.);
      else    sum += log(fabs(w[l]));   

      if(++j==N) {
	 x=sum; y=sum; sum=1.; j=0;
	 for(k=N-1; k>0; k--){
	    sum += y; y*=x/(N-k+1);
	 }
	 x2.data[m++] = x-log(sum);
	 sum=0.;
      }
      if(m>10000) break;
   }
   x2.resize(m);
   return x2;

}








