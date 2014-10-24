wavearray<double> wavefun(int layer, int level, int tree=0)
{
   int fLength=30.;
   int nn = (1<<level)*32;  
   wavearray<double> Y(nn/2);
   wavearray<double> X(nn);
   wavearray<double> a;

   X.rate((1<<level));
   Y.rate((1<<level));

   Daubechies<double> Db(30,tree);
   Biorthogonal<double> Bb(30,tree);
   WSeries<double> W(X,Db);

   W.Forward(level);
   W.getLayer(a,layer);
   W=0.;
   a=0.;
   a.data[a.size()/2]=1.;
   W.putLayer(a,layer);
   W.Inverse();

   int nn2 = W.size()/2;
   double x, y, z;
   W.FFT();

// calculate power spectrum from Fourier coefficients
// without f=0;
//cout<<"nn="<<nn<<" Rate="<<sp.rate()<<endl; 

   x = W.data[0];
   W.data[0] *= x;
   x = W.data[1];
   W.data[1] *= x;
    
   for (int i = 1; i < nn2; i++) {
      x = W.data[2*i];
      y = W.data[2*i + 1];
      z = (x*x + y*y);
      W.data[2*i] = z;
      W.data[2*i+1] = 0.;
   }
   
   W.FFT(-1);
   X = W;
   Y.cpf(X,Y.size());
   x = Y[0];
   Y *= 1/x;
   return Y;
}








