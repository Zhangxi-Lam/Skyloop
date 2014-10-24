wavearray<double> wavefun(int order, int layer, int level, int tree=0)
{
   int fLength=30.;
   int nn = (1<<level)*256;  
   wavearray<double> X(nn);
   wavearray<double> a;

   X.rate(1024*8);

   Daubechies<double> Db(order,tree);
   Symlet<double> Sb(order,tree);
   Haar<double> Hb(tree);
   Meyer<double> Mb(tree);
   Biorthogonal<double> Bb(order,tree);

   WSeries<double> W(X,Sb);

   W.Forward(level);
   int ll = W.maxLayer();
   if(ll<layer) { cout<<"max layer = "<<ll<<endl; layer = ll; }
   W.getLayer(a,layer);
   W=0.;
   a=0.;
   a.data[a.size()/2]=1.;
   W.putLayer(a,layer);
   W.Inverse();

   int m = int(2*(1<<level)*gRandom->Rndm(11));

   return wavearray<double>(W);
}








