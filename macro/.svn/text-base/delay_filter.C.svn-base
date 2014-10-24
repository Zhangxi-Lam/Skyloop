{
wavearray<double> x(1024*128);
addGauss(x,1,0);
x.rate(1024*4);

Meyer<double> S(512,2);
WSeries<double> w(x,S);
detector d("L1");

char fname[64];

for(int i=1; i<10; i++){
   sprintf(fname,"Meyer512_L%1d.dat",i);
   cout<<fname<<endl;
   w.Forward(1);
   d = w;
   d.setFilter(32);
   d.writeFilter(fname);
}

}
