{
double occ;
wavearray<double> a;
wavearray<double> x(1024*8*60);
x.rate(1024*8);
wavearray<double> y(1024*8*60);
y.rate(1024*8);
x=0.;
y=0.;
AddGauss(x,1,0);
AddGauss(y,1,0);

Symlet<double> B(32,1);
WSeries<double> wx(x,B);
WSeries<double> wy(y,B);
WSeries<double> w;

wx.Forward(4);
wy.Forward(4);
wx.percentile(0.31,1);
wy.percentile(0.31,1);
w=wx;
occ = w.Coincidence(wy,0.01,1.5);
}
