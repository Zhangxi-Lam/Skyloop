{
wavearray<double> x(1024*128);
wavearray<double> y;
addGauss(x,1,0);
double r = 4096;
double t = 16.1/r;
x.rate(1024*4);

Meyer<double> S(512,2);
WSeries<double> w(x,S);
detector d("L1");
w.Forward(5);
d = w;
//d.TFmap.cpf(d.TFmap,d.TFmap.size()-32,32);
//d.setFilter(64);
d.readFilter("/home/klimenko/wat/wat-4.0.3/data/Meyer512_L5.dat"); 
d.delay(t,w);
d.getTFmap()->Inverse();
w.Inverse();
y = d.TFmap;
y.cpf(d.TFmap,d.TFmap.size()-int(t*r),int(t*r));
}
