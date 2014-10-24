{
int n = 1024*32;
int level = 5;
int delay = 9;
double av,rm;

Biorthogonal<double> B(30,1);
Daubechies<double>   D(30,1);

wavearray<double> g(n);
wavearray<double> G(n);
wavearray<double> x(n);
wavearray<double> y(n);

wavearray<double> ax;
wavearray<double> ay;
wavearray<double> w;
wavearray<double> c;
wavearray<double> f;
wavearray<double> F;
wavearray<double> C(2*delay+1);
wavearray<double> V(2*delay+1);
int k;

WSeries<double> Wx(x,D);
WSeries<double> Wy(y,D);

wavearray<double> dd(100);

//for(int mm=0; mm<100; mm++){

x = 0.;
y = 0.;
g = 0.;

AddGauss(x,1.);
AddGauss(y,1.);
AddGauss(g,0.5);

G.cpf(g,g.size()-5,5);

x += G;
y += g;

Wx.Forward(x,level);
Wy.Forward(y,level);

C = 0.;

 for(int i=1; i<(1<<level); i++){
// for(int i=1; i<16; i++){
    Wx.getLayer(ax,i);
    ax.getStatistics(av,rm);
//    ax *= 1./rm;
    Wy.getLayer(ay,i);
    ay.getStatistics(av,rm);
//    ay *= 1./rm;
    c = lagcor(ax,ay,5);
    f = c;
    w = wavefun(i,level,1);
//    cout<<"i="<<i<<" "<<c.data[5]<<endl;
    k = int(w.rate()+0.5); 

    for(int j=-delay; j<=delay; j++){
       f.data[5] = w[abs(j)];
//       f.data[5] = 0;

       for(int m=1; m<=5; m++){
	  f.data[5-m] = w[m*k+j];
	  f.data[5+m] = w[m*k-j];
       }
       F = f;
       F *= f;
       f *= 1./(F.mean()*F.size());
       F = f;
//       F *= f;
//       cout<<"i="<<i<<"  j="<<j<<"  f="<<F.mean()*F.size()<<endl;
       f *= c;
       C.data[j+delay] += f.mean() * f.size();
    }
 }

 C*=1./((1<<level)-1);

// dd.data[mm] = C[4];

V = lagcor(x,y,delay);

//}
}










