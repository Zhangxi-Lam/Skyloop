{
sprintf(filtername,"/home/klimenko/wat/wat-4.3.0/data/Meyer1024_L5.dat");
wavearray<double> x(32*1024);
x.rate(4096); 
AddGauss(x,0,1);
Meyer<double> M(1024,1);
WSeries<double> w(M);

w.Forward(x,5);

detector L1("L1");


double a, e, L, c, u, v;
double T = sqrt(2);
c = 0.1;
for(int j=1; j<=100; j++){
  e = j*0.01;
/*
  a = e/(e+c*(1-e));
  L = (2*e+c*(1-e))/2./(e+c*(1-e));
  x.data[j-1] =intL(T,a,1000)-intL(L,a,1000);
  u = T;
  v = log(1+(u+3.14)*pow((1-a),3./4)/2);
  y.data[j-1] =u+log(1.+v*v)-1; 
  v = log(1+(u+3.14)*(1-a)/2);
  z.data[j-1] =u+log(1.+v*v)-1; 
*/

  x.data[j-1] = sqrt(1.-e);
  y.data[j-1] = 1./T - T*(e-0.5)/2.-pow(2,3./2)*pow(e-0.5,2)/4. - pow(2,5./2)*pow(e-0.5,3)/24.;
  y.data[j-1]+= -pow(2,7./2)*pow(e-0.5,4)*5./16/2./3./4.;

}
Plot(x,0,1);
//Plot(z,1,4);
Plot(y,1,2);
}
