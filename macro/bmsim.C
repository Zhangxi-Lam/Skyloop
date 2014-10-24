{
int k = 200;
wavearray<double> g(400);
int n,m,l;
double X = 1.;
double a = -5.;
double v = 1.03;
double x, y;
double s,sn,sm,s2,;


for(int j=0; j<400; j++){

  x = 1.+(gRandom->Rndm(11)-0.5)*0.8;
  //  x = 1.;
  n=0; m=0; s=0; sn=0.; sm=0.; s2=0;

  for(int i=0; i<k; i++){
    s  += x;
    s2 += x*x;
    //    g.data[i] = x; 
    y = 1./(1+pow(x/X,a));
    if(y<0.2) y=0.2;
    l=gRandom->Binomial(1,y);
    if(l) { sn+=x; n++; x /= n>=-2 ? v : 1.;}
    else  { sm+=x; n--; x *= n<0 ? v : 1.;}
    //    cout<<x<<" "<<v<<" "<<n<<endl;
  }
  s  /= k;
  s2 /= k;
  sn /= k;
  sm /= k;

  //  g.data[j] = ((n-m)*s2+k*s*(sm-sn))/((n-m)*s+k*(sm-sn)); 
  //  g.data[j] = 0.5*((n-m)*s+k*(sm-sn))/(k*s*s-k*s2); 
  g.data[j] = s; 
}
}


