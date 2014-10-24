wavearray<double> sg(double f, double s, int m=128)
{
   int nn = 1024*8;  
   wavearray<double> XX(nn);

   XX.rate(1024*8);
   double step = 1./XX.rate();
   double time;
   int shift = int(m*gRandom->Rndm(11));

//set f0l [ list   100.   153.   235.   361.   554.   850.  1304.  2000. ]
//set tdl [ list 0.0200 0.0130 0.0085 0.0055 0.0036 0.0024 0.0015 0.0010]


   for(int i=0; i<nn; i++){
      time = (i-nn/2+shift)*step;
      XX[i] = exp(-pow(time/s,2));
      XX[i] *= sin(2*PI*f*time);
   }
   XX*=1.;

   return wavearray<double>(XX);
}

wavearray<double> sgE(double f, double s)
{
   wavearray<double> a;
   wavearray<double> b;
   wavearray<double> c;
   Symlet<double> S60(58,0);      
   WSeries<double> z(S60); 
   double sum = 0.;
   double tot = 0.;
   int j,n,k,m;

   wavearray<double> w;
   w=wavefun(58,25,6,1);

   b = sg(f,s);
   b = 0;
   a = b;
   c = b;

   cout<<w.size()<<"  "<<b.size()<<endl;

   for(k=0; k<20; k++){

      m = int(128*gRandom->Rndm(11));

      a = 0.; bh(a,80.);
//      a = sg(f,s);
//      a = 0;
//      if(a.size()>=w.size()) a.cpf(w,w.size(),m,(a.size()-w.size())/2);
//      else                   a.cpf(w,a.size(),(w.size()-a.size())/2+m);
      cout<<k<<" "<<tot<<endl;

      z.Forward(a,6);
      a = z;
      a *= z;
      a.rank(0.);
      
      for(int i=0; i<z.size(); i++){
	 j = int(a.data[i]+0.5)-1;
	 b.data[j] = z.data[i]*z.data[i];
      }
      for(int i=0; i<z.size()-1; i++){
	 sum += b.data[i];
	 c.data[i+1] += sum;
      }
      tot += sum;
      sum=0;
   }

   c *= 100./tot;
   c.rate(1.);

   return c;
}

wavearray<double> sgK(double f, double s, double fE=0.8)
{
   int events = 1000;
   wavearray<double> a;
   wavearray<double> b;
   wavearray<double> c(events);
   Symlet<double> S60(58,1);      
   WSeries<double> z(S60); 
   double sum = 0.;
   double tot = 0.;
   int j,n,k,m;

   wavearray<double> w;
   w=wavefun(58,25,6,1);

   b = sg(f,s);
   b = 0;
   a = b;
   c = 0;

//   cout<<w.size()<<"  "<<b.size()<<endl;

   for(k=0; k<events; k++){

      m = int(128*gRandom->Rndm(11));

      a = 0.; bh(a,40.);
//      a = sg(f,s);
//      a = 0;
//      if(a.size()>=w.size()) a.cpf(w,w.size(),m,(a.size()-w.size())/2);
//      else                   a.cpf(w,a.size(),(w.size()-a.size())/2+m);
      cout<<k<<"  "<<m<<" "<<tot;

      z.Forward(a,6);
      a = z;
      a *= z;
      a.rank(0.);
      
      tot = 0.;
      for(int i=0; i<z.size(); i++){
	 j = int(a.data[i]+0.5)-1;
	 b.data[j] = z.data[i]*z.data[i];
	 tot += b.data[j];
      }
      for(int i=0; i<z.size()-1; i++){
	 sum += b.data[i];
	 if(sum/tot<fE) c.data[k] += 1;
      }
      sum=0;
      cout<<"  "<<c.data[k]<<endl;
   }

   c += 1;
   c.rate(1.);

   return c;
}

int bh(wavearray<double> &X, double M=20.)
{
   int nn = X.size();  
   wavearray<double> t;
   wavearray<double> hp;
   wavearray<double> hx;
   t=readAsciiD("macro/m20t.dat");
   t-=t.data[0];
   hp=readAsciiD("macro/m20hp.dat");
   hx=readAsciiD("macro/m20hx.dat");

   double step = 1./X.rate();
   double time,dt,tj;
   double a,b;
   double h,h1,h2;
   int j0 = 0;

   int shift = int(128*gRandom->Rndm(11));
   a = gRandom->Rndm(11); b=1.-a;

//set f0l [ list   100.   153.   235.   361.   554.   850.  1304.  2000. ]
//set tdl [ list 0.0200 0.0130 0.0085 0.0055 0.0036 0.0024 0.0015 0.0010]


   for(int i=nn/2+shift; i<nn; i++){
      time = (i-(nn/2+shift)+1)*step;
      if(j0>t.size()-2) break;

      for(int j=j0; j<t.size(); j++){
	 tj = t.data[j]*M/20.; 
	 if(time>tj) continue;
	 dt = tj-time;
	 h  = dt * (a*hp.data[j]+b*hx.data[j]);
	 if(j-1 >= 0) {
	    h += (time-t.data[j-1]*M/20.)*(a*hp.data[j-1]+b*hx.data[j-1]);
	    dt += time-t.data[j-1]*M/20.;
//	    cout<<h/dt<<"  "<<t.data[j-1]<<"  "<<time<<"  "<<t.data[j]<<endl;
	 }	       
	 X.data[i] += h/dt;
	 j0 = j==0 ? 0 : j-1;
	 break;
      }
   }

   return shift;
}













