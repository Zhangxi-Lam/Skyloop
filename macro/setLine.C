{   
   WaveData a;
   WaveData k;
   WaveData f;
   WaveData t;
   WaveData s;
   WaveData n;
   WaveData N;

   LineFilter F(1.);
   F.LoadTrend("H0-PEM-EY_V1-60.0689013137.lm");
   F.LoadTrend("../aug8-16.dat");
//   F.ReadBinary("/ligo/tmp3/sw/E2/trend/dec29-15.dat");
//   F.ReadBinary("/ligo/tmp3/sw/E2/trend/jan4-15.dat");
//   F.ReadBinary("/ligo/tmp3/sw/E2/trend/jan25lsc-32.dat");
//   F.ReadBinary("b.dat");


   a=F.getTrend(1,'a');
   f=F.getTrend(1,'f');
   s=F.getTrend(2,'s');
   n=F.getTrend(0,'n');
   N=F.getTrend(0,'N');
   t=F.getTrend(0,'t');
   k=F.getTrend(0,'K');
   p=F.getTrend(1,'p');

   wavereal avr, rms;
   double t0 = t.data[0];
   int ncount = 0;

   for(int j=0; j<5; j++){
      ncount = 0;
      N.getStatistics(avr,rms);
      for(int i=0; i<N.N; i++)
	 if(N.data[i] == 0.) ncount += 1;

      rms = (N.N*rms*rms - ncount*avr*avr)/(N.N-ncount);
      avr = N.N*avr/(N.N-ncount);
      rms = sqrt(rms);
      cout << "average="<<avr <<"  rms="<<rms<<"\n";
      
      for(int i=0; i<N.N; i++)
	 if(N.data[i] > avr+(j+3)*rms)
	    N.data[i] = 0.; 
   }


   for(int i=0; i<f.N; i++){ 
      t.data[i] -= t0;
      t.data[i] /= 3600.;
      
      if(N.data[i]>(avr+4*rms) || N.data[i] < avr/1000.){
	 a.data[i] = 0.;
	 f.data[i] = 0.;
	 k.data[i] = 0.;
	 n.data[i] = 0.;
	 s.data[i] = 0.;
	 N.data[i] = 0.;
	 p.data[i] = 0.;
      }
//      if(f.data[i]<0.) f.data[i] *= -1.;
   }

}







