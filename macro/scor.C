void scor(wavearray<float> &w, wavearray<float> &a, int n){

wavearray<short> s(w.size());
wavearray<short> *ps = &s;

 for(int i=0; i<s.size(); i++) s.data[i] = short(w.data[i]);

 int pp=0;
 int na = a.size();
 int nn;
 int N = s.size();
 short sign;

 an.data[0]=1.;

 for(int j=1; j<na && (N-j*n)>j*n; j++){
    pp = 0;
    nn = 0;
    for(int i=0; i<N-j*n; i++){
       sign = s.data[i]*s.data[i+j*n];
       if(sign){
	  pp += sign;
	  nn++;
       }
    }
    an.data[j] = float(pp)/float(nn);
 }
 an.rate(ss.rate()/8.);
}
