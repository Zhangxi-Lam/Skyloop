{
int i;
int N = 50;
double T=70.;
NET.initwc(T,0.1);
NET.setRMS();
NET.likelihood3('L',false,0.,1,0);
skymap sm = NET.nLikelihood;

for(i=1; i<N; i++){
  NET.initwc(T+i*0.1,0.1);
  NET.setRMS();
  NET.likelihood3('L',false,0.,1,0);
  sm += NET.nLikelihood;
}

sm *= 1./N;

}
