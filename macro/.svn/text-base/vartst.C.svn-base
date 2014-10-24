{
char* ifo  = "H2";
char name[512];
char file[512];
char flal[512];
sprintf(name,"%s:LSC-AS_Q",ifo);

//double start = 730891952.;
//double skip  = 0.;
//sprintf(file,"/cdf/tmp1/ligosoft/sazonov/work2004/waveform/frames/HL-AS_Q1-%d-16.gwf",int(start));

double start = 732631072.; // event 2: 732631083
double skip  = 10.;
wavearray<float>* px;
wavearray<float>* py;
sprintf(file,"/cdf/tmp1/ligosoft/sazonov/work2004/waveform/frames/HL-AS_Q1-%d-16.gwf",int(start));
px=ReadFrFile(122.,skip,name,file);
py=ReadFrFile(122.,0.,name,file);

int M = 1024*8*int(skip+0.5);

Biorthogonal<double> B(8);
wavearray<double> xx;

wavearray<double> x(px->size());
wavearray<double> y(py->size());
wavearray<double> X,Y;
wavearray<double> z(1024*8*120);
z.rate(1024*8); z=0.;

for(int i=0; i<px->size(); i++){
  x.data[i] = px->data[i];
  y.data[i] = py->data[i];
}
x.rate(px->rate());
y.rate(py->rate());

Biorthogonal<double> B(30);
Symlet<double> S(60,1);

WSeries<double> wx(x,B);
wx.Forward(1);
wx.getLayer(X,0);
WSeries<double> wy(y,B);
wy.Forward(1);
wy.getLayer(Y,0);


WSeries<double> vx(X,S);
WSeries<double> vy(Y,S);
WSeries<double> r;
wavearray<double> varx;
wavearray<double> vary;
WSeries<double> WX, WY;

vx.Forward(9);
r=vx.white(122.);
r.rate(1);
varx=vx.hammer(0.5);
WSeries<double> warx(varx,B); warx.Forward(3);
warx.getLayer(xx,0); warx=0; warx.putLayer(xx,0); 
warx.Inverse(); warx-=varx; warx*=-1.; warx+=1.;
varx=vx.hammer(0);
vx.Inverse(1);

vy.Forward(9);
r=vy.white(122.);
r.rate(1);
vary=vy.hammer(0.5);
WSeries<double> wary(vary,B); wary.Forward(3);
wary.getLayer(xx,0); wary=0; wary.putLayer(xx,0); 
wary.Inverse(); wary-=vary; wary*=-1.; wary+=1.;
vary=vy.hammer(0);
vy.Inverse(1);

WX=vx; WX.resize(1024*8*120);
WX.setWavelet(*(vx.pWavelet));
WX.start(0.);
z.cpf(vx,1024*8*120,1024*8,0);
WX=z;

WY=vy; WY.resize(1024*8*120);
WY.setWavelet(*(vy.pWavelet));
WY.start(0.);
z.cpf(vy,1024*8*120,1024*8,0);
WY=z;

double a,b,c;

WX.pWavelet->m_Level=8;
Wy.pWavelet->m_Level=8;

WX.percentile(0.1,1);
WY.percentile(0.1,1);

WX.pWavelet->m_Level=8;
Wy.pWavelet->m_Level=8;

/*
for(int i=0; i<WX.size(); i++) {
  if(i>=WX.size()-M) WX.data[i] = 0.;
  if(i<M) WY.data[i] = 0.;
  a = WX.data[i];
  b = i+M<WY.size() ? WY.data[i+M] : 0.;
  if(a!=0.) {
    WX.data[i] = a>0 ? log(fabs(a)) : -log(fabs(a));
    if(b!=0.) {
      WX.data[i] -= b>0 ? log(fabs(b)) : -log(fabs(b));
    }
    else WX.data[i] = 0.;
  }
}
*/

TCanvas *c1 = new TCanvas("c","C",0,0,800,600);
c1->SetBorderMode(0);
c1->SetFillColor(0);
gPad->SetBorderMode(0);
gPad->SetFillColor(0);

}








