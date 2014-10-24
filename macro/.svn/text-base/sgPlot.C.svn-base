{
//set f0l [ list   100.   153.   235.   361.   554.   850.  1304.  2000. ]
//set tdl [ list 0.0200 0.0130 0.0085 0.0055 0.0036 0.0024 0.0015 0.0010]
double f=850.;
double g=0.1;


gROOT->LoadMacro("macro/sg.C"); 

TCanvas *c1 = new TCanvas("c","C",0,0,600,600);
c1->SetBorderMode(0);
c1->SetFillColor(0);
//c1->Divide(3,2);

Biorthogonal<double> B16(16,1);
Symlet<double> S16(16,1);      
Daubechies<double> D60(58,1);
Biorthogonal<double> B32(64,1);
Symlet<double> S60(58,1);      
Symlet<double> S32(32,1);      
Meyer<double> M32(1);      
Daubechies<double> D32(32,1);


int m = 1024*8;
int nn = 1024*8*16;  
wavearray<double> X(nn);
wavearray<double> a;
WSeries<double> w; 
X=0.;
X.rate(double(m));
a.rate(double(m));

a=sg(f,g);
X.cpf(a,m,0,m*7);
AddGauss(X,100);

cout<<X.size()<<endl;

/*
WSeries<double> z(S16); 
c1->cd(1); z.Forward(X,6); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1); 
c1->cd(2); z.Forward(X,8); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1);
WSeries<double> z(a,D60); 
c1->cd(3); z.Forward(X,6); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1); 
c1->cd(4); z.Forward(X,8); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1);
WSeries<double> z(a,B16); 
c1->cd(5); z.Forward(X,6); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1); 
c1->cd(6); z.Forward(X,8); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1);

c1->Update();
c1->SaveAs("sg850_16.gif");
*/

WSeries<double> z(S60); 
c1->cd(4); z.Forward(X,6); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1); 
//c1->cd(4); z.Forward(X,8); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1);
//WSeries<double> z(a,D60); 
//c1->cd(5); z.Forward(X,6); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1); 
//c1->cd(5); z.Forward(X,8); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1);
//WSeries<double> z(a,B32); 
//c1->cd(6); z.Forward(X,6); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1); 
//c1->cd(6); z.Forward(X,8); w.percentile(z,0.1,1); w=z[slice(7*m,m,1)]; WTSpectrum(w,0,1);

c1->Update();
c1->SaveAs("sg850_100.gif");

}






