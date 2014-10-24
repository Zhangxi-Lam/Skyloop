//
// this example draw to to 2d hist the spherical harmonic coefficients
// Author : Gabriele Vedovato
// fitsName = fits file name

void DrawAlmFromFits(TString fitsName) {

  #include <complex>

  skymap sm(fitsName.Data());
  int L = sm.size();
  cout << "L " << L << endl;
  cout << "mean " << sm.mean() << endl;

  double en=0;
  for(int i=0;i<L;i++) en+=pow(sm.get(i),2);

  double dw = 1./L;
  cout << "EN " << en*dw << endl;

  wat::Alm alm = sm.getAlm(256);
  cout << "alm(0,0).real()/sqrt(4*TMath::Pi()) : " << alm(0,0).real()/sqrt(4*TMath::Pi()) << endl;
  double norm=0;
  for(int l=0;l<=alm.Lmax();l++) {
    int limit = TMath::Min(l,alm.Mmax());
    for (int m=0; m<=limit; m++) {
      double mod = pow(alm(l,m).real(),2)+pow(alm(l,m).imag(),2);
      norm+= m==0 ? mod : 2*mod;
    }
  }
  norm = norm/(4*TMath::Pi());
  cout << "norm : " << norm << " = en " << en*dw << " from Parseval Formula" << endl;

  TCanvas* canvas = new TCanvas("Alm", "LVC experiment", 300,40, 600, 600);

  TH2D* h2 = new TH2D("alm","alm", alm.Lmax(), 0, alm.Lmax(), alm.Lmax(), 0, alm.Lmax());
  h2->SetStats(kFALSE);
  h2->SetTitleFont(12);
  h2->SetFillColor(kWhite);
  h2->GetXaxis()->SetTitle("l");
  h2->GetYaxis()->SetTitle("m");

  double min=+100;
  double max=-100;
  for(int l=1;l<=alm.Lmax();l++) {  // l=0 is excluded !!!
    int limit = TMath::Min(l,alm.Mmax());
    for (int m=0; m<=limit; m++) {
      double mod = pow(alm(l,m).real(),2)+pow(alm(l,m).imag(),2);
      h2->SetBinContent(l[i],m[i],log10(mod));
      if(max<log10(mod)) max=log10(mod);
      if(min>log10(mod)) min=log10(mod);
      //h2->SetBinContent(m[i],l[i],log10(mod));
    }
  }
  cout << "min " << min << " max " << max << endl;
  h2->GetZaxis()->SetRangeUser(min,max);

  h2->Draw("colz");

}
