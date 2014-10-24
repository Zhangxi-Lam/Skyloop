//
// This example show how to display a fits skymap (fitsName = fits file name)
// Author : Gabriele Vedovato

//#define SAVE_FITS 

void DrawFits(TString fitsName) {

  skymap sm(fitsName.Data());
  gskymap* gsm = new gskymap(sm);
  gsm->Draw();

#ifdef SAVE_FITS 
  wat::Time date("2009-07-09T07:06:19.99");
  gsm->Dump2fits("test.fits",date.GetGPS(),"1G un-modeled","PROB","pix-1",'C');
  //gsm->Dump2fits("test.fits.gz",date.GetGPS(),"1G un-modeled","PROB","pix-1",'C');
#endif
}
