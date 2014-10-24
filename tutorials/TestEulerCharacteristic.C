//
// Test Euler characteristic
// Author : Gabriele Vedovato

#define COORDINATES "cWB"
#define PROJECTION ""

{
  // create healpix skymap order=3
  skymap sm(int(3));

  wavearray<int> index;

  // patch with 9 pixels
  sm.set(30,10); 
  // get the pixel numbers of the SW, W, NW, N, NE, E, SE and S neighbor pixel=30
  index = sm.neighbors(30);
  for(int i=0;i<index.size();i++) sm.set(index[i],20+5*(i%4)); 

  // patch with 1 pixel
  sm.set(80,10); 

  // patch with hole
  // get the pixel numbers of the SW, W, NW, N, NE, E, SE and S neighbor pixel=200
  index = sm.neighbors(200);
  for(int i=0;i<index.size();i++) sm.set(index[i],20+5*(i%4)); 

  // get number of rings 
  int nrings = sm.getRings();

  // get start ring pixel ring=nrings/2
  int startpix = sm.getStartRingPixel(nrings/2);
  // get number of pixels in the ring=nrings/2
  int npixels = sm.getRingPixels(nrings/2);
  // fill npixels/2 in the ring=nrings/2 
  for(int i=0;i<npixels/2;i++) sm.set(startpix+i,20); 
  // fill npixels2/2 in the ring=nrings/2+1 
  int startpix2 = sm.getStartRingPixel(nrings/2+1);
  int npixels2 = sm.getRingPixels(nrings/2+1);
  for(int i=0;i<npixels2/2;i++) sm.set(startpix2+i,30); 

  int EC = sm.getEulerCharacteristic(0.5);

  cout << "Euler characteristic : " << EC << endl;

  // draw skymap
  gskymap* gSM = new gskymap(sm);
  gSM->SetOptions(PROJECTION,COORDINATES);
  gSM->Draw();

  // dump skymap to fits
  // used aladin to display fits
  // -> aladin euler_characteristic.fits
  gSM->Dump2fits("euler_characteristic.fits");

}
