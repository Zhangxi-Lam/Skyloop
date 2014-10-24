/********************************************************/
/* Wavelet Analysis Tool                                */
/* file readfrfile.hh                                    */
/********************************************************/

//#ifndef _READFRFILE_H
//  #define _READFRFILE_H

#ifndef _STRING_H
  #include <string.h>
#endif
#include "waverdc.hh"
#include "lossy.hh"
#include "FrameL.h"

bool 
ReadFrFile(wavearray<double> &out, 
	   double tlen, 
	   double tskip, 
	   char *cname, 
	   char *fname,
	   bool seek=true, 
	   char *channel_type="adc");

wavearray<float>* 
ReadFrFile(double tlen, 
	   double tskip, 
	   char *cname, 
	   char *fname,
	   bool seek=true, 
	   char *channel_type="adc");

//#endif // _READFRFILE_H
