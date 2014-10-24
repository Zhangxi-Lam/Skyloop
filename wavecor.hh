// Wavelet Analysis Tool  
// Sergey Klimenko, University of Florida
// universal data container for x-correlation analysis
// used with DMT and ROOT
//

#ifndef WAVECOR_HH
#define WAVECOR_HH

#include <iostream>
#include "wavearray.hh"
#include <vector>
#include <list>

typedef std::vector<int> vector_int;


class wavecor
{
  public:
      
      // constructors
      
      //: Default constructor
      wavecor();
    
      //: Copy constructor
      //!param: value - object to copy from 
      wavecor(const wavecor&);
      
      //: destructor
      virtual ~wavecor();
    
      // operators

      wavecor& operator= (const wavecor&);

      // accessors

      //: kendall x-correlation for two wavearrays; 
      //!param: two wavearrays, integration window, interval for time lag analysis
      //!       and skip parameter for running integration window
      //!       if skip=0 window is shifted by one sample
      //!put x-correlation in this
      virtual void kendall(wavearray<double>&, wavearray<double>&,
			  double, double, size_t=0);

      //: initialize wavecor class from two wavearrays; 
      //!param: two wavearrays, integration window, interval for time lag analysis
      //!       and skip parameter for running integration window
      //!       if skip=0 window is shifted by one sample
      //!put x-correlation in this
      virtual void init(wavearray<double>&, wavearray<double>&,
			double, double, size_t=0);

      //:function for thresholding of x-correlation samples
      //:A sample of x-correlation statistics is set to zero if module
      //:of x-correlation is below a threshold defined by input parameter.
      //:returns fraction of selected samples 
      virtual double select(double);

      //:coincidence with x-correlation sample defined by par2
      //:coincidence window is defined by par1 
      //:sample in this is set to 0. if there are now non-zero samples 
      //:form input x-correlation sample in the window defined by par1
      //!param: coincidence window,  pointer to wavecor object
      //:returns fraction of selected samples 
      virtual double coincidence(double, wavecor*);

// data members

//#ifndef __CINT__
      float shift;     // time shift
      int ifo;         // detector index: 1/2/3 - L1H1/H1H2/H2L1
      int run;         // run ID
      double window;   // integration window in seconds
      double lagint;   // lag interval in seconds

      //: x-correlaton
      wavearray<float> xcor;
      //: time delays
      wavearray<float> xlag;
      //: cluster list
      std::list<vector_int> cList;
//#endif

      // used by THtml doc
      ClassDef(wavecor,1)	

}; // class wavecor
	

#endif // WAVECOR_HH
