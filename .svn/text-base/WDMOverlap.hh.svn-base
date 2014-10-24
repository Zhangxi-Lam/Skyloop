#ifndef WDMOVERLAP_HH
#define WDMOVERLAP_HH

#include "WDM.hh"
#include "netcluster.hh"
#include <algorithm>
#include "stdint.h"

using namespace std;

struct overlaps{int32_t index;  float ovlpAA, ovlpAQ, ovlpQA, ovlpQQ;}; 
struct ovlArray{struct overlaps* data; int size;};

template <class T>
class WDMOverlap{
public:
   
   // default constructor
   WDMOverlap();
   
   // constructor that builds the catalog
   //! param: vector of pointers to WDM objects
   //! param: number of pointers (resolutions)
   //! param: threshold on overlap values
   WDMOverlap(WDM<T>** wdm, int nRes, double minOvlp = 0.01);
   
   // constructor that reads the catalog from a file
   //! param: filename
   WDMOverlap(char* filename);
   
   // copy constructor
   //! param: other WDMOverlap object
   WDMOverlap(const WDMOverlap<T>& x);
   
   // destructor
   virtual ~WDMOverlap();
   
   // performs memory deallocation
   void deallocate();
   
   // write to file
   //! param: filename
   void write(char* filename);
   
   // read from file
   //! param: filename
   void read(char* filename);
   
   // access function that returns all 4 overlap values between two pixels
   //! param: defines resolution 1 (by number of layers)
   //! param: defines pixel 1 at resolution 1
   //! param: defines resolution 2 (by number of layers)
   //! param: defines pixel 2 at resolution 2
   // returns a struct overlap containing the four possible combinations
   // amplitude-amplitude, amplitude-quadrature, quadrature-amplitude, quad-quad
   struct overlaps getOverlap(int nLay1, size_t indx1, int nLay2, size_t indx2);
   
   // access function that returns one overlap value between two pixels
   //! param: defines resolution 1 (by number of layers)
   //! param: defines whether it's amplitude (0) or quadrature (1) for resolution 1
   //! param: defines pixel 1 at resolution 1
   //! param: defines whether it's amplitude (0) or quadrature (1) for resolution 2
   //! param: defines resolution 2 (by number of layers)
   //! param: defines pixel 2 at resolution 2
   // returns the overlap value
   float getOverlap(int nLay1, int quad1, size_t indx1, int nLay2, int quad2, size_t indx2);  //quad: 0/1, 1 for quadrature
   
   // FILL cluster overlap amplitudes
   //! param: pointer to netcluster structure
   //! param: which cluster to process
   //! param: number of pixels to process 
   //! param: address where to store the values (format: vector<vector<struct overlap> > )
   void getClusterOverlaps(netcluster* pwc, int clIndex, int nPix, void* q);
       
   //void PrintSums();
     
//protected:
   struct ovlArray (***catalog)[2];     // stores overlap values [r1][r2][r1_freq][parity] ; r2<=r1
   int nRes;                            // number of resolutions
   int* layers;                         // M for each resolution
   
//   ClassDef(WDMOverlap,1)
 
};

#endif
