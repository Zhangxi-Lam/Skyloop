// Wavelet Analysis Tool
//--------------------------------------------------------------------
// Implementation of 
// Daubeches wavelets using Fast Wavelet Transform 
// References:
//   I.Daubechies, Ten lectures on wavelets
//   ISBN 0-89871-274-2, 1992
//--------------------------------------------------------------------

//$Id: Daubechies.hh,v 0.2 2001/08/06 19:37:00 klimenko Exp $
#ifndef DAUBECHIES_HH
#define DAUBECHIES_HH

#include "WaveDWT.hh"

//namespace datacondAPI {
//namespace wat {

template<class DataType_t>
class Daubechies : public WaveDWT<DataType_t>
{
   private:

      //: forward LP filter coefficients.
      double *pLForward;	//!
      //: inverse LP filter coefficients.
      double *pLInverse;	//!
      //: forward LP filter coefficients.
      double *pHForward;	//!
      //: inverse LP filter coefficients.
      double *pHInverse;	//!

      void setFilter();

   public:
      
      //: construct from base class
      Daubechies(const Wavelet &);

      //: copy constructors
      Daubechies(const Daubechies<DataType_t> &);

      //: construct from wavelet parameters
      Daubechies(int order=4, int tree=0, enum BORDER border=B_CYCLE);

      //: destructor
      virtual ~Daubechies();

      //: Duplicate on heap
      virtual Daubechies* Clone() const;

      //: decomposition method
      virtual void forward(int level, int layer);
      //: reconstruction method      
      virtual void inverse(int level, int layer);

      ClassDef(Daubechies,1)

}; // class Daubechies

//}; // namespace wat
//}; // namespace datacondAPI

#endif // DAUBECHIES_HH












