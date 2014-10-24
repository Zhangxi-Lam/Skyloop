#ifndef SYMMARRAYSSE_HH
#define SYMMARRAYSSE_HH

#include "stdio.h"
#include "TNamed.h"

// meant to be used with int32, int64, float, double

// a[-n], a[-n+1] a[-n+2] ... a[0] a[1] .... a[n]

template <class Record>
class SymmArraySSE : public TNamed{
public:
   SymmArraySSE(unsigned int n=0);
   SymmArraySSE(const SymmArraySSE&);       //copy constructor
   virtual ~SymmArraySSE();
   SymmArraySSE& operator=(const SymmArraySSE& other);
   void Init(Record x);
   void Resize(int nn); // new n
   void Write(FILE* f);
   void Read(FILE* f);
   Record& operator[](int i){ return zero[i];}
   Record* SSEPointer(){   return rec;}  
   int SSESize(){ return SizeSSE;}    
   int Last() {return last;}
   void ZeroExtraElements();
   
protected:
   void allocateSSE(); // aligned allocation; uses SizeSSE, last; sets rec, zero
   int last, SizeSSE;   // SizeSSE in bytes (multiple of 8)
   Record* rec;      //!
   Record* zero;     //! always in the middle of the allocated space
   int recSize;
   
   ClassDef(SymmArraySSE,1)
};



#endif 
