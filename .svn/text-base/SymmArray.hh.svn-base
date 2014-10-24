#ifndef SYMMARRAY_HH
#define SYMMARRAY_HH

#include "stdio.h"
#include "TNamed.h"

// guaranteed to work only with (struct of) atomic types

// a[-n], a[-n+1] a[-n+2] ... a[0] a[1] .... a[n]

template <class Record>
class SymmArray : public TNamed {
public:
   SymmArray(unsigned int n=0);
   SymmArray(const SymmArray&);       //copy constructor
   virtual ~SymmArray();
   SymmArray& operator=(const SymmArray& other);
   void Init(Record x);
   void Resize(int sz);
   void Write(FILE* f);
   void Read(FILE* f);
   Record& operator[](int i){ return zero[i];}
   int Last() {return Size/2;}
   
   
protected:
   void Resize0(int sz);
   int Size;
   Record* rec;		//!
   Record* zero;	//!
   int recSize;

   ClassDef(SymmArray,1)
};

#endif 
