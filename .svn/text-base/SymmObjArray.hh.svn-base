#ifndef SYMMOBJARRAY_HH
#define SYMMOBJARRAY_HH

#include "stdio.h"
#include "TNamed.h"

// guaranteed to work only with classes that implement the "persistent" interface (Read/Write)

// a[-n], a[-n+1] a[-n+2] ... a[0] a[1] .... a[n]

template <class T>
class SymmObjArray : public TNamed {
public:
   explicit SymmObjArray(unsigned int n=0);
   explicit SymmObjArray(const SymmObjArray&);       //copy constructor
   virtual ~SymmObjArray();
   SymmObjArray& operator=(const SymmObjArray& other);
   void Resize(unsigned int sz);                      // data is lost
   void Write(FILE* f);
   void Read(FILE* f);
   T& operator[](int i){ return zero[i];}
   unsigned int Last() {return Size/2;}
   
protected:
   void Resize0(unsigned int sz);
   int Size;
   T* rec;	//!
   T* zero;	//!

   ClassDef(SymmObjArray,1)
};


#endif 
