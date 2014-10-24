//This File Reads Text Files and Places Them in a WAVEDATA Arr


void writeAscii(char *str, wavearray<float> &a)
{
   ifstream myOUTput;
   using namespace std;
   double x;
   int j;
   int n = a.size();
         
   myOUTput.open(str);
   if (!myInput) {
      cerr << "Unable to open that file!.!";
      exit(1);
   }
   
   for (j=0; j<n; j++){
      myOUTput << a.data[j] << "   "<<endl;
   }
   
         
   myOUTput.close();
}

void writeAscii(char *str, wavearray<float> &a, wavearray<float> &b )
{
   ifstream myOUTput;
   using namespace std;
   int j;
   int n = a.size()>b.size() ? b.size() : a.size();
         
   myOUTput.open(str);
   if (!myInput) {
      cerr << "Unable to open that file!.!";
      exit(1);
   }
   
   for (j=0; j<n; j++){
      myOUTput << a.data[j] << "   "<<b.data[j]<<endl) 
   }
   
         
   myOUTput.close();
}

