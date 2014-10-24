//This File Reads Text Files and Places Them in a WAVEDATA Arr


wavearray<float>* readAscii(char *str,int lines)
{
   ifstream myInput;
   wavearray<float> *arr=new wavearray<float>(lines);
   using namespace std;
   double x;
   char* entry;
   int j=0;
         
   myInput.open(str);
   if (!myInput) {
      cerr << "Unable to open that file!.!";
      exit(1);
   }

   while (myInput >> x )  lines++;  // count lines
   myInput.close();

   if(!lines) exit(1);

   myInput.open(str);

   while (myInput >> x ) 
   {
      arr->data[j]= x;
      j++;
      if (j>lines)break;
   }
   
         
   myInput.close();
  return arr; 
}
