//This File Reads Text Files and Places Them in a WAVEDATA Arr


wavearray<float>* readAsciiF(char *str, int lines=0)
{
   ifstream myInput;
   using namespace std;
   double x;
   int j=0;
         
   myInput.open(str);
   if (!myInput) {
      cerr << "Unable to open that file! "<<str<<endl;
      exit(1);
   }
   
   if(!lines){
    while (myInput >> x ) lines++;
    myInput.close();
    myInput.open(str);
   }

   wavearray<float> *arr=new wavearray<float>(lines);

   while (myInput >> x ) 
   {
      arr->data[j] = x;
      j++;
      if (j>lines)break;
   }
   
         
   myInput.close();
  return arr; 
}

wavearray<double>* readAsciiID(char *str, int lines=0)
{
   ifstream myInput;
   using namespace std;
   double x;
   int ind;
   int j=0;
         
   myInput.open(str);
   if (!myInput) {
      cerr << "Unable to open that file! "<<str<<endl;
      exit(1);
   }
   
   if(!lines){
     while (myInput >> ind >> x ) lines++;
     myInput.close();
     myInput.open(str);
   }

   wavearray<double> *arr=new wavearray<double>(lines);

   while (myInput >> ind >> x ) 
   {
      arr->data[j] = x;
      j++;
      if(j>lines) break;
   }
   
         
   myInput.close();
  return arr; 
}

wavearray<double>* readAscii(char *str, int n)
{
   FILE* fp;
   double x;
   int j=0;
         
   fp=fopen(str,"r");
   if (!fp) {
      cerr << "Unable to open that file! "<<str<<endl;
      exit(1);
   }
   
   wavearray<double> *arr=new wavearray<double>(n);

   j=0;
//   while (fscanf(fp,"%lf",&x)&&j<arr->size())
   while (j<arr->size())
   {
     fscanf(fp,"%lf",&x);
     arr->data[j++] = x;
   }
   
   fclose(fp);
   return arr; 
}

wavearray<double>* readAsciiD(char *str, int lines=0)
{
   ifstream myInput;
   using namespace std;
   double x;
   int j=0;
         
   myInput.open(str);
   if (!myInput) {
      cerr << "Unable to open that file! "<<str<<endl;
      exit(1);
   }
   
   if(!lines){
    while (myInput >> x ) lines++;
    myInput.close();
    myInput.open(str);
   }

   wavearray<double> *arr=new wavearray<double>(lines);

   while (myInput >> x ) 
   {
      arr->data[j] = x;
      j++;
      if (j>lines)break;
   }
   
         
   myInput.close();
  return arr; 
}






