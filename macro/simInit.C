{
  
  char in_file[256]  =  "r2bsg1.inj";   // injection configuration file
  char simdir[256] = "plot/SG1";        // output directory for simplot.C
  char inj_run[16]   =  "L1H1H2V1G1";   // suffix

  char   iname[32][128];   // injection name
  double f_low[32];        // injection low frequencies
  double fhigh[32];        // injection high frequencies
  size_t itype[32];        // injection type
  size_t inDex[32];        // type reference array

  Color_t colors[32] = { 6, 3, 2, 8,43, 7, 8, 4, 4, 2,43, 1, 3, 1, 6, 7,
			 6, 3, 2, 8,43, 7, 8, 4, 4, 2,43, 1, 3, 1, 6, 7 };
  Style_t markers[32]= {20,21,22,23,29,29,21,22,20,21,22,23,29,20,21,22,
			21,22,23,29,29,21,22,20,21,22,23,29,20,21,22,20 };
 
  int n = 0;
  size_t ninj;
  char str[1024];
  char* p;
  
  for(int i=0; i<32; i++) inDex[i] = 32;

  FILE* in = fopen(in_file,"r");
  while(fgets(str,1024,in) != NULL){
    if(str[0] == '#') continue;
    //    cout<<str<<endl;
    if((p = strtok(str," \t")) == NULL) continue;
    itype[n] = (size_t) atoi(p);         // get injection type

    if((p = strtok((char*)NULL," \t")) == NULL) continue;
    sprintf(iname[n],"%s",p);   // get injection name

    f_low[n] = 0.;
    fhigh[n] = 2096;
    if((p = strtok((char*)NULL," \t")) == NULL) continue;
    f_low[n] = atof(p);         // get low frequency
    if((p = strtok((char*)NULL," \t")) == NULL) continue;
    fhigh[n] = atof(p);         // get high frequency

    inDex[itype[n]] = n;
    if(n<30) n++;
  }
  ninj = n;
  fclose(in);
  
  sprintf(iname[ninj],"%s",inj_run);

// sigmoid fit function

TF1 *f4=new TF1("logNfit",logNfit,pow(10.0,-22.0),pow(10.0,-18.5),4);
f4->SetParameters(-21.0,0.7,1.,1.);
f4->SetParNames("hrss50","sigma","betam","betap");
f4->SetParLimits(0,-22.,-18.5);
f4->SetParLimits(1,0.1,10.);
f4->SetParLimits(2,0.1,4.);
f4->SetParLimits(3,0.1,4.);

// legend pannel

TLegend *legend = new TLegend(0.6,0.17,0.92,0.7,"","brNDC");
legend->SetLineColor(1);
legend->SetLineStyle(1);
legend->SetLineWidth(1);
legend->SetFillColor(10);
legend->SetFillStyle(1001);

}



