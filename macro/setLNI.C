{   
   WaveData a;
   a.ReadBinary("m2a.dat");
   a.Rate=5000;
   a*=167;

   WaveData a1(2500);
   WaveData a2(2500);
   WaveData a3(2500);
   WaveData a4(2500);

   a1.cpf(a,2500,0);
   a2.cpf(a,2500,2500);
   a3.cpf(a,2500,5000);
   a4.cpf(a,2500,7500);

   TF1 *fit=new TF1("fit",fitf,0.01,0.3,4);
   fit->SetParameters(0.045,19.2,-1.48,0.);
//   fit->SetParameters(0.045,17.7,1.48,0.);
}
