{
int toff=1024*0;
wavearray<float> *S;
wavearray<float> *Lh;
wavearray<float> *Ll;
wavearray<float> *p;
wavearray<float> xh;
wavearray<float> xl;
wavearray<float> ss;
Biorthogonal<float> B(8);
Wavelet *pW = &B;
pW->setLevel(10);
WSeries<float> W(*pW);

//S=ReadFrFile(512*4,toff+0,"H0:PEM-LVEA_SEISX","/cms/tmp3/ligo/E7/lho/H-R-693844176-16.sw");
//S=ReadFrFile(512*6,toff+128,"H0:PEM-LVEA2_V1","/cms/tmp3/ligo/E7/lho/H-R-693844176-16.sw");
S=ReadFrFile(512*4,128,"H2:LSC-DARM_CTRL","/cms/tmp3/ligo/E7/lho/H-R-693844176-16.sw");
Lh=ReadFrFile(512*4,toff+128,"H2:LSC-LA_State_Bits_Read","/cms/tmp3/ligo/E7/lho/H-R-693844176-16.sw");
W.resize(S->size());
p=&W; *p = *S; delete S;
W.getLayer(xh,5);

//S=ReadFrFile(512*4,toff+0,"L0:PEM-LVEA_SEISX","/cms/tmp3/ligo/E7/llo/L-R-693844304-16.sw");
//S=ReadFrFile(512*6,toff+0,"L0:PEM-LVEA_V1","/cms/tmp3/ligo/E7/llo/L-R-693844304-16.sw");
S=ReadFrFile(512*4,0,"L1:LSC-DARM_CTRL","/cms/tmp3/ligo/E7/llo/L-R-693844304-16.sw");
Ll=ReadFrFile(512*4,toff+0,"L1:LSC-LA_State_Bits_Read","/cms/tmp3/ligo/E7/llo/L-R-693844304-16.sw");
p=&W; *p = *S; delete S;
W.getLayer(xl,5);

ss=xl;
double T;
bool lock;
int nL;

cout << " Rate="<< ss.rate() <<endl;

for(int i=0; i<ss.size(); i++){
   T = i/ss.rate();
   nL = int(T*Lh->rate());
   lock = true;
   if(Lh->data[nL]!=61 && Lh->data[nL]!=59) lock=false;
   if(Ll->data[nL]!=61 && Ll->data[nL]!=59) lock=false;

   if(lock) ss.data[i] *= xh.data[i];
   else ss.data[i] = 0.;
}

wavearray<float> an(256*3);
xsp1=xcor(ss,ss,16*8192);

}



