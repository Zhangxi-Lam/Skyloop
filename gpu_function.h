#define XIFO 3
#if XIFO == 3
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                                       \
P2                                       \
P3                              
#endif

#if XIFO < 5
#define _NET(P1,P2) \
P1                              
#endif   

inline void pnt_(float** q, float** p, short** m, int l, int n) 
{
// point 0-7 float pointers to first network pixel
	NETX(q[0] = (p[0] + m[0][l]*n);,
	     q[1] = (p[1] + m[1][l]*n);,
	     q[2] = (p[2] + m[2][l]*n);,
	     q[3] = (p[3] + m[3][l]*n);,
	     q[4] = (p[4] + m[4][l]*n);,
	     q[5] = (p[5] + m[5][l]*n);,
	     q[6] = (p[6] + m[6][l]*n);,
	     q[7] = (p[7] + m[7][l]*n);)
	return;
}   

inline void cpp_(float*& a, float** p) 
{
// copy to a data defined by array of pointers p and increment pointer
	NETX(*(a++) = *p[0]++;,
	     *(a++) = *p[1]++;,
	     *(a++) = *p[2]++;,
	     *(a++) = *p[3]++;,
	     *(a++) = *p[4]++;,
	     *(a++) = *p[5]++;,
	     *(a++) = *p[6]++;,
	     *(a++) = *p[7]++;)
	return;
}

inline void cpf_(float*& a, double** p, size_t i) //GV
{ 
// copy to a data defined by array of pointers p and increment target pointer
	NETX(*(a++) = p[0][i];,
	     *(a++) = p[1][i];,
	     *(a++) = p[2][i];,                             
	     *(a++) = p[3][i];,  
	     *(a++) = p[4][i];,
	     *(a++) = p[5][i];,
	     *(a++) = p[6][i];,
	     *(a++) = p[7][i];)
	return;
} 

static inline void _sse_zero_ps(__m128* _p) 
{          
	_NET(_p[0] = _mm_setzero_ps();,                     
             _p[1] = _mm_setzero_ps();)                     
      	return;
}    

static inline void _sse_cpf_ps(__m128* _a, __m128* _p) {
   	_NET(*_a = *_p;, *(_a+1) = *(_p+1);)                
}    

static inline void _sse_add_ps(__m128* _a, __m128* _b) 
{
// _a += _b   
    	_NET(_a[0] = _mm_add_ps(_a[0],_b[0]);,
  	     _a[1] = _mm_add_ps(_a[1],_b[1]);)
	return;
}

static inline float _sse_abs_ps(__m128* _a, __m128* _A) {
	float x[4];
	float out;
	_NET(_mm_storeu_ps(x,_mm_add_ps(_mm_mul_ps(_a[0],_a[0]),_mm_mul_ps(_A[0],_A[0]))); out =x[0]+x[1]+x[2]+x[3];,
             _mm_storeu_ps(x,_mm_add_ps(_mm_mul_ps(_a[1],_a[1]),_mm_mul_ps(_A[1],_A[1]))); out+=x[0]+x[1]+x[2]+x[3];)
	return out;
}

static inline void _sse_mul_ps(__m128* _a, __m128* _b) {
  	_NET(_a[0] = _mm_mul_ps(_a[0],_b[0]);,
             _a[1] = _mm_mul_ps(_a[1],_b[1]);)
}

static inline float _sse_maxE_ps(__m128* _a, __m128* _A) {
// given input 00 and 90 data vectors
// // returns energy of dominant detector (max energy)
	float out;
	float x[4];
	__m128 _o1;
	__m128 _o2 = _mm_setzero_ps();
	_NET(_o1 = _mm_add_ps(_mm_mul_ps(_a[0],_a[0]),_mm_mul_ps(_A[0],_A[0]));,
	     _o2 = _mm_add_ps(_mm_mul_ps(_a[1],_a[1]),_mm_mul_ps(_A[1],_A[1]));)
	_o1 = _mm_max_ps(_o1,_o2); _mm_storeu_ps(x,_o1); out=x[0];
	if(out<x[1]) out=x[1];
	if(out<x[2]) out=x[2];
	if(out<x[3]) out=x[3];
	return out;
}
	
static inline void _sse_dpf4_ps(__m128* _Fp, __m128* _Fx, __m128* _fp, __m128* _fx) {
// transformation to DPF for 4 consecutive pixels.
// rotate vectors Fp and Fx into DPF: fp and fx
	__m128 _c, _s;
	_sse_ort4_ps(_Fp,_Fx,&_s,&_c);                                        // get sin and cos
	_sse_rot4p_ps(_Fp,&_c,_Fx,&_s,_fp);                                   // get fp=Fp*c+Fx*s  
	_sse_rot4m_ps(_Fx,&_c,_Fp,&_s,_fx);                                   // get fx=Fx*c-Fp*s 
	return;
}
 
tatic inline __m128 _sse_like4_ps(__m128* fp, __m128* fx, __m128* am, __m128* AM) {
// input fp,fx - antenna patterns in DPF
// input am,AM - network amplitude vectors
// returns: (xp*xp+XP*XP)/|f+|^2+(xx*xx+XX*XX)/(|fx|^2)
	__m128 xp = _sse_dot4_ps(fp,am);                                 // fp*am
	__m128 XP = _sse_dot4_ps(fp,AM);                                 // fp*AM
	__m128 xx = _sse_dot4_ps(fx,am);                                 // fx*am
	__m128 XX = _sse_dot4_ps(fx,AM);                                 // fx*AM
	__m128 gp = _mm_add_ps(_sse_dot4_ps(fp,fp),_mm_set1_ps(1.e-12)); // fx*fx + epsilon 
	__m128 gx = _mm_add_ps(_sse_dot4_ps(fx,fx),_mm_set1_ps(1.e-12)); // fx*fx + epsilon 
	xp = _mm_add_ps(_mm_mul_ps(xp,xp),_mm_mul_ps(XP,XP));     // xp=xp*xp+XP*XP
	xx = _mm_add_ps(_mm_mul_ps(xx,xx),_mm_mul_ps(XX,XX));     // xx=xx*xx+XX*XX
	return _mm_add_ps(_mm_div_ps(xp,gp),_mm_div_ps(xx,gx));          // regularized projected energy
}

static inline __m128 _sse_sum_ps(__m128** _p) {
   __m128 _q = _mm_setzero_ps();
   NETX(_q = _mm_add_ps(_q, *_p[0]);,
        _q = _mm_add_ps(_q, *_p[1]);,
        _q = _mm_add_ps(_q, *_p[2]);,
        _q = _mm_add_ps(_q, *_p[3]);,
        _q = _mm_add_ps(_q, *_p[4]);,
        _q = _mm_add_ps(_q, *_p[5]);,
        _q = _mm_add_ps(_q, *_p[6]);,
        _q = _mm_add_ps(_q, *_p[7]);)
      return _q;
}

static inline void _sse_minSNE_ps(__m128* _pE, __m128** _pe, __m128* _es) {
// put pixel minimum subnetwork energy in _es
// input _es should be initialized to _pE before call
	NETX(*_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[0]++));,
	*_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[1]++));,
	*_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[2]++));,
	*_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[3]++));,
	*_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[4]++));,
	*_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[5]++));,
	*_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[6]++));,
	*_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[7]++));)	
}                                                                                                                                                                   

inline int network::_sse_MRA_ps(float* amp, float* AMP, float Eo, int K, class monster *wdmMRA) {
// fast multi-resolution analysis inside sky loop                         
// // select max E pixel and either scale or skip it based on the value of residual 
// // pointer to 00 phase amplitude of monster pixels                               
// // pointer to 90 phase amplitude of monster pixels                               
// // Eo - energy threshold                                                         
// //  K - number of principle components to extract                                
// // returns number of MRA pixels                                                  

#ifndef __CINT__
	int j,n,mm;
	int k = 0;
	int m = 0;
	int f = NIFO/4;
	int V = (int)this->rNRG.size();
	float*  ee = this->rNRG.data;                            // residual energy   
	float*  pp = this->pNRG.data;                            // residual energy   
	float   EE = 0.;                                         // extracted energy  
	float   E;
	float mam[NIFO];
	float mAM[NIFO];
	this->pNRG=-1;
	for(j=0; j<V; ++j) if(ee[j]>Eo) pp[j]=0;
	
	__m128* _m00 = (__m128*) mam;
	__m128* _m90 = (__m128*) mAM;
	__m128* _amp = (__m128*) amp;
	__m128* _AMP = (__m128*) AMP;
	__m128* _a00 = (__m128*) a_00.data;
	__m128* _a90 = (__m128*) a_90.data;
	
	while(k<K)
	{
	
		for(j=0; j<V; ++j) if(ee[j]>ee[m]) m=j;               // find max pixel
		if(ee[m]<=Eo) break;  mm = m*f;
		
		//cout<<" V= "<<V<<" m="<<m<<" ee[m]="<<ee[m];
		
		E = _sse_abs_ps(_a00+mm,_a90+mm); EE += E;     // get PC energy
		int    J = wdmMRA->getXTalk(m)->size()/7;
		float* c = wdmMRA->getXTalk(m)->data;                  // c1*c2+c3*c4=c1*c3+c2*c4=0
		
		if(E/EE < 0.01) break;                                // ignore small PC
		
		_sse_cpf_ps(mam,_a00+mm);                             // store a00 for max pixel
		_sse_cpf_ps(mAM,_a90+mm);                             // store a90 for max pixel
		_sse_add_ps(_amp+mm,_m00);                            // update 00 PC           
		_sse_add_ps(_AMP+mm,_m90);                            // update 90 PC           
		
		for(j=0; j<J; j++) 
		{
			n = int(c[0]+0.1);
			if(ee[n]>Eo) 
			{
				ee[n] = _sse_rotsub_ps(_m00,c[1],_m90,c[2],_a00+n*f);    // subtract PC from a00
				ee[n]+= _sse_rotsub_ps(_m00,c[3],_m90,c[4],_a90+n*f);    // subtract PC from a90
			}	
			c += 7;
		}
		//cout<<" "<<ee[m]<<" "<<k<<" "<<E<<" "<<EE<<" "<<endl;                               
		pp[m] = _sse_abs_ps(_amp+mm,_AMP+mm);    // store PC energy                           
		k++;
	}

	/* 
	cout<<"EE="<<EE<<endl;
	EE = 0;               
	for(j=0; j<V; ++j) {  
	if(pp[j]>=0) EE += ee[j];
	if(pp[j]>=0.) cout<<j<<"|"<<pp[j]<<"|"<<ee[j]<<" ";               // find max pixel
	}                                                                                     
	cout<<"EE="<<EE<<endl;                                                                
	*/
	return k;
#else
	return 0;
#endif
} 

                                                                                                                                    
