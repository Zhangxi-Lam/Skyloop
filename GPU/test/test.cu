#include<iostream>
#include<xmmintrin.h>
#include<math.h>

using namespace std;

int main(void)
{
	float sm = 0.f;
	float gR = -0.449986;
	float n = 0.503547;
	float o = 1.e-24;
//	gR = ((!sm)&&gR) + (n + o);
	static const __m128 _sm = _mm_set1_ps(-0.f); 
	__m128 _gR = _mm_set1_ps(-0.449986);
	float tmp[4];

//	gR = ((!sm)*gR);
	gR = abs(gR);
	cout<<gR<<endl;
	_gR = _mm_andnot_ps(_sm, _gR);
	_mm_storeu_ps(tmp, _gR);
	cout<<tmp[0]<<endl;
	return;
	
}
