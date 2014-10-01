#include<iostream>
#include<fstream>
#include<emmintrin.h>
using namespace std;

int main(void)
{
	
	float a[4];
	float b[4];
	float c[4];
	float d[4];
	float e[4];
	
	__m128* _a;
	__m128* _b;
	__m128* _c;
	__m128* _d;
	__m128* _e;
	
	_a = (__m128*) a;
	_b = (__m128*) b;
	_c = (__m128*) c;
	_d = (__m128*) d;
	_e = (__m128*) e;
	
/*	a[0] = 41.938118; 
	b[0] = 21.751024;
	c[0] = 26.384308;
	d[0] = 24.660862;*/
	a[0] = 22.259048;
	b[0] = 21.751024;
	c[0] = 26.384308;
	d[0] = 24.660862;
	e[0] = 19.679070;
	a[1] = b[0];
	a[2] = c[0];
	a[3] = d[0];
	for(int j=1; j<4; j++)
	{
		a[j] = 0;
		b[j] = 0;
		c[j] = 0;
		d[j] = 0;
		e[j] = 0;
	}
	/*for(int j=1; j<4; j++)	
		a[0] = a[0] + a[j];
	a[0] += 0.01;*/
	/**_a = _mm_add_ps(*_a, *_b);
	_mm_storeu_ps(a, *_a);
	printf("a = %f\n", a[0]);
	*_a = _mm_add_ps(*_a, *_c);
	_mm_storeu_ps(a, *_a);
	printf("a = %f\n", a[0]);
	*_a = _mm_add_ps(*_a, *_d);
	_mm_storeu_ps(a, *_a);
	printf("a = %f\n", a[0]);*/
//	a[0] = a[0] + b[0] + c[0] + d[0] + e[0] +  0.01;
//	a[0] = e[0] + d[0] + c[0] + b[0] + a[0] + 0.01;
	a[0] = a[0] + b[0];
	printf("a = %f\n", a[0]);
	a[0] = a[0] + c[0];
	printf("a = %f\n", a[0]);
	a[0] = a[0] + d[0];
	printf("a = %f\n", a[0]);
	a[0] = a[0] + e[0];
	printf("a = %f\n", a[0]);
	a[0] += 0.01;
	_mm_storeu_ps(a, *_a);
	printf("a = %f\n", a[0]);
	return 0;		
}	
