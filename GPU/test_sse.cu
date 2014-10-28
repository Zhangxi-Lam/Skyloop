#include <xmmintrin.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
using namespace std;
int main()
{
	double Clock[10];
	int Lsky = 196608;	
	float vvv[4];
	__m128 _E_o;
	__m128 _E_n;

	cout<<"nvcc"<<endl;
	Clock[0] = clock();
	for(int c=190000; c<Lsky; c++)
		for(int d=0; d<Lsky; d++)
		{
			_E_o = _mm_setzero_ps();
			_E_n = _mm_set1_ps(1.);
			_E_o = _mm_add_ps(_E_o, _E_n);				
		}
	_mm_storeu_ps(vvv,_E_o);
	cout<<"vvv = "<<(vvv[0]+vvv[1]+vvv[2]+vvv[3])<<endl;
	Clock[1] = clock();	
	printf("time = %f\n", (double)(Clock[1]-Clock[0])/CLOCKS_PER_SEC);
}
