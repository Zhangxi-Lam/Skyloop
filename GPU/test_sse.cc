#include <xmmintrin.h>
#include <time.h>
#include <iostream>
#include <stdio.h>
#define CLOCK_SIZE 10
using namespace std;

int main()
{
        double Clock[CLOCK_SIZE];
        int Loop = 196608;
        float test[4];
        __m128 _zz;
        __m128 _yy;
	float zz[4];
	float yy[4];

        cout<<"nvcc"<<endl;
        Clock[8] = clock();
        for(int c=190000; c<Loop; c++)
                for(int d=0; d<Loop; d++)
                {
                        _zz = _mm_setzero_ps();
                        _yy = _mm_set1_ps(1.);
                        _zz = _mm_add_ps(_zz, _yy);
                }
        _mm_storeu_ps(test,_zz);
        Clock[9] = clock();
        printf("sse_time = %f\n", (double)(Clock[9]-Clock[8])/CLOCKS_PER_SEC);
        cout<<"test = "<<(test[0]+test[1]+test[2]+test[3])<<endl;

	Clock[8] = clock();
        for(int c=190000; c<Loop; c++)
                for(int d=0; d<Loop; d++)
		{
			zz[0] = 0;
			zz[1] = 0;
			zz[2] = 0;
			zz[3] = 0;
			yy[0] = 1;
			yy[1] = 1;
			yy[2] = 1;
			yy[3] = 1;
			zz[0] += yy[0];
			zz[1] += yy[1];
			zz[2] += yy[2];
			zz[3] += yy[3];
		}	
	test[0] = zz[0];
	test[1] = zz[1];
	test[2] = zz[2];
	test[3] = zz[3];
	Clock[9] = clock();
        printf("serial_time = %f\n", (double)(Clock[9]-Clock[8])/CLOCKS_PER_SEC);
        cout<<"test = "<<(test[0]+test[1]+test[2]+test[3])<<endl;
}

