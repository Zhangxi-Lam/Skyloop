#include<fstream>
#include<iostream>
using namespace std;

int main(void)
{

//	FILE *file = fopen("../../skyloop_eTD", "r");
	ifstream file("../../skyloop_eTD");
	int eTDDim;
	string t;
	int line;
	float temp[4];
	eTDDim = 219 * 300;	
	line = 4;
	
	for(int i=0; i<line; i++)
		getline(file, t);
	//fseek(file, 4, 0);
	for(int i = 0; i<4; i++)
		//fread(temp, 4, 1, file);
		file>>temp[i];
	
	for(int i = 0; i<4; i++)
		cout<<temp[i]<<" ";
	cout<<endl;
}
