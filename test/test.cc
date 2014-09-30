#include<iostream>
#include<fstream>
using namespace std;

int main(void)
{
	float s[10];
	ifstream file1("test");
	
	for(int i=0; i<10; i++)
	{
		file1>>s[i];
	}
	cout<<"s[0] = "<<s[0]<<endl;
	cout<<"s[1] = "<<s[1]<<endl;
	cout<<"s[2] = "<<s[2]<<endl;
	return 0;		
}	
