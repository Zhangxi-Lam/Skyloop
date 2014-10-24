#include "main.cuh"
#include "gpu_network.hh"

void test_function(network *net)
{
	cout<<"inside gpu function"<<endl;
	cout<<net->ifoList.size()<<endl;
	return;
}
