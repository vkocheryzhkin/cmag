#define BOOST_TEST_MODULE FirstTest
#include <boost/test/unit_test.hpp>
#include <vector_types.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>

typedef unsigned int uint;
#include "fluidSystem.cuh"
#include "fluidSystem.h"
#include "magUtil.cuh"

using namespace std;

BOOST_AUTO_TEST_CASE(FluidTest)
{		
	int num = 25;
	uint3 fluidParticlesSize = make_uint3(num, num, 1);	    
	uint3 gridSize = make_uint3(64, 64, 64);   		

	DamBreakSystem *psystem = new DamBreakSystem(fluidParticlesSize, gridSize, 1.0f / 64, false); 
	psystem->reset();

	float *hPos = (float*)malloc(sizeof(float) * 4 * psystem->getNumParticles());
	float *hrPos = (float*)malloc(sizeof(float) * 4 * psystem->getNumParticles());	
	float *htemp = (float*)malloc(sizeof(float) * 4 * psystem->getNumParticles());		
	float *hacc = (float*)malloc(sizeof(float) * 4 * psystem->getNumParticles());		
	uint* hHash = new uint[psystem->getNumParticles()];
	uint* hIndex = new uint[psystem->getNumParticles()];		

	for(int i =0;i<100;i++)
		psystem->update();	
	//getCudaAcceleration
	//getCudaMeasures
	copyArrayFromDevice(hPos, psystem->getCudaSortedPosition(),0, sizeof(float)*4*psystem->getNumParticles());	
	copyArrayFromDevice(htemp, psystem->getCudaMeasures(),0, sizeof(float)*4*psystem->getNumParticles());		
	copyArrayFromDevice(hHash, psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
	copyArrayFromDevice(hIndex, psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());			


	thrust::device_ptr<float4> dev_ptr((float4*)psystem->getCudaMeasures());	
	thrust::host_vector<float4> h_vec(psystem->getNumParticles());
	
	thrust::copy(dev_ptr, dev_ptr + psystem->getNumParticles(), h_vec.begin());

	struct compare_float4{
		__host__ bool operator()(float4 a, float4 b){
				return a.x < b.x;
		}
	};

	compare_float4 comp;
	thrust::sort(h_vec.begin(),h_vec.end(),comp);
		
	 for(int i = 0; i < h_vec.size(); i++)
        std::cout << "d_vec[" << i << "] = " << ((float4)h_vec[i]).x << std::endl;

	//for(uint i = 0; i < psystem->getNumParticles(); i++) 
	//{	
	////cout << dev_ptr[0] << endl;
	////printf("%f", (float)dev_ptr[0]);	
	//}


	//float tt = 0.0f;
	//for(uint i = 0; i < psystem->getNumParticles(); i++) 
	//{					
	//	tt+=htemp[4 * i + 0];
	//	printf("1: %f 2: %f 3: %f\n", htemp[4 * i + 0],htemp[4 * i + 1],htemp[4 * i + 3]);		
	//	//printf("acc =%1.15f %1.15f\n", htemp[4 * hIndex[i] + 0], htemp[4 * hIndex[i] + 1]);		
	//}
	//printf("avg=%f\n", tt / ( num * num ) );		

	delete [] hPos; 
	delete [] hrPos; 	
	delete [] htemp; 
	delete [] hacc; 	
	delete [] hHash; 
	delete [] hIndex; 	
	delete psystem;	
}
