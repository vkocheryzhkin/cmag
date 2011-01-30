#define BOOST_TEST_MODULE FirstTest
#include <boost/test/unit_test.hpp>
#include <vector_types.h>
typedef unsigned int uint;

#include "fluidSystem.cuh"
#include "fluidSystem.h"
BOOST_AUTO_TEST_CASE(FluidTest)
{	
	cudaInit(1,(char **) &"");
	int num = 25;
	uint3 fluidParticlesSize = make_uint3(num, num, num);	    
	uint3 gridSize = make_uint3(64, 64, 64);   	

	FluidSystem *psystem = new FluidSystem(fluidParticlesSize, gridSize, false); 
	psystem->reset();

	float *hPos = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());
	float *hrPos = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());	
	float *htemp = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());		
	float *hacc = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());		
	uint* hHash = new uint[psystem->getNumParticles()];
	uint* hIndex = new uint[psystem->getNumParticles()];		

	psystem->update();	

	copyArrayFromDevice(hPos,psystem->getCudaSortedPosition(),0, sizeof(float)*4*psystem->getNumParticles());	
	copyArrayFromDevice(htemp,psystem->getCudaMeasures(),0, sizeof(float)*4*psystem->getNumParticles());		
	copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
	copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());			

	float max = 0.0f;
	for(uint i = 0; i < psystem->getNumParticles(); i++) 
	{			
		float &dens = htemp[4 * i + 0];
		if(max < dens)
			max = dens;		
	}
	printf("max=%f\n", max);		
	printf("---------------------\n");			

	delete [] hPos; 
	delete [] hrPos; 	
	delete [] htemp; 
	delete [] hacc; 	
	delete [] hHash; 
	delete [] hIndex; 
	delete psystem;
}
