#define BOOST_TEST_MODULE FirstTest
#include <boost/test/unit_test.hpp>
#include <vector_types.h>
typedef unsigned int uint;

#include "fluidSystem.cuh"
#include "fluidSystem.h"
BOOST_AUTO_TEST_CASE(FluidTest)
{	
	cudaInit(1,(char **) &"");
	uint numFluidParticles = 5 * 5 * 5;
	uint3 gridSize;
	gridSize.x = gridSize.y = gridSize.z = 64;		

	FluidSystem *psystem = new FluidSystem(numFluidParticles, gridSize, false); 
	psystem->reset();

	float *hPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());
	float *hrPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());	
	float *htemp = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());		
	float *hacc = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());		
	uint* hHash = new uint[numFluidParticles];
	uint* hIndex = new uint[numFluidParticles];	

	for(uint j = 0; j < 1; j++)
	{				
		psystem->update();	
	}
	copyArrayFromDevice(hPos,psystem->getCudaSortedPosition(),0, sizeof(float)*4*psystem->getNumParticles());	
	copyArrayFromDevice(htemp,psystem->getCudaAcceleration(),0, sizeof(float)*4*psystem->getNumParticles());		
	copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
	copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());			

	for(uint i=0; i<numFluidParticles; i++) 
	{			
		printf("%d id=%d (%d %2d) %f %1.10f %f w=%f\n", 
			0,
			i,
			hHash[i],
			hIndex[i],	

			htemp[4*i+0],
			htemp[4*i+1],
			htemp[4*i+2],
			hPos[4*i+3]
		);	
	}
	printf("---------------------\n");			



	delete [] hPos; 
	delete [] hrPos; 	
	delete [] htemp; 
	delete [] hacc; 	
	delete [] hHash; 
	delete [] hIndex; 
	delete psystem;
}
