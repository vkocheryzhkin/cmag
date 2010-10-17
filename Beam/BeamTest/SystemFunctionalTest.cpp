#define BOOST_TEST_MODULE FirstTest
#include <boost/test/unit_test.hpp>
#include <vector_types.h>
#include "beamSystem.cuh"
#include <beamSystem.h>

typedef unsigned int uint;
BOOST_AUTO_TEST_CASE(Check_system)
{	
	cudaInit(1,(char **) &"");
	uint numParticles = 1*1*2;
	uint3 gridSize;
	gridSize.x = gridSize.y = gridSize.z = 64;		

    ParticleSystem *psystem = new ParticleSystem(numParticles, gridSize, false); 
	psystem->reset();

	float *hPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());
	float *hBuf = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());		
	uint* hHash = new uint[numParticles];
	uint* hIndex = new uint[numParticles];	

	for(uint i=0; i<1200; i++)
	{
		psystem->update();	
	
		copyArrayFromDevice(hPos,psystem->getCudaPositionVBO(),0, sizeof(float)*4*psystem->getNumParticles());
		copyArrayFromDevice(hBuf,psystem->getCudavDisplacementGradient(),0, sizeof(float)*4*psystem->getNumParticles());	
		copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
		copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());	

		for(uint i=0; i<numParticles; i++) 
		{
			printf("hash: %d id: %2d pos=%f y=%f  w = %d \n", 
					hHash[i],
					hIndex[i],	
					hPos[4*hIndex[i]+1],
					hBuf[4*i+1],
					i
				);	
		}
	}

	delete [] hPos; 
	delete [] hBuf; 
	delete [] hHash; 
	delete [] hIndex; 
	delete psystem;
}