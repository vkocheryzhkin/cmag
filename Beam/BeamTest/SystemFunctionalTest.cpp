#define BOOST_TEST_MODULE FirstTest
#include <boost/test/unit_test.hpp>
#include <vector_types.h>
#include "beamSystem.cuh"
#include <beamSystem.h>

typedef unsigned int uint;
BOOST_AUTO_TEST_CASE(Check_system)
{	
	cudaInit(1,(char **) &"");
	uint numParticles = 1*1*3;
	uint3 gridSize;
	gridSize.x = gridSize.y = gridSize.z = 64;		

    ParticleSystem *psystem = new ParticleSystem(numParticles, gridSize, false); 
	psystem->reset();

	float *hPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());
	float *hRefPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());		
	uint* hHash = new uint[numParticles];
	uint* hIndex = new uint[numParticles];	

	for(uint i=0; i<700; i++)
	{
		psystem->update();	
	
		copyArrayFromDevice(hPos,psystem->getCudaPositionVBO(),0, sizeof(float)*4*psystem->getNumParticles());
		copyArrayFromDevice(hRefPos,psystem->getCudaSortedReferencePosition(),0, sizeof(float)*4*psystem->getNumParticles());	
		copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
		copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());	

		for(uint i=0; i<numParticles; i++) 
		{
			if(i!=2){
				printf("hash: %d id: %2d 0=%f  : 1=%f \n", 
						hHash[i],
						hIndex[i],	
						hRefPos[4*0+1] - hPos[4*hIndex[0]+1],					
						hRefPos[4*1+1] - hPos[4*hIndex[1]+1]					
					);	
			}
		}
	}

	delete [] hPos; 
	delete [] hRefPos; 
	delete [] hHash; 
	delete [] hIndex; 
	delete psystem;
}