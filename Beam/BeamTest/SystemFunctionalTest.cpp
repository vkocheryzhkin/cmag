#define BOOST_TEST_MODULE FirstTest
#include <boost/test/unit_test.hpp>
#include <vector_types.h>
#include "beamSystem.cuh"
#include <beamSystem.h>

typedef unsigned int uint;
BOOST_AUTO_TEST_CASE(Check_system)
{	
	cudaInit(1,(char **) &"");
	uint numParticles = 1*1*4;
	uint3 gridSize;
	gridSize.x = gridSize.y = gridSize.z = 64;		

    ParticleSystem *psystem = new ParticleSystem(numParticles, gridSize, false); 
	psystem->reset();

	float *hPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());
	float *htemp = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());		
	uint* hHash = new uint[numParticles];
	uint* hIndex = new uint[numParticles];	

	for(uint j=0; j<1; j++)
	{		
		psystem->update();	
	
		copyArrayFromDevice(hPos,psystem->getCudaPositionVBO(),0, sizeof(float)*4*psystem->getNumParticles());
		copyArrayFromDevice(htemp,psystem->getCudaMeasures(),0, sizeof(float)*4*psystem->getNumParticles());	
		copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
		copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());	

		for(uint i=0; i<numParticles; i++) 
		{
			if(i!=22){
				printf("%d (%d %2d) y%d dens=%f vol=%f w=%f\n", 
						j,
						hHash[i],
						hIndex[i],	
						hIndex[i],

						htemp[4*hIndex[i]+0],
						htemp[4*hIndex[i]+1],
						hPos[4*hIndex[i]+3]
					);	
			}
		}
	}
	

	delete [] hPos; 
	delete [] htemp; 
	delete [] hHash; 
	delete [] hIndex; 
	delete psystem;
}