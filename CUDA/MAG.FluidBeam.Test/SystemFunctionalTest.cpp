#define BOOST_TEST_MODULE FirstTest
#include <boost/test/unit_test.hpp>
#include <vector_types.h>
typedef unsigned int uint;


#include "fluidbeamSystem.cuh"
#include "fluidbeamSystem.h"
BOOST_AUTO_TEST_CASE(FluidBeamTest)
{	
	cudaInit(1,(char **) &"");
	uint3 fluidParticlesGrid = make_uint3(3, 3, 3);
	uint3 beamParticlesGrid = make_uint3(0, 0, 0);
	uint3 gridSize = make_uint3(64, 64, 64);
	float particleRadius = 1.0f / 64;
	int boundaryOffset = 3;
	uint numFluidParticles = fluidParticlesGrid.x * fluidParticlesGrid.y * fluidParticlesGrid.z;

    FluidBeamSystem *psystem = new FluidBeamSystem(
		fluidParticlesGrid,
		beamParticlesGrid,
		boundaryOffset,
		gridSize,
		particleRadius,
		false); 
	psystem->reset();

	float *hPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());
	float *hrPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());	
	float *htemp = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());		
	float *hacc = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());		
	uint* hHash = new uint[psystem->getNumParticles()];
	uint* hIndex = new uint[psystem->getNumParticles()];	

	for(uint j = 0; j < 1; j++)
	{				
		psystem->update();	
	}
	//getCudaMeasures getCudaSortedPosition getCudaPositions
		copyArrayFromDevice(hPos,psystem->getCudaSortedPosition(),0, sizeof(float)*4*psystem->getNumParticles());
		//copyArrayFromDevice(hrPos,psystem->getCudaSortedReferencePosition(),0, sizeof(float)*4*psystem->getNumParticles());
		copyArrayFromDevice(htemp,psystem->getCudaMeasures(),0, sizeof(float)*4*psystem->getNumParticles());	
		//copyArrayFromDevice(hacc,psystem->getCudaAcceleration(),0, sizeof(float)*4*psystem->getNumParticles());			
		copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
		copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());			
				
		for(uint i=0; i < psystem->getNumParticles(); i++) 
		{			
			if(hPos[4*i+3] == 2.0f){
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
