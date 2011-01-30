#define BOOST_TEST_MODULE FirstTest
#include <boost/test/unit_test.hpp>
#include <vector_types.h>
typedef unsigned int uint;


#include "fluidbeamSystem.cuh"
#include "fluidbeamSystem.h"
BOOST_AUTO_TEST_CASE(FluidBeamTest)
{	
	cudaInit(1,(char **) &"");	
	int boundaryOffset = 5;
	uint3 fluidParticlesGrid = make_uint3(0, 0, 0);
	//uint3 fluidParticlesGrid = make_uint3(0, 0, 0);
	uint3 beamParticlesGrid = make_uint3(1, 2 + 2 * boundaryOffset, 2 + 2 * boundaryOffset);	
	uint3 gridSize = make_uint3(64, 64, 64);
	float particleRadius = 1.0f / 64;
	
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

	float extrm = 0.0f;	
    bool indicator = true;	
	bool pindicator = true;	
	for(uint j = 0; j < 1; j++)
	{				
		psystem->update();	
		
		copyArrayFromDevice(hPos,psystem->getCudaSortedPosition(),0, sizeof(float)*4*psystem->getNumParticles());		
		copyArrayFromDevice(htemp,psystem->getWDisplacementGradient(),0, sizeof(float)*4*psystem->getNumParticles());			
		copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
		copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());			
				
		int cx = 0;
		for(uint i=0; i < psystem->getNumParticles(); i++) 
		{									
				if(hPos[4*i+3] ==1.0f)
			printf("%d id=%d (%d %2d) %f %1.15f %1.15f w=%f\n", 
					cx++,
					i,
					hHash[i],
					hIndex[i],	
				    //use hIndex[i] for not sorted items
					/*htemp[4*hIndex[i]+0],
					htemp[4*hIndex[i]+1],
					htemp[4*hIndex[i]+2],*/
					htemp[4*i+0],
					htemp[4*i+1],
					htemp[4*i+2],
					hPos[4*i+3]
				);					
		}			
	}
	
	

	delete [] hPos; 
	delete [] hrPos; 	
	delete [] htemp; 
	delete [] hacc; 	
	delete [] hHash; 
	delete [] hIndex; 
	delete psystem;
}
