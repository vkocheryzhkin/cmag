#define BOOST_TEST_MODULE PoiseuilleFlowTest
#include <boost/test/unit_test.hpp>
#include <vector_types.h>
#include <vector_functions.h>
typedef unsigned int uint;

#include "poiseuilleFlowSystem.cuh"
#include "poiseuilleFlowSystem.h"
#include "magUtil.cuh"

BOOST_FIXTURE_TEST_SUITE(SystemTest)//, InitCuda)

BOOST_AUTO_TEST_CASE(SystemTest1)
{		
	int boundaryOffset = 3;	
	//uint3 gridSize = make_uint3(16, 64, 2);    
	uint3 gridSize = make_uint3(16, 64, 4);    
	float particleRadius = 1.0f / (2 * gridSize.y * 1000);	
	uint3 fluidParticlesSize = make_uint3(gridSize.x, gridSize.y -  2 * boundaryOffset, 1);	       
		

	PoiseuilleFlowSystem *psystem = new PoiseuilleFlowSystem(
		fluidParticlesSize,
		boundaryOffset, 
		gridSize, 
		particleRadius,
		false); 
	psystem->reset();

	float *hPos = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());
	float *hrPos = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());	
	float *htemp = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());		
	float *hacc = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());		
	uint* hHash = new uint[psystem->getNumParticles()];
	uint* hIndex = new uint[psystem->getNumParticles()];		

	float minDens = 3.0f;
	for(int j = 0; j < 1; j++)
	//{
		psystem->update();
		//getCudaMeasures
		//getCudaAcceleration
		copyArrayFromDevice(hPos,psystem->getCudaSortedPosition(),0, sizeof(float)*4*psystem->getNumParticles());	
		copyArrayFromDevice(htemp,psystem->getCudaMeasures(),0, sizeof(float)*4*psystem->getNumParticles());		
		copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
		copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());			

		float sum = 0.0f;
		int cx = 0;
		for(uint i = 0; i < psystem->getNumParticles(); i++) 
		{		
			if(hPos[4*i+3] == 0.0f){		
				sum += htemp[4*i+0];
				printf("%d id=%d (%d %2d) %1.16f %f %f w=%f\n", 
						cx++,
						i,
						hHash[i],
						hIndex[i],	
						//use hIndex[i] for not sorted items
					/*	htemp[4*hIndex[i]+0],
						htemp[4*hIndex[i]+1],
						htemp[4*hIndex[i]+2],*/
						htemp[4*i+0],
						htemp[4*i+1],
						htemp[4*i+2],
						hPos[4*i+3]
					);		
			}
		}	
	printf("%f ---------------------\n", sum / (gridSize.x*(gridSize.y - 2 * boundaryOffset)));			
	//}

	delete [] hPos; 
	delete [] hrPos; 	
	delete [] htemp; 
	delete [] hacc; 	
	delete [] hHash; 
	delete [] hIndex; 
	delete psystem;
}

BOOST_AUTO_TEST_SUITE_END()
