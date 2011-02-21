#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
typedef unsigned int uint;

#include "poiseuilleFlowSystem.cuh"
#include "poiseuilleFlowSystem.h"
#include "magUtil.cuh"

BOOST_FIXTURE_TEST_SUITE(CalculateMass)

BOOST_AUTO_TEST_CASE(CalculateMass)
{	
	int boundaryOffset = 3;	
	uint3 gridSize = make_uint3(16, 64, 4);    	
	float particleRadius = 1.0f / (2 * (gridSize.y - 2 * boundaryOffset) * 1000);	
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
	
	psystem->update();
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
					htemp[4*i+0],
					htemp[4*i+1],
					htemp[4*i+2],
					hPos[4*i+3]
				);		
		}
	}	
	printf("%f ---------------------\n", sum / (gridSize.x * (gridSize.y - 2 * boundaryOffset)));			
	delete [] hPos; 
	delete [] hrPos; 	
	delete [] htemp; 
	delete [] hacc; 	
	delete [] hHash; 
	delete [] hIndex; 
	delete psystem;
}
BOOST_AUTO_TEST_SUITE_END()