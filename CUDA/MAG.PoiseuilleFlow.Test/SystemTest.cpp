#define BOOST_TEST_MODULE PoiseuilleFlowTest
#include <boost/test/unit_test.hpp>
#include <vector_types.h>
#include <vector_functions.h>
#include <stdio.h>
typedef unsigned int uint;

#include "poiseuilleFlowSystem.cuh"
#include "poiseuilleFlowSystem.h"
#include "magUtil.cuh"

BOOST_FIXTURE_TEST_SUITE(SystemTest)//, InitCuda)

BOOST_AUTO_TEST_CASE(SystemTest1)
{		
	FILE *fp1= fopen("PoiseuilleFlowOutput1", "w");
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

	while(psystem->getElapsedTime() <= 0.0225)
		psystem->update();
	
				
	copyArrayFromDevice(hPos,psystem->getCudaSortedPosition(),0, sizeof(float)*4*psystem->getNumParticles());	
	copyArrayFromDevice(htemp,psystem->getLeapFrogVelocity(),0, sizeof(float)*4*psystem->getNumParticles());		
	copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
	copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());			

	for(uint i = 0; i < psystem->getNumParticles(); i++) 
	{		
		float posx = hPos[4*i+0];		
		if((posx > psystem->getHalfWorldXSize() + psystem->getWorldOrigin().x) 
			&& (posx < psystem->getHalfWorldXSize() + 2 * psystem -> getParticleRadius()+
			+ psystem->getWorldOrigin().x))
		{
			fprintf(fp1, "%1.16f %1.16f \n", htemp[4*hIndex[i]+0], hPos[4*i+1]
			+ abs(psystem->getWorldOrigin().y) - boundaryOffset * 2 *psystem -> getParticleRadius());					
		}										
	}
	printf("%1.16f", particleRadius);
	fclose(fp1);
	delete [] hPos; 
	delete [] hrPos; 	
	delete [] htemp; 
	delete [] hacc; 	
	delete [] hHash; 
	delete [] hIndex; 
	delete psystem;
}

BOOST_AUTO_TEST_SUITE_END()
