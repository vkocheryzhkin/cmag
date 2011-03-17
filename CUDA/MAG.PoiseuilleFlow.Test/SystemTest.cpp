//#define BOOST_TEST_MODULE PoiseuilleFlowTest
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

BOOST_FIXTURE_TEST_SUITE(SystemOutput)

void mytest(float recordTime)
{	
	std::string fileNameEnding = str(boost::format("%1%") % recordTime);	
	std::string fileName = str(boost::format("PoiseuilleFlowOutput%1%") % fileNameEnding.replace(1,1,"x")); 
	FILE *fp1= fopen(fileName.c_str(), "w");
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

	while(psystem->getElapsedTime() < recordTime)
		psystem->update();	
				
	copyArrayFromDevice(hPos,psystem->getCudaSortedPosition(),0, sizeof(float)*4*psystem->getNumParticles());	
	copyArrayFromDevice(htemp,psystem->getLeapFrogVelocity(),0, sizeof(float)*4*psystem->getNumParticles());		
	copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
	copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());			

	fprintf(fp1, "%f %f \n", 0.0f, 0.0f);
	for(int i = 0; i < (psystem->getNumParticles()); i++) 
	{		
		float posx = hPos[4*i + 0];		
		float posy = hPos[4*i + 1];	
		if((posx > psystem->getHalfWorldXSize() + psystem->getWorldOrigin().x) 
			&& (posx < psystem->getHalfWorldXSize() + 2 * psystem -> getParticleRadius()+ psystem->getWorldOrigin().x)
			&& (posy > psystem->getWorldOrigin().y + 2 * psystem -> getParticleRadius() * boundaryOffset)
			&& (posy < psystem->getHalfWorldYSize() - 2 * psystem -> getParticleRadius() * boundaryOffset))
		{
			fprintf(fp1, "%1.16f %1.16f \n", htemp[4*hIndex[i]+0], hPos[4*i+1]
			+ abs(psystem->getWorldOrigin().y) - boundaryOffset * 2 *psystem -> getParticleRadius());					
		}										
	}
	fprintf(fp1, "%f %f \n", 0.0f, pow(10.0f,-3));	
	fclose(fp1);
	delete [] hPos; 
	delete [] hrPos; 	
	delete [] htemp; 
	delete [] hacc; 	
	delete [] hHash; 
	delete [] hIndex; 
	delete psystem;
}

BOOST_AUTO_TEST_CASE(SystemOutput)
{
	/*mytest(0.0225f);
	mytest(0.045f);
	mytest(0.1125f);
	mytest(0.225f);
	mytest(1.0f);*/
	//mytest(2.0f);
	//mytest(3.0f);
}

BOOST_AUTO_TEST_SUITE_END()
