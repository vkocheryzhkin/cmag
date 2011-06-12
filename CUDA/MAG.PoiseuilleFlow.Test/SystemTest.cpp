#define BOOST_TEST_MODULE Default
#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "poiseuilleflowsystem.cuh"
#include "poiseuilleflowsystem.h"
#include "magutil.cuh"

typedef unsigned int uint;
using namespace std;

BOOST_FIXTURE_TEST_SUITE(SystemOutput)

void mytest(float recordTime)
{	
	std::string fileNameEnding = str(boost::format("%1%") % recordTime);	
	std::string fileName = str(boost::format("PoiseuilleFlowOutput%1%") % fileNameEnding.replace(1,1,"x")); 
	FILE *fp1= fopen(fileName.c_str(), "w");
	int boundaryOffset = 3;
	int sizex = 64;
	float soundspeed = powf(10.0f, -4.0f);
	float3 gravity = make_float3(powf(10.0f, -4.0f), 0.0f, 0.0f); 
	//float3 gravity = make_float3(0.0f, 0.0f, 0.0f); 
	float radius = 1.0f / (2 * (sizex - 2 * boundaryOffset) * 1000);
	uint3 gridSize = make_uint3(sizex, 64, 4);   
	uint3 fluidParticlesSize = make_uint3(gridSize.x, gridSize.y -  2 * boundaryOffset, 1);
	float amplitude =0;// 6 * radius;
	float sigma = 0;//(sizex / 32) * CUDART_PI_F / ((fluidParticlesSize.x - 1) * 2 * radius);		
	float frequency = 0;// soundspeed * sigma ;
	float delaTime = powf(10.0f, -4.0f);
	PoiseuilleFlowSystem *psystem = new PoiseuilleFlowSystem(
			delaTime,
			fluidParticlesSize,			
			0,0,0,
			/*amplitude,
			sigma,
			frequency,*/
			soundspeed,
			gravity,
			boundaryOffset, 
			gridSize,								
			radius,
			false);					
		
	psystem->reset();		

	float *hPos = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());
	float *hrPos = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());	
	float *htemp = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());		
	float *hacc = (float*)malloc(sizeof(float)*4*psystem->getNumParticles());		
	uint* hHash = new uint[psystem->getNumParticles()];
	uint* hIndex = new uint[psystem->getNumParticles()];		

	while(psystem->getElapsedTime() < recordTime)
		psystem->Update();	
				
	copyArrayFromDevice(hPos,psystem->getCudaPosVBO(),0, sizeof(float)*4*psystem->getNumParticles());	
	copyArrayFromDevice(htemp,psystem->getLeapFrogVelocity(),0, sizeof(float)*4*psystem->getNumParticles());		
	copyArrayFromDevice(hHash,psystem->getCudaHash(),0, sizeof(uint)*psystem->getNumParticles());
	copyArrayFromDevice(hIndex,psystem->getCudaIndex(),0, sizeof(uint)*psystem->getNumParticles());			

	fprintf(fp1, "%f %f \n", 0.0f, 0.0f);
	int cx = 0;
	int tt = 0;
	for(int i = 0; i < (psystem->getNumParticles()); i++) 
	{		
		float posx = hPos[4*hIndex[i] + 0];		
		float posy = hPos[4*hIndex[i] + 1];	
		float bottom = psystem->getWorldOrigin().y + 2 * psystem -> getParticleRadius() * boundaryOffset + amplitude;
		if((posx > 0) 
			&& (posx <  2 * psystem -> getParticleRadius())
			&& (posy > bottom)
			&& (posy < bottom + fluidParticlesSize.y * 2.0f * radius)
			)
		{			
			fprintf(fp1, "%1.16f %1.16f \n", htemp[4*hIndex[i]+0], hPos[4*hIndex[i]+1]
			- psystem->getWorldOrigin().y
			- boundaryOffset * 2 *psystem -> getParticleRadius()
			- amplitude);
			//fprintf(fp1, "%d %d \n", cx,tt);
			tt++;
		}
		cx++;
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

BOOST_AUTO_TEST_CASE(DumpVelocityProfile)
{
	/*mytest(0.0225f);
	mytest(0.045f);
	mytest(0.1125f);
	mytest(0.225f);
	mytest(1.0f);*/	
}

BOOST_AUTO_TEST_SUITE_END()
