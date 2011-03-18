#define BOOST_TEST_MODULE ModuleStart
#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>
#include <vector_types.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>
 #include<stack>

typedef unsigned int uint;
#include "fluidSystem.cuh"
#include "fluidSystem.h"
#include "magUtil.cuh"

using namespace std;

BOOST_AUTO_TEST_CASE(XFrontOutput)
{			
	FILE *file= fopen("XFrontOutput", "w");

	int num = 32;
	uint3 fluidParticlesSize = make_uint3(num, 2 * num, 1);	    
	uint3 gridSize = make_uint3(128, 64, 4);   		
	float radius = 1.0f / 64;

	DamBreakSystem *psystem = new DamBreakSystem(fluidParticlesSize, gridSize, radius, false); 
	psystem->reset();
	
	struct compare_float4{
		__host__ bool operator()(float4 a, float4 b){
			return a.x > b.x;
		}
	};
	compare_float4 comparator;

	//Table 2. An Experimental Study o f the Collapse of Liquid Columns on a Rigid Horizontal
	//Plane.  J. C. Martin and W. J. Moyce
	stack<float2> timeFrames;				
	timeFrames.push(make_float2(3.11f, 3.89f));
	timeFrames.push(make_float2(2.97f, 3.67f));
	timeFrames.push(make_float2(2.83f, 3.44f));
	timeFrames.push(make_float2(2.65f, 3.22f));
	timeFrames.push(make_float2(2.51f, 3.00f));
	timeFrames.push(make_float2(2.32f, 2.78f));
	timeFrames.push(make_float2(2.20f, 2.56f));
	timeFrames.push(make_float2(1.98f, 2.33f));
	timeFrames.push(make_float2(1.83f, 2.11f));
	timeFrames.push(make_float2(1.63f, 1.89f));
	timeFrames.push(make_float2(1.43f, 1.67f));
	timeFrames.push(make_float2(1.19f, 1.44f));
	timeFrames.push(make_float2(0.84f, 1.22f));
	timeFrames.push(make_float2(0.41f, 1.11f));
	timeFrames.push(make_float2(0.00f, 1.0f));	

	float xwidth = 2 * radius * num;
	float timeScale = sqrt(2* abs(psystem->getGravity().y) / xwidth);	
	
	while (!(timeFrames.empty())){
		float2 expData = timeFrames.top();
		timeFrames.pop();

		while(psystem->getElapsedTime() * timeScale < expData.x)
			psystem->update();	

		thrust::device_ptr<float4> dev_ptr((float4*)psystem->getCudaPosVBO());	
		thrust::host_vector<float4> h_vec(psystem->getNumParticles());

		thrust::copy(dev_ptr, dev_ptr + psystem->getNumParticles(), h_vec.begin());		
		thrust::sort(h_vec.begin(),h_vec.end(),comparator);	

		volatile float x = ((float4)h_vec[0]).x;
		fprintf(file, "%f %f %f %f \n", 
			expData.x, //dimensionless experimental time
			psystem->getElapsedTime(), //real time
			expData.y, //dimensionless experimental width
			(x + radius - psystem->getWorldOrigin().x) / xwidth); //dimensional width
	}
	fclose(file);
	delete psystem;	
}
