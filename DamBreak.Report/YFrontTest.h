#include <vector_types.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>
#include<stack>

typedef unsigned int uint;
#include "fluidSystem.cuh"
#include "fluidSystem.h"

using namespace std;

void YFrontTest(){				
	int num = 128;
	uint3 fluidParticlesSize = make_uint3(num, 2 * num, 1);	    
	uint3 gridSize = make_uint3(512, 256, 4);   			
	float radius = 1.0f / (2 * num);
	int boundaryOffset = 1;

	DamBreakSystem *psystem = new DamBreakSystem(fluidParticlesSize, boundaryOffset, gridSize, radius, false); 
	psystem->reset();

	//relax system
	while(psystem->getElapsedTime() < 1.5f)
		psystem->update();
	cout << "YFront System relaxed" << endl;
	psystem->removeRightBoundary();
	
	struct compareFloat4Y{
		__host__ bool operator()(float4 a, float4 b){
			if(a.w == Fluid && b.w !=Fluid)
				return true;
			if(a.w != Fluid && b.w == Fluid)
				return false;
			if(a.w != Fluid && b.w !=Fluid)
				return false;
			return a.y > b.y;
		}
	};
	compareFloat4Y comparator;

	//Table 6 (n^2 = 2). An Experimental Study o f the Collapse of Liquid Columns on a Rigid Horizontal
	//Plane.  J. C. Martin and W. J. Moyce
	stack<float2> timeFrames;				
	timeFrames.push(make_float2(3.44f, 0.22f));	
	timeFrames.push(make_float2(3.06f, 0.28f));	
	timeFrames.push(make_float2(2.70f, 0.33f));		
	timeFrames.push(make_float2(2.45f, 0.39f));	
	timeFrames.push(make_float2(2.21f, 0.44f));	
	timeFrames.push(make_float2(2.00f, 0.50f));	
	timeFrames.push(make_float2(1.84f, 0.56f));	
	timeFrames.push(make_float2(1.66f, 0.61f));	
	timeFrames.push(make_float2(1.46f, 0.67f));	
	timeFrames.push(make_float2(1.28f, 0.72f));	
	timeFrames.push(make_float2(1.08f, 0.78f));	
	timeFrames.push(make_float2(0.93f, 0.83f));	
	timeFrames.push(make_float2(0.77f, 0.89f));	
	timeFrames.push(make_float2(0.56f, 0.94f));	
	timeFrames.push(make_float2(0.0f, 1.0f));	

	thrust::device_ptr<float4> dev_ptr((float4*)psystem->getCudaPosVBO());	
	thrust::host_vector<float4> h_vec(psystem->getNumParticles());
	thrust::copy(dev_ptr, dev_ptr + psystem->getNumParticles(), h_vec.begin());		
	thrust::sort(h_vec.begin(),h_vec.end(),comparator);		
	float yheight = ((float4)h_vec[0]).y + radius - psystem->getWorldOrigin().y;

	float timeScale = sqrt(2 * abs(psystem->getGravity().y) / yheight);	
	FILE *file= fopen("YFrontOutput", "w");
	while (!(timeFrames.empty())){
		float2 expData= timeFrames.top();
		timeFrames.pop();

		while(psystem->getElapsedTime() * timeScale < expData.x)
			psystem->update();	

		thrust::device_ptr<float4> dev_ptr((float4*)psystem->getCudaPosVBO());	
		thrust::host_vector<float4> h_vec(psystem->getNumParticles());

		thrust::copy(dev_ptr, dev_ptr + psystem->getNumParticles(), h_vec.begin());		
		thrust::sort(h_vec.begin(),h_vec.end(),comparator);	

		float y = ((float4)h_vec[0]).y;
		fprintf(file, "%f %f %f %f \n",
			expData.x, //dimensionless experimental time
			psystem->getElapsedTime(), //real time
			expData.y, //dimensionless experimental height
			(y + radius - psystem->getWorldOrigin().y) / yheight ); //dimensional height
	}
	fclose(file);
	delete psystem;	
}
