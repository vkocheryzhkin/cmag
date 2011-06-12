#include <boost/test/unit_test.hpp>
#include <boost/format.hpp>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <iostream>
#include <math.h>
#include "poiseuilleFlowSystem.cuh"
#include "poiseuilleFlowSystem.h"
#include "magUtil.cuh"
using namespace std;
using namespace thrust;

BOOST_FIXTURE_TEST_SUITE(AvgDensity_Suite)
BOOST_AUTO_TEST_CASE(AvgDensityTest)
{	
	int boundary_offset = 3;		
	/*uint3 gridSize = make_uint3(64, 64, 4);   
	uint3 fluidParticlesSize = make_uint3(64, 64 -  2 * boundary_offset, 1);*/
	uint3 gridSize = make_uint3(8, 16, 4);   			
	uint3 fluidParticlesSize = make_uint3(8, 8 -  2 * boundary_offset, 1);
	float soundspeed = powf(10.0f, -4.0f);															
	float radius = 1.0f / (2 * (64 - 6) * 1000);							
	float3 gravity = make_float3(0,0,0);
	float amplitude = 6 * radius;
	float sigma = (64 / 32) * CUDART_PI_F / ((fluidParticlesSize.x - 1) * 2 * radius);		
	float frequency = 100 * soundspeed * sigma;

	float delaTime = powf(10.0f, -4.0f);
	PoiseuilleFlowSystem* psystem = new PoiseuilleFlowSystem(
		delaTime,
		fluidParticlesSize,					
		amplitude,
		sigma,
		frequency,
		soundspeed,
		gravity,
		boundary_offset, 
		gridSize,								
		radius,
		false);	

	uint numParticles = psystem->getNumParticles();		
	psystem->reset();				
	psystem->Update();					
	
	host_vector<float4> h_position(numParticles);	
	host_vector<uint> h_index(numParticles);
	host_vector<float4> h_density(numParticles);

	device_ptr<float4> position((float4*)psystem->getCudaPosVBO());	
	device_ptr<uint> index((uint*)psystem->getCudaIndex());
	device_ptr<float4> density((float4*)psystem->getMeasures());								

	FILE *fp1= fopen("AvgDensityTest.csv", "w");
	
	thrust::copy(position, position + numParticles, h_position.begin());	
	thrust::copy(index, index + numParticles, h_index.begin());	
	thrust::copy(density, density + numParticles, h_density.begin());	
		
	for(int i = 0; i < numParticles; i++){
		float type = h_position[h_index[i]].w;
		if(type == 0)
			fprintf(fp1, "%f %f \n", 
				h_position[h_index[i]].w,
				h_density[i].x);
	}				
	fclose(fp1);
	delete psystem;
}

BOOST_AUTO_TEST_SUITE_END()