#include <boost/test/unit_test.hpp>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iostream>
#include <ctime>

typedef unsigned int uint;
#include "fluidSystem.cuh"
#include "fluidSystem.h"
#include "magUtil.cuh"

using namespace std;

BOOST_AUTO_TEST_CASE(Benchmark)
{				
	int num = 32;
	uint3 fluidParticlesSize = make_uint3(num, num, 1);	    
	uint3 gridSize = make_uint3(4 * num, 2 * num, 4);   			
	float radius = 1.0f / (2 * num);
	int boundaryOffset = 0;	

	DamBreakSystem *psystem = new DamBreakSystem(fluidParticlesSize, boundaryOffset, gridSize, radius, false); 
	psystem->reset();

	psystem->update();
	thrust::device_ptr<float4> dev_ptr((float4*)psystem->getCudaMeasures());	
	thrust::host_vector<float4> h_vec(psystem->getNumParticles());

	thrust::copy(dev_ptr, dev_ptr + psystem->getNumParticles(), h_vec.begin());		
	float avg = 0.0f;
	for (int i= 0; i < h_vec.size(); i++)
	{
		avg+= ((float4)h_vec[i]).x;
	}
	cout << avg / (num * num) << endl;
	delete psystem;	
}