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
	int num = 128;
	uint3 fluidParticlesSize = make_uint3(num, num, 1);	    
	uint3 gridSize = make_uint3(4 * num, 2 * num, 4);   			
	float radius = 1.0f / (2 * num);
	int boundaryOffset = 1;

	DamBreakSystem *psystem = new DamBreakSystem(fluidParticlesSize, boundaryOffset, gridSize, radius, false); 
	psystem->reset();

	//relax system
	while(psystem->getElapsedTime() < 1.5f)
		psystem->update();

	cout << "Benchmark is running" << endl;
	psystem->changeRightBoundary();
	
	unsigned int start = clock();

	while(psystem->getElapsedTime() < 2.0f)
		psystem->update();	

	cout << "Time taken in millisecs: " << clock() - start;
		
	delete psystem;	
}

