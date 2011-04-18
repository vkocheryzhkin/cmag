//#include <boost/test/unit_test.hpp>
//#include <boost/format.hpp>
//#include <vector_types.h>
//#include <vector_functions.h>
//#include <thrust/device_ptr.h>
//#include <thrust/device_vector.h>
//#include <stack>
//#include <math.h>
//#include <iostream>
//#include "poiseuilleFlowSystem.cuh"
//#include "poiseuilleFlowSystem.h"
//#include "magUtil.cuh"
//
//BOOST_AUTO_TEST_CASE(VelocityField)
//{	
//	int boundaryOffset = 3;
//	float soundspeed = powf(10.0f, -1.0f);
//	float3 gravity = make_float3(0.0f, 0.0f, 0.0f); 
//	float radius = 1.0f / (2 * (64 - 2 * boundaryOffset) * 1000);
//	uint3 gridSize = make_uint3(64, 64, 4);   
//	uint3 fluidParticlesSize = make_uint3(gridSize.x, gridSize.y -  2 * boundaryOffset, 1);
//	float amplitude = 10 * radius;
//	float sigma = 2 * CUDART_PI_F / (fluidParticlesSize.x * 2 * radius);		
//	float omega = soundspeed * sigma / 50;	
//	float delaTime = powf(10.0f, -4.0f);
//
//	PoiseuilleFlowSystem *psystem = new PoiseuilleFlowSystem(
//		delaTime,
//		fluidParticlesSize,
//		amplitude,
//		sigma,
//		omega,
//		soundspeed,
//		gravity,
//		boundaryOffset, 
//		gridSize, 
//		radius,
//		false); 
//	psystem->reset();
//
//	psystem->setBoundaryWave();
//
//	//relax system
//	while(psystem->getElapsedTime() < 1.5f)
//		psystem->update();
//
//	std::cout << "System relaxed";
//
//	float period = 2 * CUDART_PI_F / omega;
//	psystem->StartBoundaryMotion();//reset timer here
//		
//	std::stack<float> timeFrames;				
//	/*timeFrames.push(0.0225f);
//	timeFrames.push(0.045f);
//	timeFrames.push(0.1125f);
//	timeFrames.push(0.225f);*/
//	
//	timeFrames.push(2.5 * period);
//	timeFrames.push(2 * period);
//	
//	while (!(timeFrames.empty())){
//		float timeSlice = timeFrames.top();
//		timeFrames.pop();
//
//		while(psystem->getElapsedTime() < timeSlice)
//			psystem->update();	
//
//		thrust::device_ptr<float4> dev_positions((float4*)psystem->getCudaPosVBO());	
//		thrust::device_ptr<float4> dev_velocities((float4*)psystem->getCudaVelVBO());	
//		thrust::host_vector<float4> h_positions(psystem->getNumParticles());
//		thrust::host_vector<float4> h_velocities(psystem->getNumParticles());
//		thrust::copy(dev_positions, dev_positions + psystem->getNumParticles(), h_positions.begin());		
//		thrust::copy(dev_velocities, dev_velocities + psystem->getNumParticles(), h_velocities.begin());		
//
//		std::string fileNameEnding = str(boost::format("%1%") % timeSlice);	
//		std::string fileName = str(boost::format("VectorField%1%") % fileNameEnding.replace(1,1,"x")); 
//		FILE *file= fopen(fileName.c_str(), "w");
//		for (int i = 0; i < h_positions.size(); i+=2){
//			float4 p = (float4)h_positions[i];
//			float4 v = (float4)h_velocities[i];
//			
//			fprintf(file, "%f %f %f %f \n", p.x, p.y -amplitude, v.x, v.y);
//		}		
//		fclose(file);	
//	}	
//	delete psystem;	
//}
//
