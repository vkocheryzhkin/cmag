#include <vector_types.h>
#include <vector_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <stack>
#include <math.h>
//#include <iostream>
#include <fstream>
#include <sstream>
#include "peristalsisSystem.cuh"
#include "peristalsisSystem.h"

using namespace std;
using namespace thrust;

void velocity_filed(){
	int boundary_offset = 3;		
	uint3 gridSize = make_uint3(256, 128, 4);   
	uint3 fluid_size = make_uint3(256, 64 -  2 * boundary_offset, 1);	
	float soundspeed = powf(10.0f, -4.0f);
	float radius = 1.0f / (2 * (64 - 6) * 1000);							
	float3 gravity = make_float3(0,0,0);
	float amplitude = 0.6 * 35 * radius;		
	float wave_speed = 100 * soundspeed;
	float delaTime = powf(10.0f, -4.0f);
	PeristalsisSystem* psystem = new PeristalsisSystem(
		delaTime,
		fluid_size,			
		//0,0,0,
		amplitude,			
		wave_speed,
		soundspeed,
		gravity,
		boundary_offset, 
		gridSize,								
		radius,
		false);					

	uint numParticles = psystem->getNumParticles();		
	psystem->Reset();		

		
	std::stack<float> timeFrames;				
	timeFrames.push(0.4);
	timeFrames.push(0.3);
	
	while (!(timeFrames.empty())){
		float timeSlice = timeFrames.top();
		timeFrames.pop();

		while(psystem->GetElapsedTime() < timeSlice)
			psystem->Update();	

		thrust::device_ptr<float4> dev_positions((float4*)psystem->getCudaPosVBO());	
		thrust::device_ptr<float4> dev_velocities((float4*)psystem->getCudaVelVBO());	
		thrust::host_vector<float4> h_positions(psystem->getNumParticles());
		thrust::host_vector<float4> h_velocities(psystem->getNumParticles());
		thrust::copy(dev_positions, dev_positions + psystem->getNumParticles(), h_positions.begin());		
		thrust::copy(dev_velocities, dev_velocities + psystem->getNumParticles(), h_velocities.begin());


		ostringstream buffer;	
		buffer << timeSlice;
		string str = "velocity_field" + buffer.str().replace(1,1,"x") + ".dat";
		ofstream fp1;	
		fp1.open(str.c_str());

		/*std::string fileNameEnding = str(boost::format("%1%") % timeSlice);	
		std::string fileName = str(boost::format("VectorField%1%") % fileNameEnding.replace(1,1,"x")); 
		FILE *file= fopen(fileName.c_str(), "w");*/
		for (int i = 0; i < h_positions.size(); i+=4){
			float4 p = (float4)h_positions[i];
			float4 v = (float4)h_velocities[i];
			
			//fprintf(file, "%f %f %f %f \n", p.x, p.y -amplitude, v.x, v.y);
			fp1 << p.x << " " << p.y << " "
				<< v.x << " " << v.y << endl;
		}		
		//fclose(file);	
		fp1.close();
	}	
	delete psystem;	
}

