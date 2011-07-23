#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "poiseuilleFlowSystem.cuh"
#include "poiseuilleFlowSystem.h"
#include <fstream>
#include "util.h"
using namespace std;
using namespace thrust;

void density_avg(){	
	int boundary_offset = 3;		
	uint3 gridSize = make_uint3(256, 128, 4);   
	uint3 fluid_size = make_uint3(256, 64 -  2 * boundary_offset, 1);	
	float soundspeed = powf(10.0f, -4.0f);
	float radius = 1.0f / (2 * (64 - 6) * 1000);							
	float3 gravity = make_float3(0,0,0);
	float amplitude = 0.6 * 35 * radius;		
	float wave_speed = 100 * soundspeed;
	float delaTime = powf(10.0f, -4.0f);
	PoiseuilleFlowSystem* psystem = new PoiseuilleFlowSystem(
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

	host_vector<float4> h_position(numParticles);	
	host_vector<uint> h_index(numParticles);
	host_vector<float4> h_density(numParticles);

	device_ptr<float4> position((float4*)psystem->getCudaPosVBO());	
	device_ptr<uint> index((uint*)psystem->getCudaIndex());
	device_ptr<float4> density((float4*)psystem->getMeasures());

	ofstream fp1;	
	string name = "density.dat";
	backup(name);
	fp1.open(name.c_str());
		
	while(psystem->GetElapsedTime() < 1.0f){
		psystem->Update();	
														
		thrust::copy(position, position + numParticles, h_position.begin());	
		thrust::copy(index, index + numParticles, h_index.begin());	
		thrust::copy(density, density + numParticles, h_density.begin());				
		
		float temp = 0.0f;
		for(int i = 0; i < numParticles; i++){
			float type = h_position[h_index[i]].w;
			if(type == 0)
				temp += h_density[i].x;
		}		
		float avg = temp / (fluid_size.x * fluid_size.y);
				
		fp1 << psystem->GetElapsedTime() << " " << avg << endl;
	}	
	fp1.close();
	delete psystem;
}