#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <fstream>
#include <math.h>
#include "poiseuilleFlowSystem.cuh"
#include "poiseuilleFlowSystem.h"
#include "util.h"
using namespace std;
using namespace thrust;

void forces_avg(){	
	int boundary_offset = 3;		
	uint3 gridSize = make_uint3(64, 128, 4);   
	uint3 fluid_size = make_uint3(64, 64 -  2 * boundary_offset, 1);
	float soundspeed = powf(10.0f, -4.0f);															
	float radius = 1.0f / (2 * (64 - 6) * 1000);							
	float3 gravity = make_float3(0,0,0);
	float amplitude = 6 * radius;	
	float sigma = (64 / 32) * CUDART_PI_F / (fluid_size.x * 2 * radius);
	float frequency = 100 * soundspeed * sigma;
	float delaTime = powf(10.0f, -4.0f);	
	uint numFluid = fluid_size.x * fluid_size.y;
	PoiseuilleFlowSystem* psystem = new PoiseuilleFlowSystem(
		delaTime,
		fluid_size,					
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
	psystem->Reset();					
	
	host_vector<float4> h_pressure(numParticles);
	host_vector<float4> h_viscosity(numParticles);
	host_vector<float4> h_position(numParticles);	
	host_vector<uint> h_index(numParticles);
	
	device_ptr<float4> pressure((float4*)psystem->pressure_force());	
	device_ptr<float4> viscosity((float4*)psystem->viscous_force());
	device_ptr<float4> position((float4*)psystem->getCudaPosVBO());	
	device_ptr<uint> index((uint*)psystem->getCudaIndex());

	ofstream f1,f2,f3,f4;	
	string names[4] = {"press_x.dat","press_y.dat","visc_x.dat","visc_y.dat"};	
	//backup(names,4);
	f1.open(names[0].c_str());
	f2.open(names[1].c_str());
	f3.open(names[2].c_str());
	f4.open(names[3].c_str());
	while(psystem->GetElapsedTime() < 1.0f){
		psystem->Update();
		thrust::copy(position, position + numParticles, h_position.begin());	
		thrust::copy(index, index + numParticles, h_index.begin());	
		thrust::copy(pressure, pressure + numParticles, h_pressure.begin());
		thrust::copy(viscosity, viscosity + numParticles, h_viscosity.begin());


		float avg_pres_x = 0.0f;
		float avg_pres_y = 0.0f;
		float avg_visc_x = 0.0f;
		float avg_visc_y = 0.0f;
		for(int i = 0; i < numParticles; i++){
			float type = h_position[h_index[i]].w;
			if(type == 0){
				avg_pres_x += h_pressure[i].x;
				avg_pres_y += h_pressure[i].y;
				avg_visc_x += h_viscosity[i].x;
				avg_visc_y += h_viscosity[i].y;
			}
		}						
		
		f1 << psystem->GetElapsedTime() << " " << avg_pres_x / numFluid << endl;
		f2 << psystem->GetElapsedTime() << " " << avg_pres_y / numFluid << endl;
		f3 << psystem->GetElapsedTime() << " " << avg_visc_x / numFluid << endl;
		f4 << psystem->GetElapsedTime() << " " << avg_visc_y / numFluid << endl;
	}	
	f1.close();
	f2.close();
	f3.close();
	f4.close();
	delete psystem;
}