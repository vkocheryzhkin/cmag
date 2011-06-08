//#include <boost/test/unit_test.hpp>
//#include <boost/format.hpp>
//#include <thrust/device_ptr.h>
//#include <thrust/device_vector.h>
//#include <vector_types.h>
//#include <vector_functions.h>
//#include <iostream>
//#include <math.h>
//#include "poiseuilleFlowSystem.cuh"
//#include "poiseuilleFlowSystem.h"
//#include "magUtil.cuh"
//using namespace std;
//using namespace thrust;
//
//BOOST_FIXTURE_TEST_SUITE(Forces_Suite)
//BOOST_AUTO_TEST_CASE(ForcesTest)
//{	
//	int boundary_offset = 3;		
//	uint3 gridSize = make_uint3(64, 64, 4);   
//	uint3 fluidParticlesSize = make_uint3(64, 64 -  2 * boundary_offset, 1);
//	float soundspeed = powf(10.0f, -4.0f);															
//	float radius = 1.0f / (2 * (64 - 6) * 1000);						
//	//float3 gravity = make_float3(1000 * powf(10.0, -4),0,0);								
//	float3 gravity = make_float3(0,0,0);
//	float amplitude = 6 * radius;
//	float sigma = (64 / 32) * CUDART_PI_F / ((fluidParticlesSize.x - 1) * 2 * radius);		
//	float frequency = soundspeed * sigma;
//
//	float delaTime = powf(10.0f, -4.0f);
//	PoiseuilleFlowSystem* psystem = new PoiseuilleFlowSystem(
//		delaTime,
//		fluidParticlesSize,					
//		amplitude,
//		sigma,
//		frequency,
//		soundspeed,
//		gravity,
//		boundary_offset, 
//		gridSize,								
//		radius,
//		false);	
//
//	uint numParticles = psystem->getNumParticles();		
//	psystem->reset();				
//	psystem->Update();				
//
//	psystem->SwitchBoundarySetup();
//	
//	host_vector<float4> h_pressure(numParticles);
//	host_vector<float4> h_viscosity(numParticles);
//	
//	device_ptr<float4> pressure((float4*)psystem->pressure_force());	
//	device_ptr<float4> viscosity((float4*)psystem->viscous_force());	
//					
//	struct sum_float4
//	{
//		__host__ float4 operator()(float4 a, float4 b)
//		{
//			return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
//		}
//	};
//
//	sum_float4 sum4;
//
//	FILE *fp1= fopen("ForcesTest.txt", "w");
//	while(psystem->getElapsedTime() < 0.3f){
//		psystem->Update();
//		thrust::copy(pressure, pressure + numParticles, h_pressure.begin());
//		thrust::copy(viscosity, viscosity + numParticles, h_viscosity.begin());
//
//		float4 avg_pressure = thrust::reduce(
//			h_pressure.begin(),h_pressure.end(),make_float4(0,0,0,0), sum4);
//
//		float4 avg_viscosity = thrust::reduce(
//			h_viscosity.begin(),h_viscosity.end(),make_float4(0,0,0,0), sum4);
//
//		fprintf(fp1, "%f %f %f %f \n", avg_pressure.x,  avg_pressure.y,
//			 avg_viscosity.x, avg_viscosity.y);		
//	}
//	fclose(fp1);
//	delete psystem;
//}
//
//BOOST_AUTO_TEST_SUITE_END()