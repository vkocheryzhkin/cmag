#ifndef PERISTALSIS_KERNEL_CUH_
#define PERISTALSIS_KERNEL_CUH_
#include "vector_types.h"
#include <math.h>

#ifndef __DEVICE_EMULATION__
#define USE_TEX 1
#endif

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

typedef unsigned int uint;

struct Peristalsiscfg {     
	uint3 gridSize;
	float3 worldOrigin;
	float3 cellSize;
	float3 worldSize;
	uint3 fluid_size;
	int cellcount; //how many neigbours cells to look at

	float3 gravity;    
	float radius;        		        
	float smoothingRadius;
	float particleMass;
	float restDensity;
	float soundspeed;	
		
	float deltaTime;		
	float boundaryDamping;	
	float mu;

	int boundaryOffset;
	float amplitude;
	float wave_speed;	
	bool IsBoundaryConfiguration;	


	float B;
	float gamma;

	__host__ __device__
	float Peristalsiscfg::BoundaryHeight()	{		
		return boundaryOffset * 2.0f * radius;
	}
	__host__ __device__
	float Peristalsiscfg::FluidHeight()	{		
		return fluid_size.y * 2.0f * radius;
	}

	__device__ float GetWave(float x, float t){
		return sinf(CUDART_PI_F / (fluid_size.x * radius)
			*((x - worldOrigin.x) - wave_speed * t));
	}
};
#endif//PERISTALSIS_KERNEL_CUH_
