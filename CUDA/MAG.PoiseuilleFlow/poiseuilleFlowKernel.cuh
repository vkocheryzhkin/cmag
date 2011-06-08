#ifndef __POISEUILLEFLOW_KERNEL_CUH__
#define __POISEUILLEFLOW_KERNEL_CUH__
#include "vector_types.h"
#ifndef __DEVICE_EMULATION__
#define USE_TEX 1
#endif

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

typedef unsigned int uint;

struct PoiseuilleParams {     
	uint3 gridSize;
	float3 worldOrigin;
	float3 cellSize;
	float3 worldSize;
	uint3 fluidParticlesSize;
	int cellcount; //how many neigbours cells to look at

	float3 gravity;    
	float particleRadius;        		        
	float smoothingRadius;
	float particleMass;
	float restDensity;
	float soundspeed;	
		
	float deltaTime;		
	float boundaryDamping;	
	float mu;

	int boundaryOffset;
	float amplitude;
	float frequency;
	float sigma;
	bool IsBoundaryMotion;	


	float B;
	float gamma;

	__host__ __device__
	float PoiseuilleParams::BoundaryHeight()	{		
		return boundaryOffset * 2.0f * particleRadius;
	}
	__host__ __device__
	float PoiseuilleParams::FluidHeight()	{		
		return fluidParticlesSize.y * 2.0f * particleRadius;
	}
};
#endif//__POISEUILLEFLOW_KERNEL_CUH__
