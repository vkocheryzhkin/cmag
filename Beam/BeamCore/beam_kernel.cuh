 #ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#ifndef __DEVICE_EMULATION__
#define USE_TEX 1
#endif

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include "vector_types.h"
#include "vector_functions.h"
typedef unsigned int uint;
#define CUDART_PI_F           3.141592654f

struct SimParams 
{        
    float particleRadius;
	float smoothingRadius; 
	float deltaTime;
	float particleMass;
	float3 gravity;

	float3 worldOrigin;
	uint3 gridSize;
	float3 cellSize;

	//Kernels
	float Poly6Kern;
	float SpikyKern;
	//Material
	float Young;
	float Poisson;
	float restDensity;
};


#endif
