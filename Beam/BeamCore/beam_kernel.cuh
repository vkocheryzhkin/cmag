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

struct float12
{
  float4 u,v,w;
};
static __inline__ __device__ float12 make_float12(float4 u, float4 v, float4 w)
{
  float12 t; t.u = u; t.v = v; t.w = w; return t;
};
static __inline__ __device__ float12 make_float12()
{
  float12 t; t.u = make_float4(0,0,0,0); t.v = make_float4(0,0,0,0); t.w = make_float4(0,0,0,0); return t;
};

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
};

#endif
