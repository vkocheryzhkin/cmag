#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include "peristalsisKernel.cuh"

#if USE_TEX
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;
texture<float4, 1, cudaReadModeElementType> oldMeasuresTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif
__constant__ Peristalsiscfg cfg;

__device__ int3 calcGridPos(float3 p){
	int3 gridPos;
	gridPos.x = floor((p.x - cfg.worldOrigin.x) * 0.5f / cfg.radius);
	gridPos.y = floor((p.y - cfg.worldOrigin.y) * 0.5f / cfg.radius);
	gridPos.z = floor((p.z - cfg.worldOrigin.z) * 0.5f / cfg.radius);
	return gridPos;
}

__device__ uint calcGridHash(int3 gridPos){
	gridPos.x = gridPos.x & (cfg.gridSize.x-1);  
	gridPos.y = gridPos.y & (cfg.gridSize.y-1);
	gridPos.z = gridPos.z & (cfg.gridSize.z-1);        
	return __umul24(__umul24(gridPos.z, cfg.gridSize.y), cfg.gridSize.x) + __umul24(gridPos.y, cfg.gridSize.x) + gridPos.x;
}

__global__ void calculatePeristalsisHashD(
	uint*   gridParticleHash,  // output
	uint*   gridParticleIndex, // output
	float4* pos,               // input
	uint    numParticles){
		uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;
	    
		volatile float4 p = pos[index];

		int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
		uint hash = calcGridHash(gridPos);

		gridParticleHash[index] = hash;
		gridParticleIndex[index] = index;
}

__global__ void reorderPeristalsisDataD(
	uint*   cellStart,        // output
	uint*   cellEnd,          // output
	float4* sortedPos,        // output
	float4* sortedVel,        // output
	uint *  gridParticleHash, // input
	uint *  gridParticleIndex,// input
	float4* oldPos,           // input
	float4* oldVel,           // input
	uint    numParticles){
		extern __shared__ uint sharedHash[];    // blockSize + 1 elements
		uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
		
		uint hash;
		if (index < numParticles) {
			hash = gridParticleHash[index];

			sharedHash[threadIdx.x+1] = hash;

			if (index > 0 && threadIdx.x == 0)
			{
				sharedHash[0] = gridParticleHash[index-1];
			}
		}

		__syncthreads();
		
		if (index < numParticles) {
			if (index == 0 || hash != sharedHash[threadIdx.x])
			{
				cellStart[hash] = index;
				if (index > 0)
					cellEnd[sharedHash[threadIdx.x]] = index;
			}

			if (index == numParticles - 1)
			{
				cellEnd[hash] = index + 1;
			}

			uint sortedIndex = gridParticleIndex[index];
			float4 pos = FETCH(oldPos, sortedIndex);       
			float4 vel = FETCH(oldVel, sortedIndex);       

			sortedPos[index] = pos;
			sortedVel[index] = vel;
		}
}

__global__ void predictCoordinatesD(
	float4* predictedPosition,
	float4* predictedVelocity,
	float4* posArray,		 
	float4* velArray,		 	
	float4* viscouseForce,	 
	float4* pressureForce,
	uint numParticles){
		uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;     		

		volatile float4 posData = posArray[index]; 	
		volatile float4 velData = velArray[index];
		volatile float4 viscouseData = viscouseForce[index];
		volatile float4 pressureData = pressureForce[index];

		float3 pos = make_float3(posData.x, posData.y, posData.z);
		float3 vel = make_float3(velData.x, velData.y, velData.z);
		float3 vis = make_float3(viscouseData.x, viscouseData.y, viscouseData.z);
		float3 pres = make_float3(pressureData.x, pressureData.y, pressureData.z);
		
		if(posData.w !=0.0f){
			predictedPosition[index] = make_float4(pos, posData.w);
			predictedVelocity[index] = make_float4(0.0f);
			return;	
		}

		float3 nextVel = vel + (cfg.gravity + (vis + pres) ) * cfg.deltaTime;		
		
		pos += nextVel * cfg.deltaTime;   
		float halfWorldXSize = cfg.gridSize.x * cfg.radius;			

		if(pos.x > halfWorldXSize){
			pos.x -= 2 * halfWorldXSize;
		}
		if(pos.x < -halfWorldXSize){
			pos.x += 2 * halfWorldXSize;
		}
		  
		predictedPosition[index] = make_float4(pos, posData.w);		
		predictedVelocity[index] = make_float4(0.0f);
}



__global__ void configureBoundaryD(	
	float4* posArray,
	float currentWaveHeight,
	uint numParticles){
		uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;  

		volatile float4 posData = posArray[index]; 	

		if(posData.w > 0.0f){//bottom						
			posArray[index] = make_float4(
				posData.x,
				-1.0f * cfg.fluid_size.y * cfg.radius +
				currentWaveHeight * cfg.GetWave(posData.x, 0) 			
				- cfg.radius * (posData.w - 1.0f),
				posData.z,
				posData.w);									
		}
		if(posData.w < 0.0f){
			posArray[index] = make_float4(
				posData.x,
				cfg.fluid_size.y * cfg.radius - 
				currentWaveHeight * cfg.GetWave(posData.x, 0)+				  
				cfg.radius * (-posData.w - 1.0f),
				posData.z,
				posData.w);
		}
}
