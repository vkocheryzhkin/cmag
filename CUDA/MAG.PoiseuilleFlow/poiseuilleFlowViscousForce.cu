#include "cutil_math.h"
#include "poiseulleFlowUtil.cu"

__device__ float4 getVelocityDiff(
	float4 iVelocity, 
	float4 iPosition, 
	float4 jVelocity,
	float4 jPosition,
	float elapsedTime)
{		
	/*float bottomBoundary = params.worldOrigin.y + params.boundaryOffset * 2.0f * params.particleRadius + params.amplitude;	
	float topBoundary = bottomBoundary + params.fluidParticlesSize.y * 2.0f * params.particleRadius ;		
	if(jPosition.w < 0.0f)
	{
		float distanceA = topBoundary - iPosition.y;
		float distanceB = jPosition.y - topBoundary;
		float beta = fmin(1000.0f, 1 + distanceB / distanceA);
		return beta * iVelocity;
	}
	
	if(jPosition.w > 0.0f)
	{
		float distanceA = iPosition.y - bottomBoundary;
		float distanceB = bottomBoundary - jPosition.y;
		float beta = fmin(1000.0f, 1 + distanceB / distanceA);
		return beta * iVelocity;
	}*/
	return iVelocity - jVelocity;	
}

__device__ float3 sumViscosity(
	int3    gridPos,
	uint    index,
	float4  pos,
	float4* oldPos, 
	float4  vel,
	float4* oldVel,
	float density,
	float4* oldMeasures,
	uint*   cellStart,
	uint*   cellEnd,
	float elapsedTime){
		uint gridHash = calcGridHash(gridPos);
		int3 shift = make_int3(EvaluateShift(gridPos.x, params.gridSize.x),
			EvaluateShift(gridPos.y, params.gridSize.y),
			EvaluateShift(gridPos.z, params.gridSize.z));							

		uint startIndex = FETCH(cellStart, gridHash);	    
		float3 force = make_float3(0.0f);
		if (startIndex != 0xffffffff) {               
			uint endIndex = FETCH(cellEnd, gridHash);
			for(uint j=startIndex; j<endIndex; j++) {
				if (j != index) {             
					float4 pos2 = FETCH(oldPos, j);
					float4 vel2 = FETCH(oldVel, j);
					float4 measure = FETCH(oldMeasures, j);
					float density2 = measure.x;

					float3 relPos = make_float3(pos.x - (pos2.x + shift.x * params.worldSize.x),
						pos.y - (pos2.y + shift.y * params.worldSize.y),
						pos.z - (pos2.z + shift.z * params.worldSize.z));
					  										
					float dist = length(relPos);
					float q = dist / params.smoothingRadius;									

					float coeff = 7.0f / (2 * CUDART_PI_F * powf(params.smoothingRadius, 3));
					float4 Vab = getVelocityDiff(vel, pos, vel2, pos2, elapsedTime);
					if(q < 2){
						float temp = coeff * (-powf(1 - 0.5f * q,3) * (2 * q + 1) + powf(1 - 0.5f * q, 4));
						force += params.particleMass * (params.mu + params.mu) * 
							make_float3(Vab) / (density * density2 *  dist) * temp;							
					}
				}
			}
		}
		return force;				
}

__global__ void computeViscousForceD(
	float4* viscousForce,	
	float4* oldMeasures,
	float4* oldPos,			
	float4* oldVel,
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles,
	float elapsedTime){
		uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;    

		float4 pos = FETCH(oldPos, index);
		float4 vel = FETCH(oldVel, index);
		float4 measure = FETCH(oldMeasures,index);
		float density = measure.x;

		int3 gridPos = calcGridPos(make_float3(pos));

		float3 force = make_float3(0.0f);		
		for(int z=-params.cellcount; z<=params.cellcount; z++) {
			for(int y=-params.cellcount; y<=params.cellcount; y++) {
				for(int x=-params.cellcount; x<=params.cellcount; x++) {
					int3 neighbourPos = gridPos + make_int3(x, y, z);
					force += sumViscosity(
						neighbourPos, 
						index, 
						pos, 
						oldPos,
						vel,
						oldVel,
						density,
						oldMeasures,
						cellStart, 
						cellEnd,
						elapsedTime);
				}
			}
		}
		uint originalIndex = gridParticleIndex[index];							
		viscousForce[originalIndex] = make_float4(force, 0.0f);
}