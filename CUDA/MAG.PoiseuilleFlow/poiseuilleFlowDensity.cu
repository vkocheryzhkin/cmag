#include "poiseulleFlowUtil.cu"

__device__ float sumDensity(
	int3    gridPos,
	uint    index,
	float4  pos,
	float4* oldPos, 
	uint*   cellStart,
	uint*   cellEnd){
		uint gridHash = calcGridHash(gridPos);
		uint startIndex = FETCH(cellStart, gridHash);

		float sum = 0.0f;
		if (startIndex != 0xffffffff) {        
			uint endIndex = FETCH(cellEnd, gridHash);
			for(uint j=startIndex; j<endIndex; j++) {				  
					float4 pos2 = FETCH(oldPos, j);				

					int3 shift = make_int3(EvaluateShift(gridPos.x, params.gridSize.x),
											EvaluateShift(gridPos.y, params.gridSize.y),
											EvaluateShift(gridPos.z, params.gridSize.z));					

					float3 relPos = make_float3(pos.x - (pos2.x + shift.x * params.worldSize.x),
												 pos.y - (pos2.y + shift.y * params.worldSize.y),
												 pos.z - (pos2.z + shift.z * params.worldSize.z)); 
						
					float dist = length(relPos);
					float q = dist / params.smoothingRadius;					
				
					float coeff = 7.0f / 4 / CUDART_PI_F / powf(params.smoothingRadius, 2);					
					if(q < 2){									
						sum += coeff *(powf(1 - 0.5f * q, 4) * (2 * q + 1));
					}				
			}
		}
		return sum;
}

__global__ void computeDensityVariationD(			
	float4* measures,
	float4* oldMeasures,
	float4* oldPos,
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles){
		uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;    

		float4 pos = FETCH(oldPos, index);		
		float4 vel = FETCH(oldVel, index);		
		int3 gridPos = calcGridPos(make_float3(pos));

		float sum = 0.0f;
		for(int z=-params.cellcount; z<=params.cellcount; z++) {
			for(int y=-params.cellcount; y<=params.cellcount; y++) {
				for(int x=-params.cellcount; x<=params.cellcount; x++) {
					int3 neighbourPos = gridPos + make_int3(x, y, z);
					sum += sumDensity(
							neighbourPos,
							index,
							pos,
							oldPos,
							cellStart,
							cellEnd);
				}
			}
		}					
		float dens = sum * params.particleMass;		
		measures[index] = make_float4(
			dens,				
			10 * params.soundspeed * dens,
			//powf(params.soundspeed,2) * dens,
			0,0);
}
