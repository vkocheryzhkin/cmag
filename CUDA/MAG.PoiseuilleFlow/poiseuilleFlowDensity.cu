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

					int3 shift = make_int3(EvaluateShift(gridPos.x, cfg.gridSize.x),
											EvaluateShift(gridPos.y, cfg.gridSize.y),
											EvaluateShift(gridPos.z, cfg.gridSize.z));					

					float3 relPos = make_float3(pos.x - (pos2.x + shift.x * cfg.worldSize.x),
												 pos.y - (pos2.y + shift.y * cfg.worldSize.y),
												 pos.z - (pos2.z + shift.z * cfg.worldSize.z)); 
						
					float dist = length(relPos);
					float q = dist / cfg.smoothingRadius;					
				
					float coeff = 7.0f / 4 / CUDART_PI_F / powf(cfg.smoothingRadius, 2);					
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
		for(int z=-cfg.cellcount; z<=cfg.cellcount; z++) {
			for(int y=-cfg.cellcount; y<=cfg.cellcount; y++) {
				for(int x=-cfg.cellcount; x<=cfg.cellcount; x++) {
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
		float dens = sum * cfg.particleMass;		
		measures[index] = make_float4(
			dens,							
			/*cfg.restDensity * powf(cfg.soundspeed,2) / 7 * 
			(powf(dens / cfg.restDensity, 7) - 1),*/
			//powf(cfg.soundspeed, 2) * dens,
			50 * powf(cfg.soundspeed, 1) * dens,
			//powf(cfg.soundspeed, 1) * dens,
			0,
			pos.w);
}
