#include "peristalsisUtil.cu"

__device__ float3 sumPressure(
	int3    gridPos,
	uint    index,
	float4  pos,
	float4* oldPos, 	
	float density,
	float pressure,				   
	float4* oldMeasures,
	uint*   cellStart,
	uint*   cellEnd,
	float elapsedTime){
		uint gridHash = calcGridHash(gridPos);
		uint startIndex = FETCH(cellStart, gridHash);
	    
		float3 force = make_float3(0.0f);
		if (startIndex != 0xffffffff) {               
			uint endIndex = FETCH(cellEnd, gridHash);
			for(uint j=startIndex; j<endIndex; j++) {
				if (j != index) {             
					float4 pos2 = FETCH(oldPos, j);						
					float4 measure = FETCH(oldMeasures, j);
					float density2 = measure.x;
					float pressure2 = measure.y;				

					int3 shift = make_int3(EvaluateShift(gridPos.x, cfg.gridSize.x),
											EvaluateShift(gridPos.y, cfg.gridSize.y),
											EvaluateShift(gridPos.z, cfg.gridSize.z));					

					float3 relPos = make_float3(pos.x - (pos2.x + shift.x * cfg.worldSize.x),
												 pos.y - (pos2.y + shift.y * cfg.worldSize.y),
												 pos.z - (pos2.z + shift.z * cfg.worldSize.z));  
										
					float dist = length(relPos);
					float q = dist / cfg.smoothingRadius;									

					float coeff = 7.0f / (2 * CUDART_PI_F * pow(cfg.smoothingRadius, 3));
					if(q < 2){
						float temp = coeff * (-pow(1 - 0.5f * q,3) * (2 * q + 1) + pow(1 - 0.5f * q, 4));
						force += -1.0f * cfg.particleMass * temp *
							(pressure / powf(density,2) + pressure2 / powf(density2,2)) * 
							normalize(relPos);						
					}
				}
			}
		}
		return force;				
}

__global__ void computePressureForceD(
	float4* pressureForce,			
	float4* oldMeasures,
	float4* oldPos,				
	uint* gridParticleIndex,
	uint* cellStart,
	uint* cellEnd,
	uint numParticles,
	float elapsedTime){
		uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;    

		float4 pos = FETCH(oldPos, index);		
		float4 measure = FETCH(oldMeasures,index);
		float density = measure.x;
		float pressure = measure.y;

		int3 gridPos = calcGridPos(make_float3(pos));

		float3 force = make_float3(0.0f);		
		for(int z=-cfg.cellcount; z<=cfg.cellcount; z++) {
			for(int y=-cfg.cellcount; y<=cfg.cellcount; y++) {
				for(int x=-cfg.cellcount; x<=cfg.cellcount; x++) {
					int3 neighbourPos = gridPos + make_int3(x, y, z);
					force += sumPressure(
						neighbourPos, 
						index, 
						pos, 
						oldPos,						
						density,
						pressure,					
						oldMeasures,
						cellStart, 
						cellEnd,
						elapsedTime);
				}
			}
		}
		uint originalIndex = gridParticleIndex[index];							
		pressureForce[originalIndex] = make_float4(force, 0.0f);									
}