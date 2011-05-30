#include "poiseulleFlowUtil.cu"

__device__ float4 sumDensity(
	int3    gridPos,
	uint    index,
	float4  pos,
	float4* oldPos, 
	uint*   cellStart,
	uint*   cellEnd){
		uint gridHash = calcGridHash(gridPos);
		uint startIndex = FETCH(cellStart, gridHash);

		float4 sum = make_float4(0.0f);
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
					//float gradCoeff = 7.0f / 2 / CUDART_PI_F / powf(params.smoothingRadius, 3);
					if(q < 2){
						/*float3 temp = gradCoeff * (-powf(1 - 0.5f * q,3) * (2 * q + 1) + powf(1 - 0.5f * q, 4))
							* normalize(relPos);*/
						//if(j==index) temp = make_float3(0.0f);
						/*sum += make_float4(
							coeff *(powf(1 - 0.5f * q, 4) * (2 * q + 1)),
							temp.x,
							temp.y,
							dot(temp, temp));	*/					
						sum += make_float4(
							coeff *(powf(1 - 0.5f * q, 4) * (2 * q + 1)),
							0,
							0,
							0);	
					}		

					/*if(q < 2)
						sum += make_float4(1.0f);	*/				
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
		//if(pos.w != 0.0f){
		//	//measures[index] = make_float4(params.restDensity, 0, 0, 0);
		//	measures[index] = make_float4(1, 0, 0, 0);
		//	return;
		//}

		float4 vel = FETCH(oldVel, index);		
		int3 gridPos = calcGridPos(make_float3(pos));

		float4 sum = make_float4(0.0f);		
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
		float4 tt =  FETCH(oldMeasures, index);
		float dens = sum.x * params.particleMass;
		float variation = dens - params.restDensity;
		float beta =0.5f * powf(params.deltaTime * params.particleMass / params.restDensity, 2);			
		measures[index] = make_float4(
			dens,				
			powf(params.soundspeed, 2) * dens,
			//params.B * (powf(dens / params.restDensity ,params.gamma) - 1.0f),
			//tt.y - 1.0f / (beta * (-dot(make_float2(sum.y, sum.z), make_float2(sum.y,sum.z)) - sum.w)) * variation,			 
			variation,
			sum.x);
}
