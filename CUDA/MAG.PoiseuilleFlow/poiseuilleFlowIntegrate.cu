__global__ void computeCoordinatesD(
	float4* posArray,		 
	float4* velArray,		 
	float4* velLeapFrogArray, 
	float4* viscouseForce,	 
	float4* pressureForce,
	uint numParticles){
		uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;     		

		volatile float4 posData = posArray[index]; 	
		if(posData.w != 0.0f) return;//skip boundary particle

		volatile float4 velData = velArray[index];
		volatile float4 velLeapFrogData = velLeapFrogArray[index];
		volatile float4 viscouseData = viscouseForce[index];
		volatile float4 pressureData = pressureForce[index];

		float3 pos = make_float3(posData.x, posData.y, posData.z);
		float3 vel = make_float3(velData.x, velData.y, velData.z);
		float3 vis = make_float3(viscouseData.x, viscouseData.y, viscouseData.z);
		float3 pres = make_float3(pressureData.x, pressureData.y, pressureData.z);
				

		float3 nextVel = vel + (params.gravity + vis + pres) * params.deltaTime;		
		//float3 nextVel = vel + (params.gravity) * params.deltaTime;		

		float3 velLeapFrog = vel + nextVel;
		velLeapFrog *= 0.5f;

		vel = nextVel;   	
		pos += vel * params.deltaTime;   

		float halfWorldXSize = params.gridSize.x * params.particleRadius;			
		if(pos.x > halfWorldXSize){
			pos.x -= 2 * halfWorldXSize;
		}
				  
		posArray[index] = make_float4(pos, posData.w);
		velArray[index] = make_float4(vel, velData.w);
		velLeapFrogArray[index] = make_float4(velLeapFrog, velLeapFrogData.w);
}