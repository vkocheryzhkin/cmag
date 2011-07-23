__global__ void computeCoordinatesD(
	float4* posArray,		 
	float4* velArray,		 
	float4* velLeapFrogArray, 
	float4* viscouseForce,	 
	float4* pressureForce,
	float elapsedTime,
	uint numParticles){
		uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index >= numParticles) return;     		

		volatile float4 posData = posArray[index]; 	
		if(posData.w != 0.0f && cfg.IsBoundaryConfiguration) return;//skip boundary particle
		if(posData.w > 0.0f)//bottom
		{
			posArray[index] = make_float4(
				posData.x,
				-1.0f * cfg.fluid_size.y * cfg.radius +
				cfg.amplitude * cfg.GetWave(posData.x, elapsedTime)
				- cfg.radius * (posData.w - 1.0f),
				posData.z,
				posData.w);									
			return;
		}
		if(posData.w < 0.0f)//top
		{
			posArray[index] = make_float4(posData.x,
				cfg.fluid_size.y * cfg.radius - 
				cfg.amplitude * cfg.GetWave(posData.x, elapsedTime) + 
				cfg.radius * (-posData.w - 1.0f),
				posData.z,
				posData.w);		
			return;
		}

		volatile float4 velData = velArray[index];
		volatile float4 velLeapFrogData = velLeapFrogArray[index];
		volatile float4 viscouseData = viscouseForce[index];
		volatile float4 pressureData = pressureForce[index];

		float3 pos = make_float3(posData.x, posData.y, posData.z);
		float3 vel = make_float3(velData.x, velData.y, velData.z);
		float3 vis = make_float3(viscouseData.x, viscouseData.y, viscouseData.z);
		float3 pres = make_float3(pressureData.x, pressureData.y, pressureData.z);											

		float3 nextVel = vel + (cfg.gravity + vis + pres) * cfg.deltaTime;		
		//float3 nextVel = vel + (cfg.gravity + pres) * cfg.deltaTime;		
		//float3 nextVel = vel + pres * cfg.deltaTime;		

		float3 velLeapFrog = vel + nextVel;
		velLeapFrog *= 0.5f;

		vel = nextVel;   	
		pos += vel * cfg.deltaTime;   

		float halfWorldXSize = cfg.gridSize.x * cfg.radius;			
		if(pos.x > halfWorldXSize)
			pos.x -= 2 * halfWorldXSize;		
		if(pos.x < -halfWorldXSize)
			pos.x += 2 * halfWorldXSize;

				  
		posArray[index] = make_float4(pos, posData.w);
		velArray[index] = make_float4(vel, velData.w);
		velLeapFrogArray[index] = make_float4(velLeapFrog, velLeapFrogData.w);
}