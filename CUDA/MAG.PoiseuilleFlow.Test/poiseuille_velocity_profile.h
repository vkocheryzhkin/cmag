#include <vector_types.h>
#include <vector_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <fstream>
#include <sstream>
#include <stack>
#include "poiseuilleflowsystem.cuh"
#include "poiseuilleflowsystem.h"
typedef unsigned int uint;
using namespace std;

void poiseuille_velocity_profile()
{	
	int boundaryOffset = 3;
	int sizex = 64;
	float soundspeed = powf(10.0f, -4.0f);
	float3 gravity = make_float3(powf(10.0f, -4.0f), 0.0f, 0.0f); 	
	float radius = 1.0f / (2 * (sizex - 2 * boundaryOffset) * 1000);
	uint3 gridSize = make_uint3(sizex, 64, 4);   
	uint3 fluid_size = make_uint3(gridSize.x, gridSize.y -  2 * boundaryOffset, 1);	
	float delaTime = powf(10.0f, -4.0f);
	PoiseuilleFlowSystem *psystem = new PoiseuilleFlowSystem(
			delaTime,
			fluid_size,			
			0,0,0,			
			soundspeed,
			gravity,
			boundaryOffset, 
			gridSize,								
			radius,
			false);					
		
	psystem->Reset();

	std::stack<float> timeFrames;								
	timeFrames.push(1.0f);
	timeFrames.push(0.225f);
	timeFrames.push(0.1125f);
	timeFrames.push(0.045f);
	timeFrames.push(0.0225f);
	
	while (!(timeFrames.empty())){
		float timeSlice = timeFrames.top();
		timeFrames.pop();

		while(psystem->GetElapsedTime() < timeSlice)
			psystem->Update();	

		thrust::device_ptr<float4> d_positions((float4*)psystem->getCudaPosVBO());	
		thrust::device_ptr<uint> d_index((uint*)psystem->getCudaIndex());	
		thrust::device_ptr<float4> d_velocity((float4*)psystem->getLeapFrogVelocity());	

		thrust::host_vector<float4> positions(psystem->getNumParticles());
		thrust::host_vector<uint> index(psystem->getNumParticles());
		thrust::host_vector<float4> velocity(psystem->getNumParticles());

		thrust::copy(d_positions, d_positions + psystem->getNumParticles(), positions.begin());		
		thrust::copy(d_index, d_index + psystem->getNumParticles(), index.begin());		
		thrust::copy(d_velocity, d_velocity + psystem->getNumParticles(), velocity.begin());		

		int cx = 0;
		int tt = 0;

		ostringstream buffer;	
		buffer << timeSlice;
		string str = "velocity_profile" + buffer.str().replace(1,1,"x") + ".dat";
		ofstream fp1;	
		fp1.open(str.c_str());
		fp1 << 0.0f << " " << 0.0f << endl;

		for(int i = 0; i < psystem->getNumParticles(); i++){		
			float posx = positions[index[i]].x;		
			float posy = positions[index[i]].y;	
			float bottom = psystem->getWorldOrigin().y +
				2 * psystem -> getParticleRadius() * boundaryOffset;
			if((posx > 0) 
				&& (posx <  2 * psystem -> getParticleRadius())
				&& (posy > bottom)
				&& (posy < bottom + fluid_size.y * 2.0f * radius)
				)
			{							
				fp1 << velocity[index[i]].x << " "
					<< positions[index[i]].y
					- psystem->getWorldOrigin().y
					- boundaryOffset * 2 *psystem -> getParticleRadius()
					<< endl;
				tt++;
			}
			cx++;
		}
		fp1 << 0.0f << " " << pow(10.0f, -3) << endl;
		fp1.close();
	}	
	delete psystem;	
}




