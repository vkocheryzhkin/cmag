#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <vector_functions.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>
#include "fluidSystem.h"
#include "fluidSystem.cuh"

using namespace std;
using namespace thrust;

void dump() 
{
	float num = 128;
	DamBreakSystem *psystem = new DamBreakSystem(
		make_uint3(num, num, 1),
		1,
		make_uint3(4 * num, 2 * num, 4),				
		1.0f / (2 * num),				
		false); 

	psystem->reset();
	while(psystem->getElapsedTime() < 1.0f)
		psystem->update();//relax
	psystem->changeRightBoundary();
	psystem->changeRightBoundary();

	uint numParticles = psystem->getNumParticles();		

	host_vector<float4> position(numParticles);	
	host_vector<uint> index(numParticles);
	host_vector<float4> scalar_field(numParticles);

	device_ptr<float4> d_position((float4*)psystem->getCudaPosVBO());	
	device_ptr<uint> d_index((uint*)psystem->getCudaIndex());
	device_ptr<float4> d_density((float4*)psystem->getCudaMeasures());
	
	std::queue<float>  timeFrames;		
	//timeFrames.push(0.0001);
	timeFrames.push(1.0);
	timeFrames.push(1.2);
	timeFrames.push(1.4);
	timeFrames.push(1.6);
	timeFrames.push(1.8);
	timeFrames.push(2.0);
	timeFrames.push(2.2);
	timeFrames.push(2.4);
	
	while (!(timeFrames.empty())){
		float timeSlice = timeFrames.front();
		timeFrames.pop();

		while(psystem->getElapsedTime() < timeSlice)
			psystem->update();

		thrust::copy(d_position, d_position + numParticles, position.begin());	
		thrust::copy(d_index, d_index + numParticles, index.begin());	
		thrust::copy(d_density, d_density + numParticles, scalar_field.begin());	

		ostringstream buffer;	
		buffer << timeSlice;
		string str = "dump" + buffer.str().replace(1,1,"x") + ".dat";
		ofstream fp1;	
		fp1.open(str.c_str());
		fp1 << "x " << "y " << "z " << "w " << "density " << "pressure " << endl;
		for(int i = 0; i < numParticles; i++){
			//if(position[index[i]].x <= 1.02f)
			if(position[index[i]].w == Fluid){				
				fp1 << position[index[i]].x << " " << position[index[i]].y << " "
					<< position[index[i]].z << " " << position[index[i]].w << " "
					<< scalar_field[i].x << " " << scalar_field[i].y << " "
					//<< 0 << " " << 0 << " "
					<< endl;
			}
		}			
		fp1.close();
	}	
	delete psystem;
}
