struct Matrix
{
	float a11,a12,a13;
	float a21,a22,a23;
	float a31,a32,a33;
};

__device__ Matrix make_Matrix()
{
	Matrix t; 
	t.a11 = 0; t.a12= 0; t.a13 = 0;
	t.a21 = 0; t.a22= 0; t.a23 = 0;
	t.a31 = 0; t.a32= 0; t.a33 = 0;
	return t;
};
__device__ Matrix operator+ (const Matrix & a, const Matrix & b) 
{ 
	Matrix r;
	r.a11 = a.a11 + b.a11;
	r.a12 = a.a12 + b.a12;
	r.a13 = a.a13 + b.a13;
	r.a21 = a.a21 + b.a21;
	r.a22 = a.a22 + b.a22;
	r.a23 = a.a23 + b.a23;
	r.a31 = a.a31 + b.a31;
	r.a32 = a.a32 + b.a32;
	r.a33 = a.a33 + b.a33;
	return r; 
}
__device__ Matrix operator- (const Matrix & a, const Matrix & b) 
{ 
	Matrix r;
	r.a11 = a.a11 - b.a11;
	r.a12 = a.a12 - b.a12;
	r.a13 = a.a13 - b.a13;
	r.a21 = a.a21 - b.a21;
	r.a22 = a.a22 - b.a22;
	r.a23 = a.a23 - b.a23;
	r.a31 = a.a31 - b.a31;
	r.a32 = a.a32 - b.a32;
	r.a33 = a.a33 - b.a33;
	return r; 
}

__device__ Matrix operator+= (Matrix & a, const Matrix & b) 
{ 	
	a.a11 += b.a11;
	a.a12 += b.a12;
	a.a13 += b.a13;

	a.a21 += b.a21;
	a.a22 += b.a22;
	a.a23 += b.a23;

	a.a31 += b.a31;
	a.a32 += b.a32;
	a.a33 += b.a33;
	return a; 
}

__device__ Matrix operator* (const Matrix & a, const Matrix & b) 
{ 
	Matrix r;
	r.a11 = a.a11 * b.a11 + a.a12 * b.a21 + a.a13 * b.a31;
	r.a12 = a.a11 * b.a12 + a.a12 * b.a22 + a.a13 * b.a32;
	r.a13 = a.a11 * b.a13 + a.a12 * b.a23 + a.a13 * b.a33;

	r.a21 = a.a21 * b.a11 + a.a22 * b.a21 + a.a23 * b.a31;
	r.a22 = a.a21 * b.a12 + a.a22 * b.a22 + a.a23 * b.a32;
	r.a23 = a.a21 * b.a13 + a.a22 * b.a23 + a.a23 * b.a33;

	r.a31 = a.a31 * b.a11 + a.a32 * b.a21 + a.a33 * b.a31;
	r.a32 = a.a31 * b.a12 + a.a32 * b.a22 + a.a33 * b.a32;
	r.a33 = a.a31 * b.a13 + a.a32 * b.a23 + a.a33 * b.a33;
	return r; 
}

__device__ float3 operator* (const Matrix & a, const float3 & b) 
{ 
	float3 r;
	r.x = a.a11 * b.x + a.a12 * b.y + a.a13 * b.z;	
	r.y = a.a21 * b.x + a.a22 * b.y + a.a23 * b.z;	
	r.z = a.a31 * b.x + a.a32 * b.y + a.a33 * b.z;	
	return r; 
}

__device__ Matrix operator* (const float & a, const Matrix & b) 
{ 
	Matrix r;
	r.a11 = a * b.a11;
	r.a12 = a * b.a12;
	r.a13 = a * b.a13;

	r.a21 = a * b.a21;
	r.a22 = a * b.a22;
	r.a23 = a * b.a23;

	r.a31 = a * b.a31;
	r.a32 = a * b.a32;
	r.a33 = a * b.a33;

	return r; 
}

__device__ Matrix Transpose (const Matrix & b) 
{ 
	Matrix r;
	r.a11 = b.a11;
	r.a12 = b.a21;
	r.a13 = b.a31;

	r.a21 = b.a12;
	r.a22 = b.a22;
	r.a23 = b.a32;

	r.a31 = b.a13;
	r.a32 = b.a23;
	r.a33 = b.a33;
	return r; 
}