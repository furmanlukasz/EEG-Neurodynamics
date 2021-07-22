// Example Compute Shader

// uniform float exampleUniform;

//in uvec3 gl_NumWorkGroups;
//in uvec3 gl_WorkGroupID;
//in uvec3 gl_LocalInvocationID;
//in uvec3 gl_GlobalInvocationID;
//in uint  gl_LocalInvocationIndex;

//idx - glsltop input index



float sum(int idx){
	float sum = 0;
	int k = 0;
	for (int i = 0; i < gl_LocalInvocationID.x; i++){
		sum += texelFetch(sTD2DInputs[idx], ivec2(i,gl_GlobalInvocationID.y), 0).r;
		k += 1;
	}
return sum;
}

layout (local_size_x = 49, local_size_y = 1) in;
void main()
{
	vec4 sftVectors, sftVectorsShift;
	
	
	sftVectors = texelFetch(sTD2DInputs[0], ivec2(gl_GlobalInvocationID.xy), 0);
	sftVectorsShift = texelFetch(sTD2DInputs[0], ivec2(gl_GlobalInvocationID.x,gl_GlobalInvocationID.y-1), 0);

	// output orginal vectors
	imageStore(mTDComputeOutputs[0], ivec2(gl_GlobalInvocationID.xy), TDOutputSwizzle(sftVectors));

	// calc and output sum of FFTvector in Y
	float sumFFT = sum(0);
	imageStore(mTDComputeOutputs[1], ivec2(0,gl_GlobalInvocationID.y), TDOutputSwizzle(vec4(sumFFT,0,0,1)));
	

	
	// calc and output dist betwen vector FFT(tn) and vector FFT(tn-1)  
	float maxSFTVector = max(sftVectors.r, sftVectorsShift.r);
	float DistSFT_Vectors = (sftVectors.r-sftVectorsShift.r);
	DistSFT_Vectors = length(DistSFT_Vectors);
	DistSFT_Vectors *= maxSFTVector;
	// DistSFT_Vectors *= abs(sftVectors.r-sftVectorsShift.r);
	
	imageStore(mTDComputeOutputs[2], ivec2(gl_GlobalInvocationID.xy), TDOutputSwizzle(vec4(DistSFT_Vectors,0,0,1)));

	

}
