// Example Compute Shader

// uniform float exampleUniform;

//in uvec3 gl_NumWorkGroups;
//in uvec3 gl_WorkGroupID;
//in uvec3 gl_LocalInvocationID;
//in uvec3 gl_GlobalInvocationID;
//in uint  gl_LocalInvocationIndex;

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
	vec4 color;
	color = texelFetch(sTD2DInputs[0], ivec2(gl_GlobalInvocationID.xy), 0);
	
	imageStore(mTDComputeOutputs[0], ivec2(gl_GlobalInvocationID.xy), TDOutputSwizzle(color));

	float sumFFT = sum(0);
	imageStore(mTDComputeOutputs[1], ivec2(0,gl_GlobalInvocationID.y), TDOutputSwizzle(vec4(sumFFT,0,0,1)));

}
