#version 420 core

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Normal;
layout(location = 2) in vec2 in_UV;
// instances
layout(location = 5) in mat4 in_InstanceMatrix;

uniform mat4 transformationMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 bet_ObjPos;
out vec3 bet_ObjNormal;
out vec3 bet_WorldPos;
out vec3 bet_WorldNormal;
out vec3 bet_ViewPos;
out vec3 bet_ViewNormal;
out vec2 bet_UV;

// instances
flat out uint bet_InstanceID;

void main() {
	mat4 modelMatrix = transformationMatrix * in_InstanceMatrix;

// local position
	vec3 pos = in_Position;

// obj space
	bet_ObjPos = pos;
	bet_ObjNormal = normalize(in_Normal);

// world space
	vec4 worldPos4 = modelMatrix * vec4(pos, 1.0);
	bet_WorldPos = worldPos4.xyz;

	mat3 normalMatrix = transpose(inverse(mat3(modelMatrix)));
	bet_WorldNormal = normalize(normalMatrix * in_Normal);

// view space
	vec4 viewPos4 = viewMatrix * worldPos4;
	bet_ViewPos = viewPos4.xyz;

	mat3 viewNormalMatrix = transpose(inverse(mat3(viewMatrix)));
	bet_ViewNormal = normalize(viewNormalMatrix * bet_WorldNormal);

// pass through
	bet_UV = in_UV;
	bet_InstanceID = gl_InstanceID;
	
	gl_Position = projectionMatrix * viewPos4;
}
