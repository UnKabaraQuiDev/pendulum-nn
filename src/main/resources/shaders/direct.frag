#version 420 core

in vec3 bet_WorldPos;
in vec3 bet_WorldNormal;
in vec2 bet_UV;
flat in uint bet_InstanceID;

out vec4 fragColor;

uniform sampler2D txt0;
uniform bool hasTexture;
uniform uint instanceCount;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
	float hue = (float(bet_InstanceID) + 0.5) / float(instanceCount);
	vec4 tint = vec4(hsv2rgb(vec3(hue, 1.0, 1.0)), 1);
    fragColor = hasTexture ? texture(txt0, bet_UV) * tint : tint;
}

