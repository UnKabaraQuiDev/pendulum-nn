#version 420 core

in vec3 bet_WorldPos;
in vec3 bet_WorldNormal;
in vec2 bet_UV;

out vec4 fragColor;

uniform sampler2D txt0;
uniform vec4 tint;
uniform bool hasTexture;

void main() {
    fragColor = hasTexture ? texture(txt0, bet_UV) * tint : tint;
}
