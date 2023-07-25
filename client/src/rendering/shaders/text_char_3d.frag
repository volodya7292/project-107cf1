#version 450
#extension GL_GOOGLE_include_directive : require

#include "../../../engine/shaders/text_char_frag_template.glsl"

void main() {
    float opacity, sd;
    calculateCharShading(0.5, opacity, sd);
    writeOutputAlbedo(vec4(vs_in.color.rgb, vs_in.color.a * opacity));
}
