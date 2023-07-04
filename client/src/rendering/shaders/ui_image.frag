#version 450
#extension GL_GOOGLE_include_directive : require

#define ENGINE_PIXEL_SHADER
#include "ui.glsl"
#include "../../../engine/shaders/common.glsl"

layout(set = SET_PER_OBJECT, binding = BINDING_OBJECT_INFO) uniform ObjectData {
    mat4 model;
    Rect clip_rect;
    vec2 img_offset;
    vec2 img_scale;
};

layout(set = SET_PER_OBJECT, binding = CUSTOM_OBJ_BINDING_START_ID) uniform sampler2D sourceImage;

layout(location = 0) in Input {
    vec2 texCoord;
} vs_in;

void main() {
    vec2 normScreenCoord = gl_FragCoord.xy / vec2(info.frame_size);
    if (isOutsideCropRegion(normScreenCoord, clip_rect)) {
        discard;
    }
    if (any(lessThan(vs_in.texCoord, vec2(0))) || any(greaterThan(vs_in.texCoord, vec2(1)))) {
        discard;
    }

    outAlbedo = texture(sourceImage, vs_in.texCoord);
    outSpecular = vec4(0.0);
    outEmission = vec4(0.0);
    outNormal = vec4(0.0);
}
