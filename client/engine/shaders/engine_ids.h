// Note: according to Pipeline Layout Compatibility, the least frequently changing descriptors are placed first.
// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#descriptorsets-compatibility

/// General descriptor set for engine-related resources
#define SET_GENERAL_PER_FRAME 0
#define BINDING_FRAME_INFO 0
#define BINDING_MATERIAL_BUFFER 1
#define BINDING_ALBEDO_ATLAS 2
#define BINDING_SPECULAR_ATLAS 3
#define BINDING_NORMAL_ATLAS 4
/// Translucency depths (only used in translucency passes)
#define BINDING_TRANSPARENCY_DEPTHS 5
/// Translucency colors (only used in translucency passes)
#define BINDING_TRANSPARENCY_COLORS 6
/// Solid depths attachment (only used in translucency depths pass)
#define BINDING_SOLID_DEPTHS 7


/// Specific to material pipeline descriptor set for its per-frame resources
#define SET_CUSTOM_PER_FRAME 1


/// MaterialPipeline-specific descriptor set for per object data (eg. model matrix)
#define SET_PER_OBJECT 2
#define BINDING_OBJECT_INFO 0
#define CUSTOM_OBJ_BINDING_START_ID 1


#define THREAD_GROUP_2D_SIZE 8
#define THREAD_GROUP_1D_SIZE (THREAD_GROUP_2D_SIZE * THREAD_GROUP_2D_SIZE)

#define OIT_N_CLOSEST_LAYERS 4
#define FARTHEST_DEPTH_UINT 0u // z is reversed
