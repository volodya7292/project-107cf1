#include "common.glsl"

struct ray_t {
	vec3 origin;
	vec3 direction;
};

struct sphere_t {
	vec3 origin;
	float radius;
	int material;
};

struct plane_t {
	vec3 direction;
	float distance;
	int material;
};

mat3 rotate_around_x(float angle_degrees) {
	float angle = radians(angle_degrees);
	float _sin = sin(angle);
	float _cos = cos(angle);
	return mat3(1, 0, 0, 0, _cos, -_sin, 0, _sin, _cos);
}

ray_t get_primary_ray(
	vec3 cam_local_point,
	inout vec3 cam_origin,
	inout vec3 cam_look_at
){
	vec3 fwd = normalize(cam_look_at - cam_origin);
	vec3 up = vec3(0, 1, 0);
	vec3 right = cross(up, fwd);
	up = cross(fwd, right);

	ray_t r = ray_t(
		cam_origin,
		normalize(fwd + up * cam_local_point.y + right * cam_local_point.x)
	);
	return r;
}

bool isect_sphere(ray_t ray, sphere_t sphere, inout float t0, inout float t1)
{
	vec3 rc = sphere.origin - ray.origin;
	float radius2 = sphere.radius * sphere.radius;
	float tca = dot(rc, ray.direction);
	float d2 = dot(rc, rc) - tca * tca;
	if (d2 > radius2) return false;
	float thc = sqrt(radius2 - d2);
	t0 = tca - thc;
	t1 = tca + thc;

	return true;
}

// scattering coefficients at sea level (m)
const vec3 betaR = vec3(5.5e-6, 13.0e-6, 22.4e-6); // Rayleigh
const vec3 betaM = vec3(21e-6); // Mie

// scale height (m)
// thickness of the atmosphere if its density were uniform
const float hR = 7994.0; // Rayleigh
const float hM = 1200.0; // Mie

float rayleigh_phase_func(float mu)
{
	return
			3. * (1. + mu*mu)
	/ //------------------------
				(16. * M_PI);
}

// Henyey-Greenstein phase function factor [-1, 1]
// represents the average cosine of the scattered directions
// 0 is isotropic scattering
// > 1 is forward scattering, < 1 is backwards
const float g = 0.76;
float henyey_greenstein_phase_func(float mu)
{
	return
						(1. - g*g)
	/ //---------------------------------------------
		((4. + M_PI) * pow(1. + g*g - 2.*g*mu, 1.5));
}

const float earth_radius = 6360e3; // (m)
const float atmosphere_radius = 6420e3; // (m)

const float sun_power = 40.0;

const sphere_t atmosphere = sphere_t(
	vec3(0, 0, 0), atmosphere_radius, 0
);

const int num_samples = 8;
const int num_samples_light = 1;


const float BIG = 1e20;
const float SMALL = 1e-20;

float approx_air_column_density_ratio_through_atmosphere(
    in float a,
    in float b,
    in float z2,
    in float r0
){
    // GUIDE TO VARIABLE NAMES:
    //  "x*" distance along the ray from closest approach
    //  "z*" distance from the center of the world at closest approach
    //  "r*" distance ("radius") from the center of the world
    //  "*0" variable at reference point
    //  "*2" the square of a variable
    //  "ch" a nudge we give to prevent division by zero, analogous to the Chapman function
    const float SQRT_HALF_PI = sqrt(M_PI/2.);
    const float k = 0.6; // "k" is an empirically derived constant
    float x0 = sqrt(max(r0*r0 - z2, SMALL));
    // if obstructed by the world, approximate answer by using a ludicrously large number
    if (a < x0 && -x0 < b && z2 < r0*r0) { return BIG; }
    float abs_a  = abs(a);
    float abs_b  = abs(b);
    float z      = sqrt(z2);
    float sqrt_z = sqrt(z);
    float ra     = sqrt(a*a+z2);
    float rb     = sqrt(b*b+z2);
    float ch0    = (1. - 1./(2.*r0)) * SQRT_HALF_PI * sqrt_z + k*x0;
    float cha    = (1. - 1./(2.*ra)) * SQRT_HALF_PI * sqrt_z + k*abs_a;
    float chb    = (1. - 1./(2.*rb)) * SQRT_HALF_PI * sqrt_z + k*abs_b;
    float s0     = min(exp(r0- z),1.) / (x0/r0 + 1./ch0);
    float sa     = exp(r0-ra) / max(abs_a/ra + 1./cha, 0.01);
    float sb     = exp(r0-rb) / max(abs_b/rb + 1./chb, 0.01);
    return max( sign(b)*(s0-sb) - sign(a)*(s0-sa), 0.0 );
}

float approx_air_column_density_ratio_along_3d_ray_for_curved_world (
    vec3  P, // position of viewer
    vec3  V, // direction of viewer (unit vector)
    float x, // distance from the viewer at which we stop the "raymarch"
    float r, // radius of the planet
    float H  // scale height of the planet's atmosphere
){
    float xz = dot(-P,V);           // distance ("radius") from the ray to the center of the world at closest approach, squared
    float z2 = dot( P,P) - xz * xz; // distance from the origin at which closest approach occurs
    return approx_air_column_density_ratio_through_atmosphere( 0.-xz, x-xz, z2, r/H );
}

bool get_sun_light(
	ray_t ray,
	inout float optical_depthR,
	inout float optical_depthM
){
	float t0, t1;
	isect_sphere(ray, atmosphere, t0, t1);

    optical_depthR =
        approx_air_column_density_ratio_along_3d_ray_for_curved_world (
            ray.origin,    // position of viewer
            ray.direction, // direction of viewer (unit vector)
            t1, // distance from the viewer at which we stop the "raymarch"
            earth_radius, // radius of the planet
            hR  // scale height of the planet's atmosphere
        );
    optical_depthM =
        approx_air_column_density_ratio_along_3d_ray_for_curved_world (
            ray.origin,    // position of viewer
            ray.direction, // direction of viewer (unit vector)
            t1, // distance from the viewer at which we stop the "raymarch"
            earth_radius, // radius of the planet
            hM  // scale height of the planet's atmosphere
        );

	return true;
}

vec3 get_incident_light(ray_t ray, vec3 sun_dir) {
	// "pierce" the atmosphere with the viewing ray
	float t0, t1;
	if (!isect_sphere(
		ray, atmosphere, t0, t1)) {
		return vec3(0);
	}

	float march_step = t1 / float(num_samples);

	// cosine of angle between view and light directions
	float mu = dot(ray.direction, sun_dir);

	// Rayleigh and Mie phase functions
	// A black box indicating how light is interacting with the material
	// Similar to BRDF except
	// * it usually considers a single angle
	//   (the phase angle between 2 directions)
	// * integrates to 1 over the entire sphere of directions
	float phaseR = rayleigh_phase_func(mu);
	float phaseM = henyey_greenstein_phase_func(mu);


	// optical depth (or "average density")
	// represents the accumulated extinction coefficients
	// along the path, multiplied by the length of that path
	float optical_depthR = 0.;
	float optical_depthM = 0.;

	vec3 sumR = vec3(0);
	vec3 sumM = vec3(0);
	float march_pos = 0.;

	for (int i = 0; i < num_samples; i++) {
		vec3 s =
			ray.origin +
			ray.direction * (march_pos + 0.5 * march_step);
		float height = length(s) - earth_radius;

		// integrate the height scale
		float hr = exp(-height / hR) * march_step;
		float hm = exp(-height / hM) * march_step;
		optical_depthR += hr;
		optical_depthM += hm;

		// gather the sunlight
		ray_t light_ray = ray_t(
			s,
			sun_dir
		);
		float optical_depth_lightR = 0.;
		float optical_depth_lightM = 0.;
		bool overground = get_sun_light(
			light_ray,
			optical_depth_lightR,
			optical_depth_lightM);

		if (overground) {
			vec3 tau =
				betaR * (optical_depthR + optical_depth_lightR) +
				betaM * 1.1 * (optical_depthM + optical_depth_lightM);
			vec3 attenuation = exp(-tau);

			sumR += hr * attenuation;
			sumM += hm * attenuation;
		}

		march_pos += march_step;
	}

	return
		sun_power *
		(sumR * phaseR * betaR +
		sumM * phaseM * betaM);
}

ray_t primary_ray(vec2 screen_size, vec2 screen_pixel_pos, vec3 cam_pos, float fov_y, mat4 camView) {
	float ct = tan(fov_y / 2.0f);
	vec2 clipCoord = (screen_pixel_pos / screen_size * 2.0f - 1.0f);
	vec2 screenPos = clipCoord * vec2(ct * (screen_size.x / screen_size.y), -ct);

	vec3 ray_orig = cam_pos;
	vec4 ray_dir = inverse(camView) * vec4(normalize(vec3(screenPos, -1)), 0.0);

	return ray_t(ray_orig, ray_dir.xyz);
}

vec3 calculateSky(vec2 fragCoord, vec2 renderSize, vec3 camPos, vec3 camDir, float fov, mat4 camView, vec3 sun_dir) {
    camPos.y += earth_radius;

    ray_t ray = primary_ray(renderSize, fragCoord * renderSize, camPos, fov, camView);
	vec3 col = get_incident_light(ray, -sun_dir);

    if (any(isnan(col))) {
        col = vec3(0);
    }

	return col;
}
