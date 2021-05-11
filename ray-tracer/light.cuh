#ifndef LIGHTH
#define LIGHTH

#include <cfloat>
#include "vec3.cuh"
#include "ray.cuh"

class Scene;

class Light {
public:
    __device__ Light(Scene* scene, bool is_dir, const vec3& light, const vec3& color, 
                     float a = 0, float b = 0, float c = 1) 
        : scene(scene), is_directional(is_dir), light(light), color(color), 
          a(a), b(b), c(c) {}

    __device__ float distAtten(const vec3& P);
    __device__ vec3 shadowAtten(const vec3& P);
    __device__ vec3 getDirection(const vec3& P);

    Scene* scene;
    bool is_directional;
    vec3 light;
    vec3 color;
    float a;
	float b;
	float c;
};

#endif