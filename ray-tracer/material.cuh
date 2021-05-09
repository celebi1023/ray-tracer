#ifndef MATERIALH
#define MATERIALH

#include "ray.cuh"
#include "scene.cuh"

class  material {
public:
    __device__ material(const vec3& a) : albedo(a) {}
    __device__ vec3 shade(const ray& r, const isect& i) {
        vec3 result(0.0, 0.0, 0.0);
        //for now, light will be directional from up top (goes in -y direction)
        vec3 light_in = vec3(1.0, 1.0, 1.0);
        vec3 dir = vec3(0.0, 1.0, 0.0);
        //todo
        vec3 shad_atten = vec3(1.0, 1.0, 1.0);
        vec3 dist_atten = vec3(1.0, 1.0, 1.0);
        vec3 diffuse = light_in * max(dot(dir, i.normal), 0.0);
        return diffuse;
    }
    vec3 albedo;
};

#endif