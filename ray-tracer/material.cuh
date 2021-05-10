#ifndef MATERIALH
#define MATERIALH

#include "ray.cuh"
#include "scene.cuh"

class material {
public:
    __device__ material(const vec3& ka, const vec3& kd) : ka(ka), kd(kd) {}
    __device__ vec3 shade(const ray& r, const isect& i) {
        vec3 result(0.0, 0.0, 0.0);
        //for now, light will be directional from up top (goes in -y direction)
        vec3 light_in = vec3(1.0, 1.0, 1.0);
        vec3 dir = vec3(0.0, 1.0, 0.0);
        //todo
        vec3 shad_atten = vec3(1.0, 1.0, 1.0);
        vec3 dist_atten = vec3(1.0, 1.0, 1.0);
        
        // TODO: need to be able to access scene object to go through light sources
        //       and calculate intersections of shadow rays

        vec3 diffuse = kd * light_in * max(dot(dir, i.normal), 0.0);
        vec3 ambient(0.3, 0.3, 0.3);
        return diffuse + ambient * ka;
    }
    vec3 ka;
    vec3 kd;
};

#endif