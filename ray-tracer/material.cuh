#ifndef MATERIALH
#define MATERIALH

#include "ray.cuh"
#include "scene.cuh"

class material {
public:
    __host__ __device__ material(const vec3& ka_, const vec3& kd_) : ka(ka_), kd(kd_) {}
    __device__ vec3 shade(Scene* scene, const ray& r, const isect& is) {
        vec3 total(0.0, 0.0, 0.0);
        for (int i = 0; i < scene->nlights; i++) {
            Light* light = scene->lights[i];
            vec3 light_in = light->distAtten(is.p) * light->shadowAtten(is.p);
            vec3 dir = light->getDirection(is.p);
            vec3 diffuse = kd * light_in * max(dot(dir, is.normal), 0.0);
            total += diffuse;
        }

        // TODO: make ambient part of scene

        vec3 ambient(0.3, 0.3, 0.3);
        return clamp(total + ambient * ka);
    }
    vec3 ke;    //emissive
    vec3 ka;    //ambient
    vec3 ks;    //specular
    vec3 kd;    //diffuse
    vec3 kr;    //reflective
    vec3 kt;    //transmissive

    bool refl;  //specular reflector?
    bool trans; //specular transmitter?
    bool recur; //either one
    bool spec;  //any kind of specular?
    bool both;  //reflection and transmission
};

#endif