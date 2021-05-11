#ifndef MATERIALH
#define MATERIALH

#include "ray.cuh"
#include "scene.cuh"
#include "vec3.cuh"

class material {
public:
    __host__ __device__ material() {}
    __host__ __device__ material(const vec3& e, const vec3& a, const vec3& s, const vec3& d, 
                                 const vec3& r, const vec3& t, bool refl_, float sh) 
        : ke(e), ka(a), ks(s), kd(d), kr(r), kt(t), refl(refl_), shininess(sh) {
        if (refl_) {
            kr = vec3(0.4, 0.4, 0.4);
        } else {
            kr = vec3(0, 0, 0);
        }
    }

    __device__ vec3 shade(Scene* scene, const ray& r, const isect& is) {
        vec3 total(0.0, 0.0, 0.0);
        for (int i = 0; i < scene->nlights; i++) {
            Light* light = scene->lights[i];
            vec3 light_in = light->distAtten(is.p) * light->shadowAtten(r.at(is.t - RAY_EPSILON));
            vec3 dir = light->getDirection(is.p);
            vec3 diffuse = kd * light_in * max(dot(dir, is.normal), 0.0);
            total += diffuse;

            // specular
            vec3 reflect = unit_vector(2 * dot(dir, is.normal) * is.normal - dir);
            float rdotv = max(dot(-r.direction(), reflect), 0.0);
            vec3 specular = ks * light_in * pow(rdotv, shininess);
            total += specular;
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

    float shininess;
};

#endif