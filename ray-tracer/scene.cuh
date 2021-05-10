#ifndef SCENEH
#define SCENEH

#include "ray.cuh"
#include "vec3.cuh"

class material;

class isect {
public:
    float t;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
    __device__ isect() {
        t = -1.0;
    }
};

class SceneObject {
public:
	__device__ virtual bool intersects(const ray& r, float t_min, float t_max, isect& i) const = 0;
};


#endif
