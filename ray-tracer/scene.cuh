#ifndef SCENEH
#define SCENEH

#include "ray.cuh"
#include "vec3.cuh"

class SceneObject {
public:
	__device__ virtual bool intersect(const ray& r, float t_min, float t_max, isect& i) const = 0;
};

#endif