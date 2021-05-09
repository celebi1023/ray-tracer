#ifndef SCENEH
#define SCENEH

#include "ray.cuh"

class SceneObject {
	public:
		__device__ virtual bool intersect(const ray&r, isect& i);
};

#endif