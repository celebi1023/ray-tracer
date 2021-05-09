#ifndef MATERIALH
#define MATERIALH

#include "ray.cuh"
#include <glm/vec3.hpp>

class material {
public:
	__device__ virtual bool scatter(const ray& r_in, const isect& i, glm::vec3& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {
public:

};

#endif