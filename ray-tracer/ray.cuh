#ifndef RAYH
#define RAYH

#include "vec3.cuh"

#define RAY_EPSILON 0.001f

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const vec3& a, const vec3& b) { source = a; dir = b; }
    __device__ vec3 origin() const { return source; }
    __device__ vec3 direction() const { return dir; }
    __device__ vec3 at(float t) const { return source + t * dir; }

    vec3 source;
    vec3 dir;
};


#endif