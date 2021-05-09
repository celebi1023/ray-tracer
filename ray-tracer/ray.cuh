#ifndef RAYH
#define RAYH

#include "vec3.cuh"

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const vec3& a, const vec3& b) { source = a; dir = b; }
    __device__ vec3 origin() const { return source; }
    __device__ vec3 direction() const { return dir; }
    __device__ vec3 point_at_parameter(float t) const { return source + t * dir; }

    vec3 source;
    vec3 dir;
};

class isect
{
public:
    float t;
    vec3 p;
    vec3 normal;
};

#endif