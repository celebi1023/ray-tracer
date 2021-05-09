#ifndef RAYH
#define RAYH

#include <glm/vec3.hpp>

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const glm::vec3& a, const glm::vec3& b) { source = a; dir = b; }
    __device__ glm::vec3 origin() const { return source; }
    __device__ glm::vec3 direction() const { return dir; }
    __device__ glm::vec3 point_at_parameter(float t) const { return source + t * dir; }

    glm::vec3 source;
    glm::vec3 dir;
};

class isect
{
private:
    double t;
    glm::vec3 N;
};

#endif