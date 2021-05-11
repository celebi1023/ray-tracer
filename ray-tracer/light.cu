#include "light.cuh"
#include "ray.cuh"
#include "vec3.cuh"
#include "scene.cuh"

__device__ float Light::distAtten(const vec3& P) {
    if (is_directional) {
        return 1.0;
    } else {
        float dist = distance(P, light);
        float denom = a + b * dist + c * dist;
        if (denom > 1.0) {
            denom = 1.0;
        } else if (denom < 0.0) {
            denom = 0.0;
        }
        return 1 / denom;
    }
}

// assumes passed P is backed up slightly from intersection point
__device__ vec3 Light::shadowAtten(const vec3& P) {
    isect is;
    ray shadow_ray(P, getDirection(P));
    if (scene->intersects(shadow_ray, 0, FLT_MAX, is)) {
        return vec3(0.0, 0.0, 0.0);
    } else {
        return color;
    }
}

__device__ vec3 Light::getDirection(const vec3& P) {
    if (is_directional) {
        return -light;
    } else {
        return direction(P, light);
    }
}
