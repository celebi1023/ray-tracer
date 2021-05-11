#ifndef SCENEH
#define SCENEH

#include <climits>
#include "ray.cuh"
#include "vec3.cuh"
#include "light.cuh"

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

class Scene {
public:
    __device__ Scene(SceneObject** objects, Light** lights, int nobjects, int nlights) 
        : objects(objects), lights(lights), nobjects(nobjects), nlights(nlights) {}
    
    __device__ bool intersects(const ray& r, float t_min, float t_max, isect& is) {
        isect i_curr;
        bool intersectFound = false;
        for (int j = 0; j < nobjects; j++) {
            if (objects[j]->intersects(r, t_min, t_max, i_curr)) {
                if (!intersectFound || i_curr.t < is.t) {
                    intersectFound = true;
                    is = i_curr;
                }
            }
        }
        return intersectFound;
    }

    __device__ vec3 background(const ray& r) {
        // TODO: linearly blend blue/white for sky or use cubemap
        
        return vec3(0.1, 0.65, 1.0);
    }

    int nobjects;
    int nlights;
    SceneObject** objects;
    Light** lights;
};

#endif
