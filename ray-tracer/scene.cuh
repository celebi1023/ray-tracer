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
        vec3 dir = r.direction();
        float x = dir[0], y = dir[1], z = dir[2];
        float absX = fabs(x);
        float absY = fabs(y);
        float absZ = fabs(z);

        // int idx;
        float maxAxis, /*, uc */ vc;
        if (x > 0 && absX >= absY && absX >= absZ) {
            maxAxis = absX;
            // uc = -z;
            vc = y;
            // idx = 0;
        } else if (x < 0 && absX >= absY && absX >= absZ) {
            maxAxis = absX;
            // uc = z;
            vc = y;
            // idx = 1;
        } else if (y > 0 && absY >= absX && absY >= absZ) {
            maxAxis = absY;
            // uc = x;
            vc = -z;
            // idx = 2;
        } else if (y < 0 && absY >= absX && absY >= absZ) {
            maxAxis = absY;
            // uc = x;
            vc = z;
            // idx = 3;
        } else if (z > 0 && absZ >= absX && absZ >= absY) {
            maxAxis = absZ;
            // uc = x;
            vc = y;
            // idx = 4;
        } else {
            maxAxis = absZ;
            // uc = -x;
            vc = y;
            // idx = 5;
        }

        // float u = 0.5 * (uc / maxAxis + 1.0);
        float v = 0.5 * (vc / maxAxis + 1.0);

        // TODO: linearly interpolate texture map at idx for cube map

        return v * vec3(0.5, 0.7, 1.0) + (1 - v) * vec3(1, 1, 1);
    }

    int nobjects;
    int nlights;
    SceneObject** objects;
    Light** lights;
};

#endif
