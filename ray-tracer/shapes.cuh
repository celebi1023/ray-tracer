#ifndef SPHEREH
#define SPHEREH

#include "scene.cuh"
#include "vec3.cuh"

class Sphere : public SceneObject
{
public:
	__device__ Sphere() {}
    __device__ Sphere(vec3 c, float r, material *m) : center(c), radius(r), mat_ptr(m) {}
    __device__ bool intersects(const ray& r, float t_min, float t_max, isect& i) const {
        vec3 oc = r.origin() - center;
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                i.t = temp;
                i.p = r.point_at_parameter(i.t);
                i.normal = (i.p - center) / radius;
                i.mat_ptr = mat_ptr;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                i.t = temp;
                i.p = r.point_at_parameter(i.t);
                i.normal = (i.p - center) / radius;
                i.mat_ptr = mat_ptr;
                return true;
            }
        }
        return false;
    }
private:
	float radius;
	vec3 center;
    material* mat_ptr;
};

#endif