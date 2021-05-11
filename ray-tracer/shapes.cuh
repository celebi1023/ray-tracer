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
                i.p = r.at(i.t);
                i.normal = unit_vector(i.p - center);
                i.mat_ptr = mat_ptr;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                i.t = temp;
                i.p = r.at(i.t);
                i.normal = unit_vector(i.p - center);
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

//plane aligned with zx plane
class Floor : public SceneObject
{
public:
    __device__ Floor() {
        minX = 0;
        maxX = 1200;
        minZ = 0;
        maxZ = 1200;
        y = 0;
        //material mat(vec3(0.9, 0.4, 0.6), vec3(0.9, 0.4, 0.6));
        //mat_ptr = new material(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
        mat_ptr = new material(vec3(0.9, 0.4, 0.6), vec3(0.9, 0.4, 0.6), false);
        normal = vec3(0.0, 1.0, 0.0);
    }

    __device__ bool intersects(const ray& r, float t_min, float t_max, isect& i) const {
        if (dot(normal, r.direction()) >= 0) return false;
        float t = (dot(normal, vec3(0.0, y, 0.0)) - dot(normal, r.origin())) / dot(normal, r.direction());
        vec3 p = r.at(t);
        if (p.x() < minX || p.x() > maxX || p.z() < minZ || p.z() > maxZ) return false;
        i.t = t;
        i.p = p;
        i.normal = normal;
        i.mat_ptr = mat_ptr;
        return true;
    }

private:
    float minX;
    float maxX;
    float minZ;
    float maxZ;
    float y;
    material* mat_ptr;
    vec3 normal;
};

class Cylinder : public SceneObject
{
public:
    __device__ Cylinder() {}
    __device__ Cylinder(vec3 p1_, vec3 p2_, float r, material* m) : p1(p1_), p2(p2_), radius(r), mat_ptr(m) {}
    __device__ bool intersects(const ray& r, float t_min, float t_max, isect& i) const {
        float t = -1;
        //shaft
        //line from p1 to p2
        vec3 va = unit_vector(p2 - p1);
        vec3 pa = p1;
        vec3 v = r.direction();
        vec3 p = r.origin();
        vec3 dp = p - pa; //delta p
        vec3 tempA = v - dot(v, va) * va;
        float A = dot(tempA, tempA);
        float B = 2 * dot(v - dot(v, va) * va, dp - dot(dp, va) * va);
        vec3 tempC = dp - dot(dp, va) * va;
        float C = dot(tempC, tempC) - radius * radius;
        float discr = B * B - 4 * A * C;
        if (discr >= 0) {
            float t1 = (-B + sqrt(discr)) / 2;
            if (t1 >= 0 && dot(va, p + v * t1) > 0 && dot(va, p + v * t1) < 0 && (t == -1 || t1 < t)) {
                t = t1;
            }
            float t2 = (-B - sqrt(discr)) / 2;
            if (t2 >= 0 && dot(va, p + v * t2) > 0 && dot(va, p + v * t2) < 0 && (t == -1 || t2 < t)) {
                t = t2;
            }
            //normals
            if (t != -1) {
                i.t = t;
                vec3 point = v * t + p;
                //va is p1 -> p2
                vec3 proj = p1 + dot(vec3(point - p1), va) / dot(va, va) * va;
                i.normal = point - proj;
                i.mat_ptr = mat_ptr;
            }
        }
        //todo: caps
        return t == -1;
    }
private:
    float radius;
    vec3 p1;
    vec3 p2;
    material* mat_ptr;
};

//axis aligned
class Box : public SceneObject
{
public:
    __device__ Box() {}
    __device__ Box(vec3 min_, vec3 max_, material* m) : min(min_), max(max_), mat_ptr(m) {}
    __device__ bool intersects(const ray& r, float t_min, float t_max, isect& i) const {
        float t = -1;
        float curT = -1;
        vec3 v = r.direction();
        vec3 p = r.origin();
        //face zmin
        curT = (min.z() - p.z()) / v.z();
        if (curT >= t_min) {
            //point of intersection
            vec3 point = v * curT + p;
            if (min.x() <= point.x() && point.x() <= max.x() && min.y() <= point.y() && point.y() <= max.y()) {
                if (t == -1 || curT < t) {
                    t = curT;
                    i.normal = vec3(0, 0, -1);
                    //i.normal = vec3(0, 1, 0); // for testing
                }
            }
        }
        //face zmax
        curT = (max.z() - p.z()) / v.z();
        if (curT >= t_min) {
            vec3 point = v * curT + p;
            if (min.x() <= point.x() && point.x() <= max.x() && min.y() <= point.y() && point.y() <= max.y()) {
                if (t == -1 || curT < t) {
                    t = curT;
                    i.normal = vec3(0, 0, 1);
                }
            }
        }
        //face ymin
        curT = (min.y() - p.y()) / v.y();
        if (curT >= t_min) {
            vec3 point = v * curT + p;
            if (min.x() <= point.x() && point.x() <= max.x() && min.z() <= point.z() && point.z() <= max.z()) {
                if (t == -1 || curT < t) {
                    t = curT;
                    i.normal = vec3(0, -1, 0);
                }
            }
        }
        //face ymax
        curT = (max.y() - p.y()) / v.y();
        if (curT >= t_min) {
            vec3 point = v * curT + p;
            if (min.x() <= point.x() && point.x() <= max.x() && min.z() <= point.z() && point.z() <= max.z()) {
                if (t == -1 || curT < t) {
                    t = curT;
                    i.normal = vec3(0, 1, 0);
                }
            }
        }
        //face xmin
        curT = (min.x() - p.x()) / v.x();
        if (curT >= t_min) {
            vec3 point = v * curT + p;
            if (min.y() <= point.y() && point.y() <= max.y() && min.z() <= point.z() && point.z() <= max.z()) {
                if (t == -1 || curT < t) {
                    t = curT;
                    i.normal = vec3(-1, 0, 0);
                }
            }
        }
        //face xmax
        curT = (max.x() - p.x()) / v.x();
        if (curT >= t_min) {
            vec3 point = v * curT + p;
            if (min.y() <= point.y() && point.y() <= max.y() && min.z() <= point.z() && point.z() <= max.z()) {
                if (t == -1 || curT < t) {
                    t = curT;
                    i.normal = vec3(1, 0, 0);
                }
            }
        }
        if (t == -1 || t >= t_max) return false;
        i.t = t;
        i.p = v * t + p;
        i.mat_ptr = mat_ptr;
        return true;
    }
private:
    vec3 min;
    vec3 max;
    material* mat_ptr;
};
#endif