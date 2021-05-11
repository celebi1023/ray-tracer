#ifndef TRIMESHH
#define TRIMESHH

#include "scene.cuh"
#include "vec3.cuh"

class TrimeshFace : public SceneObject
{
public:
    __device__ TrimeshFace(vec3 a_, vec3 b_, vec3 c_) : a(a_), b(b_), c(c_) {}
    __device__ bool intersects(const ray& r, float t_min, float t_max, isect& i) const {
        return false;
    }
private:
    vec3 a;
    vec3 b;
    vec3 c;
    material* mat_ptr;
};

class Trimesh : public SceneObject
{
public:
    __device__ Trimesh(int numFaces_) : numFaces(numFaces_) {}
    __device__ bool intersects(const ray& r, float t_min, float t_max, isect& i) const {
        return false;
    }
private:
    TrimeshFace* faces;
    int numFaces;
};

#endif
