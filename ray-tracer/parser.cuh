#ifndef PARSERH
#define PARSERH

#include <string>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "vec3.cuh"
#include "scene.cuh"
#include "check.cuh"

__global__ void createFloor(SceneObject** objects, int i, material m) {
    material* mat = new material(m);
    objects[i] = new Floor(mat);
}

__global__ void createSphere(SceneObject** objects, int i, vec3 c, float r, material m) {
    material* mat = new material(m);
    objects[i] = new Sphere(c, r, mat);
}

__global__ void createBox(SceneObject** objects, int i, vec3 p1, vec3 p2, material m) {
    material* mat = new material(m);
    objects[i] = new Box(p1, p2, mat);
}

// TODO: add constant, linear, and quadratic term coefficients as params
__global__ void createLight(Scene* scene, Light** lights, int i, bool is_dir, vec3 light, vec3 color) {
    // constant = 0, linear = 0.1, quadratic = 0.1
    lights[i] = new Light(scene, is_dir, light, color, 0, 0.1, 0.1);
}

__global__ void createScene(Scene* scene, SceneObject** objects, Light** lights, int nobjects, int nlights) {
    // not sure if this is possible or if it needs to be allocated with new
    *scene = Scene(objects, lights, nobjects, nlights);
}

__host__ vec3 parseVec(std::ifstream& infile) {
    float x;
    float y;
    float z;
    infile >> x >> y >> z;
    return vec3(x, y, z);
}

__host__ material parseMat(std::ifstream& infile) {
    std::string line;
    getline(infile, line);
    if (line.compare("material") != 0) {
        std::cerr << "file format with material:" << line << "\n";
        return material();
    }

    vec3 ke = parseVec(infile);
    vec3 ka = parseVec(infile);
    vec3 ks = parseVec(infile);
    vec3 kd = parseVec(infile);
    vec3 kr = parseVec(infile);
    vec3 kt = parseVec(infile);
    int isRefl;
    infile >> isRefl;
    float shininess;
    infile >> shininess;
    getline(infile, line);
    return material(ke, ka, ks, kd, kr, kt, isRefl == 1, shininess);
}

__host__ Scene* parse() {
    std::cout << "Load scene: ";
    std::string file;
    std::cin >> file;
    std::ifstream infile(file);
    if (infile.fail()) {
        std::cerr << "File not found\n";
        return nullptr;
    }

    std::string line;
    getline(infile, line);
    if (line.compare("number of objects") != 0) {
        std::cerr << "file format problem - number of objects\n";
        return nullptr;
    }

    int numObjects;
    infile >> numObjects;
    getline(infile, line);

    SceneObject** objects;
    cudaMalloc((void**) &objects, numObjects * sizeof(SceneObject*));

    for (int i = 0; i < numObjects; i++) {
        getline(infile, line);
        if (line.compare("---") != 0) {
            std::cerr << "file format at iteration " << i << "\n";
            return nullptr;
        }

        getline(infile, line);  // type of object
        if (line.compare("floor") == 0) {
            // TODO: add material (change floor constructor)
            // *********************************************
            material mat = parseMat(infile);
            createFloor<<<1, 1>>>(objects, i, mat);
        } else if (line.compare("sphere") == 0) {
            vec3 center = parseVec(infile);
            float radius;
            infile >> radius;
            getline(infile, line);
            material mat = parseMat(infile);
            createSphere<<<1, 1>>>(objects, i, center, radius, mat);
        } else if (line.compare("box") == 0) {
            vec3 min_pt = parseVec(infile);
            vec3 max_pt = parseVec(infile);
            getline(infile, line);
            material mat = parseMat(infile);
            createBox<<<1, 1>>>(objects, i, min_pt, max_pt, mat);
        } else {
            std::cerr << "shape not recognized, iteration: " << i << "\n";
            std::cerr << "line causing errror: " << line << "\n";
        }
    }

    // parse lights 
    getline(infile, line);
    if (line.compare("number of lights") != 0) {
        std::cerr << "file format problem - number of lights\n";
        return nullptr;
    }

    int numLights;
    infile >> numLights;
    getline(infile, line);

    Light** lights;
    cudaMalloc((void**) &lights, numLights * sizeof(Light*));

    Scene* scene;
    cudaMalloc((void**) &scene, sizeof(Scene));
    createScene<<<1, 1>>>(scene, objects, lights, numObjects, numLights);

    for (int i = 0; i < numLights; i++) {
        getline(infile, line);
        if (line.compare("---") != 0) {
            std::cerr << "file format at iteration " << i << "\n";
            return nullptr;
        }

        //type of light
        getline(infile, line);
        if (line.compare("directional") == 0) {
            vec3 dir = parseVec(infile);
            vec3 col = parseVec(infile);
            getline(infile, line);
            createLight<<<1, 1>>>(scene, lights, i, true, dir, col);
        } else if (line.compare("positional") == 0) {
            vec3 pos = parseVec(infile);
            vec3 col = parseVec(infile);
            getline(infile, line);
            createLight<<<1, 1>>>(scene, lights, i, false, pos, col);
        } else {
            std::cerr << "light not recognized, iteration: " << i << "\n";
            std::cerr << "line causing errror: " << line << "\n";
        }
    }

    infile.close();
    std::cout << "Scene parsed successfully\n";
    checkCudaErrors(cudaDeviceSynchronize());  // wait for GPU kernels to complete
    return scene;
}

#endif
