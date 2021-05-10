#include <iostream>
#include <fstream>
#include <cfloat>
#include <ctime>

#include "ray.cuh"
#include "scene.cuh"
#include "material.cuh"
#include "shapes.cuh"
#include "parser.cuh"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void create_world(SceneObject** sceneObjects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        //material mat1(vec3(1.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0));
        material* mat1 = new material(vec3(0.4, 0.6, 0.3), vec3(0.7, 1.0, 0.5));
        //material mat2(vec3(1.0, 0.35, 0.5), vec3(1.0, 0.35, 0.5));
        *(sceneObjects) = new Floor();
        *(sceneObjects + 1) = new Sphere(vec3(600, 400, 400), 200, mat1);
        *(sceneObjects + 2) = new Box(vec3(1000, 10, 400), vec3(1300, 310, 700), mat1);
    }
}

__global__ void free_world(SceneObject** sceneObjects) {
    for (int i = 0; i < 1; i++) {
        //delete ((sphere*)d_list[i])->mat_ptr;
    }
    delete* sceneObjects;
}

__device__ vec3 color(const ray& r, SceneObject** sceneObjects, int numObjects) {
    ray cur_ray = r;
    vec3 background = vec3(0.1, 0.65, 1.0);
    vec3 cur_attenuation = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < 1; i++) {
        isect is;
        bool intersectFound = false;
        isect curIs;
        
        // go through all scene objects
        for (int j = 0; j < numObjects; j++) {
            if (sceneObjects[j]->intersects(cur_ray, 0.001f, FLT_MAX, curIs)) {
                if (!intersectFound || curIs.t < is.t) {
                    intersectFound = true;
                    is = curIs;
                }
            }
        }
        if (intersectFound) {
            cur_attenuation += is.mat_ptr->shade(r, is);
        } else {
            cur_attenuation += background;
        }
    }
    return cur_attenuation;
}

__global__ void render(vec3* fb, int max_x, int max_y, SceneObject** sceneObjects, int numObjects) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    
    vec3 cameraPos(600, 400, -400);
    vec3 look = vec3(0, 0, 1);  // can change to whatever look direction
    vec3 u(1, 0, 0), v(0, 1, 0);
    //float x = (i + 0.5) / max_x - 0.5;  // normalized to [-0.5, 0.5]
    //float y = (j + 0.5) / max_y - 0.5;
    //ray r(cameraPos, unit_vector(look + (x * u) + (y * v)));
    
    float fov = 30;
    float aspectratio = max_x / float(max_y);
    float M_PI = 3.141592653589793;
    float angle = tan(M_PI * 0.5 * fov / 180);
    float xx = (2 * ((i + 0.5) / float(max_x)) - 1) * angle * aspectratio;
    float yy = (1 - 2 * ((j + 0.5) / float(max_y))) * angle;
    vec3 dir = unit_vector(vec3(xx, -yy, 0.5));
    ray r(cameraPos, dir);
    
    vec3 col = color(r, sceneObjects, numObjects);
    int pixel_index = j * max_x + i;
    fb[pixel_index] = col;
}


int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    //parse
    SceneObject** sceneObjects;
    parse ();

    int num_pixels = nx * ny;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    //screen
    //lower left is (0, 0, 0), screen plane is x and y axis
    
    int numObjects = 3;
    checkCudaErrors(cudaMalloc((void**)&sceneObjects, numObjects * sizeof(SceneObject*)));
    create_world<<<1, 1>>>(sceneObjects);

    // allocate FB
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    //render << <blocks, threads >> > (fb, nx, ny);
    render << <blocks, threads >> > (fb, nx, ny, sceneObjects, numObjects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image

    std::ofstream outfile("test.ppm");

    // Output FB as Image
    outfile << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }
    

    checkCudaErrors(cudaFree(fb));
    free_world << <1, 1 >> > (sceneObjects);
}

