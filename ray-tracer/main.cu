#include <iostream>
#include <fstream>
#include <time.h>

#include "ray.cuh"
#include "scene.cuh"
#include "material.cuh"
#include "shapes.cuh"

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
/*
__global__ void render(float* fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x * 3 + i * 3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = 0.2;
}
*/
__global__ void create_world(SceneObject** sceneObjects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(sceneObjects) = new Sphere(vec3(600, 400, 400), 200);
    }
}

__global__ void free_world(SceneObject** sceneObjects) {
    for (int i = 0; i < 1; i++) {
        //delete ((sphere*)d_list[i])->mat_ptr;
    }
    delete* sceneObjects;
}

__device__ vec3 color(const ray& r, SceneObject** sceneObjects) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 1; i++) {
        isect is;
        //go through all scene objects
        
        for (int j = 0; j < 1; j++) {
            if (sceneObjects[j]->intersects(cur_ray, 0.001f, FLT_MAX, is)) {
                cur_attenuation = vec3(0, 0, 0);
            }
        }
    }
    return cur_attenuation;
}

__global__ void render(vec3* fb, int max_x, int max_y, SceneObject** sceneObjects) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    vec3 col(0, 0, 0);
    vec3 cameraPos(600, 400, -400);
    vec3 screenPos(i, j, 0);
    ray r(cameraPos, screenPos - cameraPos);
    col = color(r, sceneObjects);
    fb[pixel_index] = col;
}


int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    //screen
    //lower left is (0, 0, 0), screen plane is x and y axis
    
    SceneObject** sceneObjects;
    int numObjects = 1;
    checkCudaErrors(cudaMalloc((void**)&sceneObjects, numObjects * sizeof(SceneObject*)));
    create_world << <1, 1 >> > (sceneObjects);
    //sceneObjects[0]->test();

    // allocate FB
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    //render << <blocks, threads >> > (fb, nx, ny);
    render << <blocks, threads >> > (fb, nx, ny, sceneObjects);
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