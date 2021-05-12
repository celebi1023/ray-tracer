#include <iostream>
#include <fstream>
#include <cfloat>
#include <ctime>
#include <string>

#include "check.cuh"
#include "ray.cuh"
#include "scene.cuh"
#include "light.cuh"
#include "material.cuh"
#include "shapes.cuh"
#include "parser.cuh"
#include "trimesh.cuh"


__device__ vec3 color(const ray& r, Scene* scene, bool debug = false) {
    ray cur_ray = r;
    vec3 total = vec3(0.0, 0.0, 0.0);
    // TODO - add bias to reflect and refract
    vec3 k_factor = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        isect is;    
        if (scene->intersects(cur_ray, RAY_EPSILON, FLT_MAX, is)) {
            if (debug) {
                printf("intersection at %f %f %f\n", is.p[0], is.p[1], is.p[2]);
            }

            if (is.mat_ptr->trans) {
                if (debug) {
                    printf("material is transmissive\n");
                }
                bool inside = dot(cur_ray.direction(), is.normal) > 0;
                float n_i = inside ? is.mat_ptr->index : 1.0001;
                float n_t = inside ? 1.0001 : is.mat_ptr->index;
                vec3 normal = !inside ? -is.normal : is.normal;
                float cosi = dot(cur_ray.direction(), normal);
                float n_ratio = n_i / n_t;
                float k = 1 - n_ratio * n_ratio * (1 - cosi * cosi);
                vec3 refract_dir;
                if (k < 0) {
                    if (debug) {
                        printf("total internal reflection\n");
                    }
                    // total internal reflection
                    refract_dir = cur_ray.direction() - 2 * dot(cur_ray.direction(), -normal) * -normal;
                } else {
                    refract_dir = n_ratio * cur_ray.direction() + (n_ratio * cosi - sqrt(k)) * normal;
                }

                if (inside) {
                    k_factor *= pow(is.mat_ptr->kt, is.t / 100);  // divide by 100 since objects are large
                }
                vec3 shading = is.mat_ptr->shade(scene, cur_ray, is, inside);
                total += k_factor * shading;
                cur_ray = ray(cur_ray.at(is.t), unit_vector(refract_dir));
                if (debug) {
                    printf("refraction direction: %f %f %f\n", refract_dir[0], refract_dir[1], refract_dir[2]);
                    printf("shading: %f %f %f, k_factor: %f %f %f\n", shading[0], shading[1], shading[2], k_factor[0], k_factor[1], k_factor[2]);
                }
            } else if (is.mat_ptr->refl) {
                vec3 reflect_dir = cur_ray.direction() - 2 * dot(cur_ray.direction(), is.normal) * is.normal;
                cur_ray = ray(cur_ray.at(is.t), unit_vector(reflect_dir));
                vec3 shading = is.mat_ptr->shade(scene, cur_ray, is);
                total += k_factor * shading;
                k_factor *= is.mat_ptr->kr;
                if (debug) {
                    printf("material is reflective\n");
                    printf("reflection direction: %f %f %f\n", reflect_dir[0], reflect_dir[1], reflect_dir[2]);
                    printf("shading: %f %f %f, k_factor: %f %f %f\n", shading[0], shading[1], shading[2], k_factor[0], k_factor[1], k_factor[2]);
                }
            } else {
                vec3 shading = is.mat_ptr->shade(scene, cur_ray, is);
                total += k_factor * shading;
                if (debug) {
                    printf("material is neither\n");
                    printf("shading: %f %f %f\n", shading[0], shading[1], shading[2]);
                }
                break;
            }
        } else {
            if (debug) {
                printf("ray hit background\n");
            }
            total += k_factor * scene->background(cur_ray);
            break;
        }

        if (debug) {
            printf("total shading %d: %f %f %f\n", i, total[0], total[1], total[2]);
        }
    }
    return clamp(total);
}

__global__ void render(vec3* fb, int max_x, int max_y, Scene* scene, int aa = 1) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    
    vec3 cameraPos(600, 400, -400);
    vec3 look = vec3(0, 0, 1);  // can change to whatever look direction
    vec3 u(1, 0, 0), v(0, 1, 0);
    //float x = (i + 0.5) / max_x - 0.5;  // normalized to [-0.5, 0.5]
    //float y = (j + 0.5) / max_y - 0.5;
    //ray r(cameraPos, unit_vector(look + (x * u) + (y * v)));
    
    int aamax_x = max_x * aa;
    int aamax_y = max_y * aa;

    vec3 pixel = vec3(0, 0, 0);
    for (int row = 0; row < aa; row++) {
        for (int col = 0; col < aa; col++) {
            float fov = 30;
            float aspectratio = aamax_x / float(aamax_y);
            float angle = tan(M_PI * 0.5 * fov / 180);
            float xx = (2 * ((i * aa + row + 0.5) / float(aamax_x)) - 1) * angle * aspectratio;
            float yy = (1 - 2 * ((j * aa + col + 0.5) / float(aamax_y))) * angle;
            vec3 dir = unit_vector(vec3(xx, -yy, 0.5));
            ray r(cameraPos, dir);
            pixel += color(r, scene);
        }
    }
    
    pixel /= aa * aa;
    int pixel_index = j * max_x + i;
    fb[pixel_index] = pixel;
}

__global__ void debugRay(int x, int y, int max_x, int max_y, Scene* scene) {
    vec3 cameraPos(600, 400, -400);
    vec3 look = vec3(0, 0, 1);
    vec3 u(1, 0, 0), v(0, 1, 0);
    
    float fov = 30;
    float aspectratio = max_x / float(max_y);
    float angle = tan(M_PI * 0.5 * fov / 180);
    float xx = (2 * ((x + 0.5) / float(max_x)) - 1) * angle * aspectratio;
    float yy = (1 - 2 * ((y + 0.5) / float(max_y))) * angle;
    vec3 dir = unit_vector(vec3(xx, -yy, 0.5));
    ray r(cameraPos, dir);
    
    vec3 col = color(r, scene, true);
}

int main() {
    int nx = 1200;
    int ny = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    //parse
    Scene* scene = parse();

    int num_pixels = nx * ny;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    // allocate FB
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
    dim3 threads(tx, ty);
    //render << <blocks, threads >> > (fb, nx, ny);
    render<<<blocks, threads>>>(fb, nx, ny, scene, 3);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image

    std::cout << "Enter name for output file: ";
    std::string filename;
    std::cin >> filename;
    filename += ".ppm";
    std::ofstream outfile(filename.c_str());

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

    std::cout << "Would you like to enter debug mode (y/n)? ";
    std::string response;
    std::cin >> response;
    if (response.size() > 0 && (response[0] == 'y' || response[0] == 'Y')) {
        int x, y;
        std::cin >> x >> y;
        while (x >= 0 && y >= 0) {
            debugRay<<<1, 1>>>(x, y, nx, ny, scene);
            checkCudaErrors(cudaDeviceSynchronize());
            std::cin >> x >> y;
        }
    }

    checkCudaErrors(cudaFree(fb));
}

