#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#include <string>

__device__ void load_image_tile(unsigned char* smem, unsigned char* d_image, const int width, const int height, const int channels, const int radius, const int row, const int col, const int smem_index, const int img_index){
    const int halo_step = radius * channels;
    const int pixel_width = blockDim.x  + 2 * radius;

    // READ CENTER PIXELS
    for(int i=0; i<channels; i++)
        smem[smem_index + i] = (row < height && col < width) ? d_image[img_index + i] : 0;

    // READ HALO SIDES
    if(threadIdx.x < radius)
        for(int i=0; i<channels; i++)
            smem[smem_index - halo_step + i] = 
                (col - radius >= 0)
                ? d_image[img_index - halo_step + i]
                : 0;
    
    if(threadIdx.x >= blockDim.x - radius)
        for(int i=0; i<channels; i++)
                smem[smem_index + halo_step + i] = 
                    (col + radius < width)
                    ? d_image[img_index + halo_step + i]
                    : 0;

    if(threadIdx.y < radius)
        for(int i=0; i<channels; i++)
            smem[smem_index - (halo_step * pixel_width) + i] = 
                (row - radius >= 0)
                ? d_image[img_index - (halo_step * width) + i]
                : 0;
    
    if(threadIdx.y >= blockDim.y - radius)
        for(int i=0; i<channels; i++)
            smem[smem_index + (halo_step * pixel_width) + i] = 
                (row + radius < height)
                ? d_image[img_index + (halo_step * width) + i]
                : 0;


    // READ HALO CORNERS
    if(threadIdx.x < radius && threadIdx.y < radius)
        for(int i=0; i<channels; i++)
            smem[smem_index - halo_step * (1 + pixel_width) + i] = 
                (col - radius >= 0 && row - radius >= 0)
                ? d_image[img_index - halo_step * (1 + width) + i]
                : 0;

    if(threadIdx.x >= blockDim.x - radius && threadIdx.y < radius)
        for(int i=0; i<channels; i++)
            smem[smem_index + halo_step * (1 - pixel_width) + i] = 
                (col + radius < width && row - radius >= 0)
                ? d_image[img_index + halo_step * (1 - width) + i]
                : 0;

    if(threadIdx.x < radius && threadIdx.y >= blockDim.y - radius)
        for(int i=0; i<channels; i++)
            smem[smem_index - halo_step * (1 - pixel_width) + i] = 
                (col - radius >= 0 && row + radius < height)
                ? d_image[img_index - halo_step * (1 - width) + i]
                : 0;

    if(threadIdx.x >= blockDim.x - radius && threadIdx.y >= blockDim.y - radius)
        for(int i=0; i<channels; i++)
            smem[smem_index + halo_step * (1 + pixel_width) + i] = 
                (col + radius < width && row + radius < height)
                ? d_image[img_index + halo_step * (1 + width) + i]
                : 0;
}

__global__ void kernel_box_blur(unsigned char* d_image, unsigned char* d_output, const int width, const int height, const int channels, const int radius){
    extern __shared__ unsigned char smem[];
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    
    const int smem_index = ((radius + threadIdx.y) * (blockDim.x + 2 * radius) + radius + threadIdx.x) * channels;
    const int img_index = (row * width + col) * channels;
    load_image_tile(smem, d_image, width, height, channels, radius, row, col, smem_index, img_index);
    __syncthreads();

    int box_size = (2 * radius + 1) * (2 * radius + 1);
    int temp_sum[] = {0,0,0,0};
    if(row < height && col < width){
        for(int i=-radius; i<=radius; i++)
            for(int j=-radius; j<=radius; j++)
                for(int c=0; c<channels; c++)
                    temp_sum[c] += smem[smem_index + (i * (blockDim.x + 2 * radius) + j) * channels + c];
        for(int c=0; c<channels; c++)
            d_output[img_index + c] = temp_sum[c] / box_size;
    }
}

void blur(unsigned char* image, const int width, const int height, const int channels, const int radius){
    //manage host and device memory
    size_t size = width * height * channels;
    unsigned char* d_image, * d_output;
    cudaMalloc(&d_image, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, size);

    //call kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    size_t sharedMemSize = (blockSize.x + 2 * radius) * (blockSize.y + 2 * radius) * channels * sizeof(unsigned char);
    kernel_box_blur<<<gridSize, blockSize, sharedMemSize>>>(d_image, d_output, width , height, channels, radius);

    cudaMemcpy(image, d_output, size, cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(d_image); cudaFree(d_output);
}

int main(int argc, char** argv){
    if (argc != 4) { printf("Usage: %s <image_path> <radius> <blurrs>\n", argv[0]); return 1; }

    int width, height, channels;

    std::string name = argv[1];
    unsigned char* image = stbi_load(name.c_str(), &width, &height, &channels, 0);
    if (!image) { fprintf(stderr, "Error loading image\n"); return 1; }
    const int radius = atoi(argv[2]);
    const int blurrs = atoi(argv[3]);

    for(int i=0; i<blurrs; i++)
        blur(image, width, height, channels, radius);

    size_t sep = name.find_last_of("/\\");
    std::string dir = name.substr(0, sep + 1);
    std::string base = name.substr(sep + 1);
    std::string output_name = dir + "blur_" + base;
    stbi_write_png(output_name.c_str(), width, height, channels, image, width*channels);
    printf("%s created\n", output_name.c_str());
    stbi_image_free(image);
}