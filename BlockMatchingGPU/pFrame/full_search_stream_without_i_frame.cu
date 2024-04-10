// inherited from full_search2.cu
// added stream
#include "util.cu"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__device__ void generatePredictedBlock(unsigned char *d_temp_frame, unsigned char *prevFrame,
                                       int best_x, int best_y, int x_idx, int y_idx)
{
    for (int y = y_idx; y < y_idx + BLOCK_SIZE; y++)
    {
        for (int x = x_idx; x < x_idx + BLOCK_SIZE; x++)
        {
            int srcX = x + best_x;
            int srcY = y + best_y;

            // if (srcX >= 0 && srcX < WIDTH && srcY >= 0 && srcY < HEIGHT) {}
            d_temp_frame[y * WIDTH + x] = prevFrame[srcY * WIDTH + srcX];
        }
    }
}

__device__ void generatePredictedBlockWithSync(unsigned char *d_temp_frame, unsigned char *d_ref_frame,
        int best_x, int best_y, int x_idx, int y_idx) {
    for (int y = y_idx; y < y_idx + BLOCK_SIZE; y++) {
        for (int x = x_idx; x < x_idx + BLOCK_SIZE; x++) {
            int srcX = x + best_x;
            int srcY = y + best_y;

            // if (srcX >= 0 && srcX < WIDTH && srcY >= 0 && srcY < HEIGHT) {}
            d_ref_frame[y * WIDTH + x] = d_temp_frame[srcY * WIDTH + srcX];
            
        }
    }

    __syncthreads();

    for (int y = y_idx; y < y_idx + BLOCK_SIZE; y++) {
        for (int x = x_idx; x < x_idx + BLOCK_SIZE; x++) {
            d_temp_frame[y * WIDTH + x] = d_ref_frame[y * WIDTH + x];
            
        }
    }
}

__device__ void generateReconstructedBlock(unsigned char *d_temp_frame, unsigned char *currFrame,
                                           int x_idx, int y_idx)
{

    float block[BLOCK_SIZE][BLOCK_SIZE];

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            block[i][j] = (float)(currFrame[(y_idx + i) * WIDTH + (x_idx + j)] - d_temp_frame[(y_idx + i) * WIDTH + (x_idx + j)]);
        }
    }
    blockManipulation(block);

    // reconstructedFrame
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            int val = (int)block[i][j] + d_temp_frame[(y_idx + i) * WIDTH + (x_idx + j)];
            d_temp_frame[(y_idx + i) * WIDTH + (x_idx + j)] = (unsigned char)fmaxf(0, fminf(255, val));
        }
    }
}

__global__ void firstPFrameReconstGPU(unsigned char *d_curr_frame, unsigned char *d_ref_frame,
                                      unsigned char *d_temp_frame, int *d_motion_vectors)
{
    int y_block = blockIdx.x * blockDim.x + threadIdx.x; // row
    int x_block = blockIdx.y * blockDim.y + threadIdx.y; // col

    int x_idx = x_block * BLOCK_SIZE;
    int y_idx = y_block * BLOCK_SIZE;

    int right_bound = WIDTH - BLOCK_SIZE;
    int bottom_bound = HEIGHT - BLOCK_SIZE;
    int row_block_num = WIDTH / BLOCK_SIZE;

    if (x_idx <= right_bound && y_idx <= bottom_bound)
    {
        int best_sad = calculateSAD(d_curr_frame, d_ref_frame, x_idx, y_idx, x_idx, y_idx);
        // motion vertor is relative distance
        int best_x = 0;
        int best_y = 0;

        // full search
        int search_size = SEARCH_RANGE * 2 + 1;         // 17
        int search_num = search_size * search_size - 1; // 288
        for (int i = 0; i <= search_num; i++)
        {
            
            int x_ref = x_idx + RANGE_8[i][1];
            int y_ref = y_idx + RANGE_8[i][0];

            if (x_ref >= 0 && x_ref <= right_bound && y_ref >= 0 && y_ref <= bottom_bound)
            {
                int sad = calculateSAD(d_curr_frame, d_ref_frame, x_idx, y_idx, x_ref, y_ref);

                // if (y_block * row_block_num + x_block == 77)
                // {
                //     // printf("%d %d %d\n", i, j, sad);
                // }

                if (sad < best_sad)
                {
                    best_sad = sad;
                    best_x = RANGE_8[i][1];
                    best_y = RANGE_8[i][0];
                }
            }
        }

        // printf("In x %d, y %d: Best x: %d, Best y: %d, Best SAD: %d\n", x_block, y_block, best_x, best_y, best_sad);
        generatePredictedBlock(d_temp_frame, d_ref_frame, best_x, best_y, x_idx, y_idx);
        generateReconstructedBlock(d_temp_frame, d_curr_frame, x_idx, y_idx);
        __syncthreads();

        int block_idx = y_block * row_block_num + x_block;
        d_motion_vectors[block_idx * 2] = best_y;
        d_motion_vectors[block_idx * 2 + 1] = best_x;
    }
}

__global__ void nextPFrameReconstGPU (unsigned char *d_curr_frame, unsigned char *d_ref_frame,
                                      unsigned char *d_temp_frame, int *d_motion_vectors)
{
    int y_block = blockIdx.x * blockDim.x + threadIdx.x; // row
    int x_block = blockIdx.y * blockDim.y + threadIdx.y; // col

    int x_idx = x_block * BLOCK_SIZE;
    int y_idx = y_block * BLOCK_SIZE;

    int right_bound = WIDTH - BLOCK_SIZE;
    int bottom_bound = HEIGHT - BLOCK_SIZE;
    int row_block_num = WIDTH / BLOCK_SIZE;

    if (x_idx <= right_bound && y_idx <= bottom_bound) {
        int best_sad = calculateSAD(d_curr_frame, d_temp_frame, x_idx, y_idx, x_idx, y_idx);
        // motion vertor is relative distance
        int best_x = 0;
        int best_y = 0;

        // full search
        int search_size = SEARCH_RANGE * 2 + 1; // 17
        int search_num = search_size * search_size - 1; // 288
        for (int i = 0; i <= search_num; i++) {
            int x_ref = x_idx + RANGE_8[i][1];
            int y_ref = y_idx + RANGE_8[i][0];

            if (x_ref >= 0 && x_ref <= right_bound && y_ref >= 0 && y_ref <= bottom_bound) {
                int sad = calculateSAD(d_curr_frame, d_temp_frame, x_idx, y_idx, x_ref, y_ref);

                if (sad < best_sad) {
                    best_sad = sad;
                    best_x = RANGE_8[i][1];
                    best_y = RANGE_8[i][0];
                }
            }
        }

        generatePredictedBlockWithSync(d_temp_frame, d_ref_frame, best_x, best_y, x_idx, y_idx);
        generateReconstructedBlock(d_temp_frame, d_curr_frame, x_idx, y_idx);
        __syncthreads();
        
        int block_idx = y_block * row_block_num + x_block;
        d_motion_vectors[block_idx * 2] = best_y;
        d_motion_vectors[block_idx * 2 + 1] = best_x;
    }
}


void block_match_full_frame_stream(Frame* allYFrames, unsigned char* allRconstFrames) {
    //int *h_motion_vectors;

    unsigned char *d_curr_frame[STREAM_NUM];
    unsigned char *d_ref_frame[STREAM_NUM];
    unsigned char *d_temp_frame[STREAM_NUM]; // used for both prediction frame and reconst frame
    int *d_motion_vectors[STREAM_NUM];

    int x_block_num = WIDTH / BLOCK_SIZE;
    int y_block_num = HEIGHT / BLOCK_SIZE;

    size_t pixel_num = WIDTH * HEIGHT;
    size_t gop_size = GOP * pixel_num;
    size_t frame_size = WIDTH * HEIGHT * sizeof(unsigned char);
    size_t sad_size = x_block_num * y_block_num * sizeof(int);

    //h_motion_vectors = (int *)malloc(2 * sad_size);
    //h_temp_frame = (unsigned char *)malloc(frame_size);

    
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaMalloc((void **)&d_curr_frame[i], frame_size);
        cudaMalloc((void **)&d_ref_frame[i], frame_size);
        cudaMalloc((void **)&d_temp_frame[i], frame_size);
        cudaMalloc((void **)&d_motion_vectors[i], 2 * sad_size);
    }

    // Blocks configuration
    dim3 grid(8, 8);
    dim3 block((y_block_num + grid.x - 1) / grid.x, (x_block_num + grid.y - 1) / grid.y);

    
    // stream start
    cudaStream_t stream[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamCreate(&stream[i]);
    }

    for (int i = 0; i < STREAM_NUM; i++) {
        cudaMemcpyAsync(d_curr_frame[i], allYFrames[1 + GOP *i].y, frame_size, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_ref_frame[i], allYFrames[0 + GOP *i].y, frame_size, cudaMemcpyHostToDevice, stream[i]);

        // Launch the full search kernel
        firstPFrameReconstGPU<<<block, grid, 0, stream[i]>>>(d_curr_frame[i], d_ref_frame[i], d_temp_frame[i], d_motion_vectors[i]);

        cudaMemcpyAsync(allRconstFrames + pixel_num + gop_size * i, d_temp_frame[i], frame_size, cudaMemcpyDeviceToHost, stream[i]);
        //cudaStreamSynchronize(stream[i]);

        for(int j = 2; j < GOP; j++) {
            cudaMemcpyAsync(d_curr_frame[i], allYFrames[j + GOP *i].y, frame_size, cudaMemcpyHostToDevice, stream[i]);

            nextPFrameReconstGPU<<<block, grid, 0, stream[i]>>>(d_curr_frame[i], d_ref_frame[i], d_temp_frame[i], d_motion_vectors[i]);

            cudaMemcpyAsync(allRconstFrames + pixel_num * j + gop_size * i, d_temp_frame[i], frame_size, cudaMemcpyDeviceToHost, stream[i]);
            
            //cudaStreamSynchronize(stream[i]);
        }
    }



    //cudaMemcpy(h_motion_vectors, d_motion_vectors, 2 * sad_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_temp_frame, d_temp_frame, frame_size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_curr_frame);
    cudaFree(d_ref_frame);
    cudaFree(d_motion_vectors);
    cudaFree(d_temp_frame);
}

int main()
{
    float qMatrix[BLOCK_SIZE][BLOCK_SIZE];
    initQuantizationMatrix(qMatrix);
    cudaError_t err = cudaMemcpyToSymbol(d_qMatrix, qMatrix, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE);

    if (err != cudaSuccess)
    {
        printf("Failed to copy to symbol d_qMatrix: %s\n", cudaGetErrorString(err));
    }

    Frame* allYFrames = process_yuv_frames("foreman_cif-1.yuv", WIDTH, HEIGHT, FRAME_NUM);
    int frameSize = WIDTH * HEIGHT;
    unsigned char* allRconstFrames = (unsigned char*)malloc(FRAME_NUM * frameSize);

    block_match_full_frame_stream(allYFrames, allRconstFrames);

    FILE *recon_yFrameFile = fopen("AllReconYFrames.txt", "w"); 
    if (recon_yFrameFile == NULL) {
        printf("Error opening Y frame file.\n");
        return -1;
    }
    saveAllFrameToText(recon_yFrameFile, allRconstFrames);

    fclose(recon_yFrameFile);
    free(allRconstFrames);

    return 0;
}
