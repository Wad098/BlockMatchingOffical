//inherited from full_search2.cu
//added stream

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8
#define FRAME_NUM 300
#define WIDTH 352 // 1920
#define HEIGHT 288 // 1080
#define SEARCH_RANGE 16
#define QP 6
#define GOP 15
#define STREAM_NUM (int)((FRAME_NUM + GOP - 1) / GOP)

__device__  const int RANGE_8[288][2] = {
    {-1, 0}, {0, -1}, {0, 1}, {1, 0}, {-2, 0}, {-1, -1}, {-1, 1}, {0, -2}, {0, 2}, {1, -1}, {1, 1}, {2, 0}, {-3, 0}, {-2, -1}, {-2, 1}, {-1, -2},
    {-1, 2}, {0, -3}, {0, 3}, {1, -2}, {1, 2}, {2, -1}, {2, 1}, {3, 0}, {-4, 0}, {-3, -1}, {-3, 1}, {-2, -2}, {-2, 2}, {-1, -3}, {-1, 3}, {0, -4}, {0, 4},
    {1, -3}, {1, 3}, {2, -2}, {2, 2}, {3, -1}, {3, 1}, {4, 0}, {-5, 0}, {-4, -1}, {-4, 1}, {-3, -2}, {-3, 2}, {-2, -3}, {-2, 3}, {-1, -4}, {-1, 4}, {0, -5},
    {0, 5}, {1, -4}, {1, 4}, {2, -3}, {2, 3}, {3, -2}, {3, 2}, {4, -1}, {4, 1}, {5, 0}, {-6, 0}, {-5, -1}, {-5, 1}, {-4, -2}, {-4, 2}, {-3, -3}, {-3, 3},
    {-2, -4}, {-2, 4}, {-1, -5}, {-1, 5}, {0, -6}, {0, 6}, {1, -5}, {1, 5}, {2, -4}, {2, 4}, {3, -3}, {3, 3}, {4, -2}, {4, 2}, {5, -1}, {5, 1}, {6, 0},
    {-7, 0}, {-6, -1}, {-6, 1}, {-5, -2}, {-5, 2}, {-4, -3}, {-4, 3}, {-3, -4}, {-3, 4}, {-2, -5}, {-2, 5}, {-1, -6}, {-1, 6}, {0, -7}, {0, 7}, {1, -6}, {1, 6},
    {2, -5}, {2, 5}, {3, -4}, {3, 4}, {4, -3}, {4, 3}, {5, -2}, {5, 2}, {6, -1}, {6, 1}, {7, 0}, {-8, 0}, {-7, -1}, {-7, 1}, {-6, -2}, {-6, 2}, {-5, -3},
    {-5, 3}, {-4, -4}, {-4, 4}, {-3, -5}, {-3, 5}, {-2, -6}, {-2, 6}, {-1, -7}, {-1, 7}, {0, -8}, {0, 8}, {1, -7}, {1, 7}, {2, -6}, {2, 6}, {3, -5}, {3, 5},
    {4, -4}, {4, 4}, {5, -3}, {5, 3}, {6, -2}, {6, 2}, {7, -1}, {7, 1}, {8, 0}, {-8, -1}, {-8, 1}, {-7, -2}, {-7, 2}, {-6, -3}, {-6, 3}, {-5, -4}, {-5, 4},
    {-4, -5}, {-4, 5}, {-3, -6}, {-3, 6}, {-2, -7}, {-2, 7}, {-1, -8}, {-1, 8}, {1, -8}, {1, 8}, {2, -7}, {2, 7}, {3, -6}, {3, 6}, {4, -5}, {4, 5}, {5, -4},
    {5, 4}, {6, -3}, {6, 3}, {7, -2}, {7, 2}, {8, -1}, {8, 1}, {-8, -2}, {-8, 2}, {-7, -3}, {-7, 3}, {-6, -4}, {-6, 4}, {-5, -5}, {-5, 5}, {-4, -6}, {-4, 6},
    {-3, -7}, {-3, 7}, {-2, -8}, {-2, 8}, {2, -8}, {2, 8}, {3, -7}, {3, 7}, {4, -6}, {4, 6}, {5, -5}, {5, 5}, {6, -4}, {6, 4}, {7, -3}, {7, 3}, {8, -2},
    {8, 2}, {-8, -3}, {-8, 3}, {-7, -4}, {-7, 4}, {-6, -5}, {-6, 5}, {-5, -6}, {-5, 6}, {-4, -7}, {-4, 7}, {-3, -8}, {-3, 8}, {3, -8}, {3, 8}, {4, -7}, {4, 7},
    {5, -6}, {5, 6}, {6, -5}, {6, 5}, {7, -4}, {7, 4}, {8, -3}, {8, 3}, {-8, -4}, {-8, 4}, {-7, -5}, {-7, 5}, {-6, -6}, {-6, 6}, {-5, -7}, {-5, 7}, {-4, -8},
    {-4, 8}, {4, -8}, {4, 8}, {5, -7}, {5, 7}, {6, -6}, {6, 6}, {7, -5}, {7, 5}, {8, -4}, {8, 4}, {-8, -5}, {-8, 5}, {-7, -6}, {-7, 6}, {-6, -7}, {-6, 7},
    {-5, -8}, {-5, 8}, {5, -8}, {5, 8}, {6, -7}, {6, 7}, {7, -6}, {7, 6}, {8, -5}, {8, 5}, {-8, -6}, {-8, 6}, {-7, -7}, {-7, 7}, {-6, -8}, {-6, 8}, {6, -8},
    {6, 8}, {7, -7}, {7, 7}, {8, -6}, {8, 6}, {-8, -7}, {-8, 7}, {-7, -8}, {-7, 8}, {7, -8}, {7, 8}, {8, -7}, {8, 7}, {-8, -8}, {-8, 8}, {8, -8}, {8, 8}
};

typedef struct
{
    unsigned char *y;
    int frame_index;
    int width;
    int height;
} Frame;

__host__ Frame *load_y(FILE *file, int width, int height)
{
    Frame *frame = (Frame *)malloc(sizeof(Frame));
    frame->y = (unsigned char *)malloc((size_t)(width * height) * sizeof(unsigned char));

    fread(frame->y, 1, width * height, file);

    // Skip U and V components
    fseek(file, width * height / 2, SEEK_CUR);

    return frame;
}

Frame *process_yuv_frames(const char *file_path, int width, int height, int FRAME_NUMs)
{
    FILE *file = fopen(file_path, "rb");
    if (file == NULL)
    {
        printf("Cannot open file.\n");
        return NULL;
    }

    Frame *frames = (Frame *)malloc(FRAME_NUMs * sizeof(Frame));

    for (int i = 0; i < FRAME_NUMs; i++)
    {
        Frame *frame = load_y(file, width, height);
        frame->frame_index = i;
        frame->width = width;
        frame->height = height;
        frames[i] = *frame;
        free(frame);
    }

    fclose(file);

    return frames;
}

__device__ void printMatrix(float matrix[BLOCK_SIZE][BLOCK_SIZE]) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            printf("%0.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

__device__ int calculateSAD(unsigned char *d_curr_frame, unsigned char *d_ref_frame,
                            int x_idx, int y_idx, int x_ref, int y_ref) {
    int sad = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            int curr_pixel = d_curr_frame[(y_idx + j) * WIDTH + (x_idx + i)];
            int ref_pixel = d_ref_frame[(y_ref + j) * WIDTH + (x_ref + i)];
            sad += abs(curr_pixel - ref_pixel);
        }
    }
    return sad;
}



__device__ void generatePredictedBlock(unsigned char *d_temp_frame, unsigned char *prevFrame,
        int best_x, int best_y, int x_idx, int y_idx) {
    for (int y = y_idx; y < y_idx + BLOCK_SIZE; y++) {
        for (int x = x_idx; x < x_idx + BLOCK_SIZE; x++) {
            int srcX = x + best_x;
            int srcY = y + best_y;

            // if (srcX >= 0 && srcX < WIDTH && srcY >= 0 && srcY < HEIGHT) {}
            d_temp_frame[y * WIDTH + x] = prevFrame[srcY * WIDTH + srcX];
            
        }
    }
}

__device__ void generatePredictedBlockWithSync(unsigned char *d_temp_frame, unsigned char *d_curr_frame,
        int best_x, int best_y, int x_idx, int y_idx) {
    for (int y = y_idx; y < y_idx + BLOCK_SIZE; y++) {
        for (int x = x_idx; x < x_idx + BLOCK_SIZE; x++) {
            int srcX = x + best_x;
            int srcY = y + best_y;

            // if (srcX >= 0 && srcX < WIDTH && srcY >= 0 && srcY < HEIGHT) {}
            d_curr_frame[y * WIDTH + x] = d_temp_frame[srcY * WIDTH + srcX];
            
        }
    }

    __syncthreads();

    for (int y = y_idx; y < y_idx + BLOCK_SIZE; y++) {
        for (int x = x_idx; x < x_idx + BLOCK_SIZE; x++) {
            d_temp_frame[y * WIDTH + x] = d_curr_frame[y * WIDTH + x];
            
        }
    }
}

__global__ void firstPFrameReconstGPU (unsigned char *d_curr_frame, unsigned char *d_ref_frame,
        unsigned char *d_temp_frame, int *d_motion_vectors, float qMatrix[BLOCK_SIZE][BLOCK_SIZE]) {
    int y_block = blockIdx.x * blockDim.x + threadIdx.x; // row
    int x_block = blockIdx.y * blockDim.y + threadIdx.y; // col

    int x_idx = x_block * BLOCK_SIZE;
    int y_idx = y_block * BLOCK_SIZE;

    int right_bound = WIDTH - BLOCK_SIZE;
    int bottom_bound = HEIGHT - BLOCK_SIZE;
    int row_block_num = WIDTH / BLOCK_SIZE;

    if (x_idx <= right_bound && y_idx <= bottom_bound) {
        int best_sad = calculateSAD(d_curr_frame, d_ref_frame, x_idx, y_idx, x_idx, y_idx);
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
                int sad = calculateSAD(d_curr_frame, d_ref_frame, x_idx, y_idx, x_ref, y_ref);
                
                if(y_block * row_block_num + x_block == 77) {
                    //printf("%d %d %d\n", i, j, sad);
                }

                if (sad < best_sad) {
                    best_sad = sad;
                    best_x = RANGE_8[i][1];
                    best_y = RANGE_8[i][0];
                }
            }
        }

        generatePredictedBlock(d_temp_frame, d_ref_frame, best_x, best_y, x_idx, y_idx);
        //generateReconstructedBlock(d_temp_frame, d_curr_frame, qMatrix, x_idx, y_idx);


        // Store the motion vector
        
        int block_idx = y_block * row_block_num + x_block;
        d_motion_vectors[block_idx * 2] = best_y;
        d_motion_vectors[block_idx * 2 + 1] = best_x;
    }
}

__global__ void nextPFrameReconstGPU (unsigned char *d_curr_frame,
        unsigned char *d_temp_frame, int *d_motion_vectors, float qMatrix[BLOCK_SIZE][BLOCK_SIZE]) {
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

        generatePredictedBlockWithSync(d_temp_frame, d_curr_frame, best_x, best_y, x_idx, y_idx);
        //generateReconstructedBlock(d_temp_frame, d_curr_frame, qMatrix, x_idx, y_idx);


        // Store the motion vector
        
        int block_idx = y_block * row_block_num + x_block;
        d_motion_vectors[block_idx * 2] = best_y;
        d_motion_vectors[block_idx * 2 + 1] = best_x;
    }
}

__host__ void initQuantizationMatrix(float qMatrix[BLOCK_SIZE][BLOCK_SIZE])
{
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            if (i + j < BLOCK_SIZE - 1)
            {
                qMatrix[i][j] = pow(2, QP);
            }
            else if (i + j == BLOCK_SIZE - 1)
            {
                qMatrix[i][j] = pow(2, QP + 1);
            }
            else
            {
                qMatrix[i][j] = pow(2, QP + 2);
            }
        }
    }
}

void saveAllFrameToText(FILE *file, unsigned char *frame) {
    if (file == NULL) {
        printf("File pointer is NULL.\n");
        return;
    }

    int frame_size = WIDTH * HEIGHT;

    for (int frameNumber = 0; frameNumber < FRAME_NUM; frameNumber++) {
        fprintf(file, "Frame %d:\n", frameNumber + 1); 
        
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                fprintf(file, "%d ", frame[y * WIDTH + x + frame_size * frameNumber]); 
            }
            fprintf(file, "\n"); 
        }
        
        fprintf(file, "\n\n"); 
    }
}

void block_match_full_frame_stream(Frame* allYFrames, unsigned char* allRconstFrames, float qMatrix[BLOCK_SIZE][BLOCK_SIZE]) {
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
        firstPFrameReconstGPU<<<block, grid, 0, stream[i]>>>(d_curr_frame[i], d_ref_frame[i], d_temp_frame[i], d_motion_vectors[i], qMatrix);

        cudaMemcpyAsync(allRconstFrames + pixel_num + gop_size * i, d_temp_frame[i], frame_size, cudaMemcpyDeviceToHost, stream[i]);
        //cudaStreamSynchronize(stream[i]);

        for(int j = 2; j < GOP; j++) {
            cudaMemcpyAsync(d_curr_frame, allYFrames[j + GOP *i].y, frame_size, cudaMemcpyHostToDevice, stream[i]);

            nextPFrameReconstGPU<<<block, grid, 0, stream[i]>>>(d_curr_frame[i], d_temp_frame[i], d_motion_vectors[i], qMatrix);

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


int main() {
    Frame* allYFrames = process_yuv_frames("foreman_cif-1.yuv", WIDTH, HEIGHT, FRAME_NUM);

    int frameSize = WIDTH * HEIGHT;

    unsigned char* allRconstFrames = (unsigned char*)malloc(FRAME_NUM * frameSize);
    
    float qMatrix[BLOCK_SIZE][BLOCK_SIZE]; 
    initQuantizationMatrix(qMatrix);

    block_match_full_frame_stream(allYFrames, allRconstFrames, qMatrix);




    /*
    FILE* file = fopen("foreman_cif-1.yuv", "rb");
    if (file == NULL) {
        printf("Error opening file.\n");
        return -1;
    }

    FILE *yFrameFile = fopen("AllYFrames.txt", "w"); 
    if (yFrameFile == NULL) {
        printf("Error opening Y frame file.\n");
        return -1;
    }

    int frameSize = WIDTH * HEIGHT + (WIDTH * HEIGHT) / 2; // YUV 4:2:0
    int yPlaneSize = WIDTH * HEIGHT;

    

    float qMatrix[BLOCK_SIZE][BLOCK_SIZE]; 
    initQuantizationMatrix(qMatrix);

    // Allocate memory
    unsigned char* FrameInGOP[GOP];
    unsigned char* yuvBuffer = (unsigned char*)malloc(frameSize);

    for (int i = 0; i < GOP; i++) {
        FrameInGOP[i] = (unsigned char*)malloc(yPlaneSize);
    }
    int frameCount = 1; // frame count in gop (e.g. range 0-14 of GOP 15)
    int GOPNum = 0;

    bool endFlag = (fread(yuvBuffer, 1, frameSize, file) == frameSize);
    memcpy(FrameInGOP[0], yuvBuffer, yPlaneSize);

    while (endFlag == 1){
        endFlag = (fread(yuvBuffer, 1, frameSize, file) == frameSize);
        memcpy(FrameInGOP[frameCount], yuvBuffer, yPlaneSize);
        frameCount++;

        if (frameCount == 15 || endFlag == 0) {
            frameCount = 0;
            unsigned char *reconst_frame = block_match_full_frame_stream(FrameInGOP, qMatrix);

            GOPNum++;
        }
    }
    
    

    FILE* file1 = fopen("reconst_GPU.txt", "w");
    for (size_t i = 0; i < HEIGHT; i++) {
        for (size_t j = 0; j < WIDTH; j++) {
            fprintf(file1, "%d ", int(reconst_frame[WIDTH * i + j]));
        }
        fprintf(file1, "\n");
    }
    fclose(file1);
    */

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
