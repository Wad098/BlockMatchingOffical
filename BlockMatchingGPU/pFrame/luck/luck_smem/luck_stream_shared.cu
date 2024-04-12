#include <iostream>
#include "util_stream.cu"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ int calculatePadSAD(unsigned char curr_block[BLOCK_SIZE][BLOCK_SIZE], unsigned char ref_block[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE],
                               int motionVectorX, int motionVectorY) {
    // printf("x_idx: %d, y_idx: %d, x_ref: %d, y_ref: %d\n", x_idx, y_idx, x_ref, y_ref);
    int sad = 0;
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            sad += abs(curr_block[i][j] - ref_block[i + motionVectorY][j + motionVectorX]);
        }
    }
    // printf("(%d, %d), (%d, %d),sad:%d\n", x_idx, y_idx, x_ref, y_ref, sad);
    // printf("(%d, %d), (%d, %d),sad:%d\n", x_idx, y_idx, x_ref, y_ref, sad);
    return sad;
}

__global__ void constructFrame(unsigned char *d_curr_frame, unsigned char *d_ref_frame,
                          unsigned char *d_temp_frame, int *d_motion_vectors)
{
    __shared__ int s_Sad[CUDA_BLOCK_SIZE]; // 17 * 17
    __shared__ int s_mv_x[CUDA_BLOCK_SIZE];
    __shared__ int s_mv_y[CUDA_BLOCK_SIZE];

    __shared__ unsigned char ref_block[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE]; //size = block_length + 2 * search_range = 16 + 2*8 = 32
    __shared__ unsigned char curr_block[BLOCK_SIZE][BLOCK_SIZE];

    int y_block = blockIdx.x; // row
    int x_block = blockIdx.y; // col

    int x_idx_p = x_block * BLOCK_SIZE + SEARCH_RANGE;
    int y_idx_p = y_block * BLOCK_SIZE + SEARCH_RANGE;

    int threadIdX = threadIdx.x; // 获取当前thread的X索引, row
    int threadIdY = threadIdx.y; // 获取当前thread的Y索引, col
    int currIdx = threadIdY * CUDA_BLOCK_LENGTH + threadIdX;

    int motionVectorX = (int)threadIdX; // - SEARCH_RANGE;
    int motionVectorY = (int)threadIdY; // - SEARCH_RANGE;

    //int x_ref = x_idx_p + motionVectorX;
    //int y_ref = y_idx_p + motionVectorY;

    int indXt = x_idx_p + threadIdY;
    int indYt = y_idx_p + threadIdX;

    s_mv_x[currIdx] = motionVectorY;
    s_mv_y[currIdx] = motionVectorX;

    if(threadIdX < BLOCK_SIZE && threadIdY < BLOCK_SIZE) {
        curr_block[threadIdX][threadIdY] = d_curr_frame[indYt * PADWIDTH + indXt];
        
        for(int i = 0; i < 4; i++) {
            ref_block[threadIdX + DIRECTION1_4[i][0]][threadIdY + DIRECTION1_4[i][1]] = d_ref_frame[(indYt + DIRECTION2_4[i][0]) * PADWIDTH + (indXt + DIRECTION2_4[i][1])];
        }  
    }
    __syncthreads();    

    s_Sad[currIdx] = calculatePadSAD(curr_block, ref_block, motionVectorX, motionVectorY); //

    __syncthreads();
    
    for (int stride = 2; stride < CUDA_BLOCK_SIZE; stride *= 2)
    {
        
        if (currIdx % stride == 0)
        {
            int targetIdx = currIdx + stride / 2;
            if (targetIdx < CUDA_BLOCK_SIZE && s_Sad[currIdx] > s_Sad[targetIdx])
            {
                s_Sad[currIdx] = s_Sad[targetIdx];
                s_mv_x[currIdx] = s_mv_x[targetIdx];
                s_mv_y[currIdx] = s_mv_y[targetIdx];

            }
        }
        __syncthreads();
    }    

    if (threadIdX < BLOCK_SIZE && threadIdY < BLOCK_SIZE)
    {
        

        // 确保 indXt 和 indYt 在有效范围内
        if (indXt >= 0 && indXt < PADWIDTH && indYt >= 0 && indYt < PADHEIGHT)
        {
            d_temp_frame[indYt * PADWIDTH + indXt] = ref_block[threadIdX + s_mv_y[0]][threadIdY + s_mv_x[0]];
        }
    }
}

void block_match_full_frame_stream_save_reconst(Frame* allYFrames, unsigned char* allRconstFrames) {
    //int *h_motion_vectors;

    unsigned char *d_curr_frame[STREAM_NUM];
    unsigned char *d_ref_frame[STREAM_NUM];
    unsigned char *d_temp_frame[STREAM_NUM]; // used for both prediction frame and reconst frame
    int *d_motion_vectors[STREAM_NUM];

    // int padedWidth = WIDTH + 2 * PADDING;
    // int padedHeight = HEIGHT + 2 * PADDING;

    int x_block_num = WIDTH / BLOCK_SIZE;
    int y_block_num = HEIGHT / BLOCK_SIZE;

    //size_t pixel_num = WIDTH * HEIGHT;
    size_t pixel_num = PADWIDTH * PADHEIGHT;
    size_t gop_size = GOP * pixel_num;
    //size_t frame_size = WIDTH * HEIGHT * sizeof(unsigned char);
    size_t frame_pad_size = PADWIDTH * PADHEIGHT * sizeof(unsigned char);
    size_t sad_size = x_block_num * y_block_num * sizeof(int);

    
    //h_motion_vectors = (int *)malloc(2 * sad_size);
    //h_temp_frame = (unsigned char *)malloc(frame_size);

    for (int i = 0; i < STREAM_NUM; i++) {
        cudaMalloc((void **)&d_curr_frame[i], frame_pad_size);
        cudaMalloc((void **)&d_ref_frame[i], frame_pad_size);
        cudaMalloc((void **)&d_temp_frame[i], frame_pad_size);
        cudaMemset(&d_temp_frame[i], 72, PADWIDTH * PADHEIGHT);
        cudaMalloc((void **)&d_motion_vectors[i], 2 * sad_size);
    }


    // Blocks configuration
    dim3 blocks(HEIGHT / BLOCK_SIZE, WIDTH / BLOCK_SIZE);
    dim3 threadsPerBlock(CUDA_BLOCK_LENGTH, CUDA_BLOCK_LENGTH);

    // stream start
    cudaStream_t stream[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamCreate(&stream[i]);
    }

    for (int i = 0; i < STREAM_NUM; i++) {

        cudaMemcpyAsync(d_curr_frame[i], allYFrames[0 + GOP *i].y, frame_pad_size, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(d_ref_frame[i], allYFrames[0 + GOP *i].y, frame_pad_size, cudaMemcpyHostToDevice, stream[i]);
        
        // Launch the full search kernel
        constructFrame<<<blocks, threadsPerBlock, 0, stream[i]>>>(d_curr_frame[i], d_ref_frame[i], d_temp_frame[i], d_motion_vectors[i]);

        cudaMemcpyAsync(allRconstFrames + gop_size * i, d_temp_frame[i], frame_pad_size, cudaMemcpyDeviceToHost, stream[i]);
    }

    /**/
    for(int j = 1; j < GOP; j++) {
        for (int i = 0; i < STREAM_NUM; i++) {
            
            //if(j % 10 == 1) printf("%d %d\n", i, j);
            cudaMemcpyAsync(d_curr_frame[i], allYFrames[j + GOP *i].y, frame_pad_size, cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(d_ref_frame[i], allYFrames[j-1 + GOP *i].y, frame_pad_size, cudaMemcpyHostToDevice, stream[i]);
            //cudaMemcpyAsync(d_ref_frame[i], d_temp_frame[i], frame_pad_size, cudaMemcpyDeviceToDevice, stream[i]);

            constructFrame<<<blocks, threadsPerBlock, 0, stream[i]>>>(d_curr_frame[i], d_ref_frame[i], d_temp_frame[i], d_motion_vectors[i]);

            cudaMemcpyAsync(allRconstFrames + pixel_num * j + gop_size * i, d_temp_frame[i], frame_pad_size, cudaMemcpyDeviceToHost, stream[i]);
        }
    }
    

    //cudaMemcpy(h_motion_vectors, d_motion_vectors, 2 * sad_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_temp_frame, d_temp_frame, frame_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < STREAM_NUM; i++) {
        cudaFree(d_curr_frame[i]);
        cudaFree(d_ref_frame[i]);
        cudaFree(d_motion_vectors[i]);
        cudaFree(d_temp_frame[i]);
    }

    // Free memory
}

int main()
{
    double start, end;
    start = getTimeStamp();

    Frame *allYFrames = process_yuv_frames("long_input.yuv", WIDTH, HEIGHT, FRAME_NUM);

    end = getTimeStamp();
    printf("File reading time: %f\n", end - start);
    start = getTimeStamp();
    
    //int mVSize = (int)(PADHEIGHT / BLOCK_SIZE) * (int)(PADWIDTH / BLOCK_SIZE) * 2;
    int frameSize = PADWIDTH * PADHEIGHT;
    unsigned char *allRconstFrames = (unsigned char *)malloc(FRAME_NUM * frameSize * sizeof(unsigned char));

    block_match_full_frame_stream_save_reconst(allYFrames, allRconstFrames);

    end = getTimeStamp();
    printf("Processing time: %f\n", end - start);

    /*
    FILE *recon_yFrameFile = fopen("AllReconYFrames.txt", "w"); 
    if (recon_yFrameFile == NULL) {
        printf("Error opening Y frame file.\n");
        return -1;
    }
    saveAllFrameToText(recon_yFrameFile, allRconstFrames);

    fclose(recon_yFrameFile);
    */

    cudaDeviceReset();

    free(allRconstFrames);
}