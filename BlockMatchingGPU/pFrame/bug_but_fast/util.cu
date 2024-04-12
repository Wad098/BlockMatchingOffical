#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WIDTH 352  // 1920
#define HEIGHT 288 // 1080

#define SEARCH_RANGE 8
#define QP 1
#define FRAME_NUM 3000
#define GOP 30
#define STREAM_NUM (int)((FRAME_NUM + GOP - 1) / GOP)
#define PADDING SEARCH_RANGE
#define PADWIDTH WIDTH + 2 * PADDING
#define PADHEIGHT HEIGHT + 2 * PADDING

__constant__ float d_qMatrix[BLOCK_SIZE][BLOCK_SIZE];

__device__ const int RANGE_8[288][2] = {
    {-1, 0}, {0, -1}, {0, 1}, {1, 0}, {-2, 0}, {-1, -1}, {-1, 1}, {0, -2}, {0, 2}, {1, -1}, {1, 1}, {2, 0}, {-3, 0}, {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {0, -3}, {0, 3}, {1, -2}, {1, 2}, {2, -1}, {2, 1}, {3, 0}, {-4, 0}, {-3, -1}, {-3, 1}, {-2, -2}, {-2, 2}, {-1, -3}, {-1, 3}, {0, -4}, {0, 4}, {1, -3}, {1, 3}, {2, -2}, {2, 2}, {3, -1}, {3, 1}, {4, 0}, {-5, 0}, {-4, -1}, {-4, 1}, {-3, -2}, {-3, 2}, {-2, -3}, {-2, 3}, {-1, -4}, {-1, 4}, {0, -5}, {0, 5}, {1, -4}, {1, 4}, {2, -3}, {2, 3}, {3, -2}, {3, 2}, {4, -1}, {4, 1}, {5, 0}, {-6, 0}, {-5, -1}, {-5, 1}, {-4, -2}, {-4, 2}, {-3, -3}, {-3, 3}, {-2, -4}, {-2, 4}, {-1, -5}, {-1, 5}, {0, -6}, {0, 6}, {1, -5}, {1, 5}, {2, -4}, {2, 4}, {3, -3}, {3, 3}, {4, -2}, {4, 2}, {5, -1}, {5, 1}, {6, 0}, {-7, 0}, {-6, -1}, {-6, 1}, {-5, -2}, {-5, 2}, {-4, -3}, {-4, 3}, {-3, -4}, {-3, 4}, {-2, -5}, {-2, 5}, {-1, -6}, {-1, 6}, {0, -7}, {0, 7}, {1, -6}, {1, 6}, {2, -5}, {2, 5}, {3, -4}, {3, 4}, {4, -3}, {4, 3}, {5, -2}, {5, 2}, {6, -1}, {6, 1}, {7, 0}, {-8, 0}, {-7, -1}, {-7, 1}, {-6, -2}, {-6, 2}, {-5, -3}, {-5, 3}, {-4, -4}, {-4, 4}, {-3, -5}, {-3, 5}, {-2, -6}, {-2, 6}, {-1, -7}, {-1, 7}, {0, -8}, {0, 8}, {1, -7}, {1, 7}, {2, -6}, {2, 6}, {3, -5}, {3, 5}, {4, -4}, {4, 4}, {5, -3}, {5, 3}, {6, -2}, {6, 2}, {7, -1}, {7, 1}, {8, 0}, {-8, -1}, {-8, 1}, {-7, -2}, {-7, 2}, {-6, -3}, {-6, 3}, {-5, -4}, {-5, 4}, {-4, -5}, {-4, 5}, {-3, -6}, {-3, 6}, {-2, -7}, {-2, 7}, {-1, -8}, {-1, 8}, {1, -8}, {1, 8}, {2, -7}, {2, 7}, {3, -6}, {3, 6}, {4, -5}, {4, 5}, {5, -4}, {5, 4}, {6, -3}, {6, 3}, {7, -2}, {7, 2}, {8, -1}, {8, 1}, {-8, -2}, {-8, 2}, {-7, -3}, {-7, 3}, {-6, -4}, {-6, 4}, {-5, -5}, {-5, 5}, {-4, -6}, {-4, 6}, {-3, -7}, {-3, 7}, {-2, -8}, {-2, 8}, {2, -8}, {2, 8}, {3, -7}, {3, 7}, {4, -6}, {4, 6}, {5, -5}, {5, 5}, {6, -4}, {6, 4}, {7, -3}, {7, 3}, {8, -2}, {8, 2}, {-8, -3}, {-8, 3}, {-7, -4}, {-7, 4}, {-6, -5}, {-6, 5}, {-5, -6}, {-5, 6}, {-4, -7}, {-4, 7}, {-3, -8}, {-3, 8}, {3, -8}, {3, 8}, {4, -7}, {4, 7}, {5, -6}, {5, 6}, {6, -5}, {6, 5}, {7, -4}, {7, 4}, {8, -3}, {8, 3}, {-8, -4}, {-8, 4}, {-7, -5}, {-7, 5}, {-6, -6}, {-6, 6}, {-5, -7}, {-5, 7}, {-4, -8}, {-4, 8}, {4, -8}, {4, 8}, {5, -7}, {5, 7}, {6, -6}, {6, 6}, {7, -5}, {7, 5}, {8, -4}, {8, 4}, {-8, -5}, {-8, 5}, {-7, -6}, {-7, 6}, {-6, -7}, {-6, 7}, {-5, -8}, {-5, 8}, {5, -8}, {5, 8}, {6, -7}, {6, 7}, {7, -6}, {7, 6}, {8, -5}, {8, 5}, {-8, -6}, {-8, 6}, {-7, -7}, {-7, 7}, {-6, -8}, {-6, 8}, {6, -8}, {6, 8}, {7, -7}, {7, 7}, {8, -6}, {8, 6}, {-8, -7}, {-8, 7}, {-7, -8}, {-7, 8}, {7, -8}, {7, 8}, {8, -7}, {8, 7}, {-8, -8}, {-8, 8}, {8, -8}, {8, 8}};

double getTimeStamp() {
        struct timeval tv;
        gettimeofday( &tv, NULL );
        return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

typedef struct
{
    unsigned char *y;
    int frame_index;
    int width;
    int height;
    // unsigned char *data;
} Frame;

__host__ Frame *load_y(FILE *file, int width, int height)
{
    Frame *frame = (Frame *)malloc(sizeof(Frame));
    frame->y = (unsigned char *)malloc((size_t)(width * height) * sizeof(unsigned char));

    fread(frame->y, 1, width * height, file);

    // Skip U and V components
    // fseek(file, width * height / 2, SEEK_CUR);

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

// Frame *process_yuv_frames(const char *file_path, int width, int height, int FRAME_NUMs)
// {
//     FILE *file = fopen(file_path, "rb");
//     if (file == NULL)
//     {
//         printf("Cannot open file.\n");
//         return NULL;
//     }

//     Frame *frames = (Frame *)malloc(FRAME_NUMs * sizeof(Frame));
//     int paddedWidth = width + 2 * 8;
//     int paddedHeight = height + 2 * 8;
//     for (int i = 0; i < FRAME_NUMs; i++)
//     {
//         Frame *frame = load_y(file, width, height);
//         frame->frame_index = i;
//         frame->width = paddedWidth;
//         frame->height = paddedHeight;
        
//         unsigned char *paddedData = (unsigned char *)malloc(paddedWidth * paddedHeight * sizeof(unsigned char));
//         memset(paddedData, 1, paddedWidth * paddedHeight);

//         for (int y = 0; y < height; y++)
//         {
//             for (int x = 0; x < width; x++)
//             {
//                 // 计算在原始帧和带padding帧中的索引
//                 int origIndex = y * width + x;
//                 int paddedIndex = (y + 8) * paddedWidth + (x + 8);
//                 paddedData[paddedIndex] = frame->data[origIndex];
//             }
//         }
//         free(frame->data); // 释放原始的frame数据内存
//         frame->data = paddedData; // 更新为带padding的数据

//         frames[i] = *frame;
//         free(frame);
//     }

//     fclose(file);

//     return frames;
// }



__device__ int calculateSAD(unsigned char *d_curr_frame, unsigned char *d_ref_frame,
                            int x_idx, int y_idx, int x_ref, int y_ref)
{
    int sad = 0;
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            int curr_pixel = d_curr_frame[(y_idx + j) * WIDTH + (x_idx + i)];
            int ref_pixel = d_ref_frame[(y_ref + j) * WIDTH + (x_ref + i)];
            sad += abs(curr_pixel - ref_pixel);
        }
    }
    return sad;
}

__device__ int calculatePadSAD(unsigned char *d_curr_frame, unsigned char *d_ref_pad_frame,
                            int x_idx, int y_idx, int x_ref, int y_ref)
{
    int sad = 0;
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            int curr_pixel = d_curr_frame[(y_idx + j) * WIDTH + (x_idx + i)];
            int ref_pixel = d_ref_pad_frame[(y_ref + j) * PADWIDTH + (x_ref + i)];
            sad += abs(curr_pixel - ref_pixel);
        }
    }
    return sad;
}

// Discrete Cosine Transform
__device__ void DCT(float block[BLOCK_SIZE][BLOCK_SIZE])
{
    float alpha, beta;
    float sum;
    float temp[BLOCK_SIZE][BLOCK_SIZE];
    int u, v, x, y;

    for (u = 0; u < BLOCK_SIZE; u++)
    {
        for (v = 0; v < BLOCK_SIZE; v++)
        {
            alpha = (u == 0) ? sqrt(1.0 / BLOCK_SIZE) : sqrt(2.0 / BLOCK_SIZE);
            beta = (v == 0) ? sqrt(1.0 / BLOCK_SIZE) : sqrt(2.0 / BLOCK_SIZE);

            sum = 0.0;
            for (x = 0; x < BLOCK_SIZE; x++)
            {
                for (y = 0; y < BLOCK_SIZE; y++)
                {
                    sum += block[x][y] *
                           cos((2 * x + 1) * u * M_PI / (2.0 * BLOCK_SIZE)) *
                           cos((2 * y + 1) * v * M_PI / (2.0 * BLOCK_SIZE));
                }
            }
            temp[u][v] = alpha * beta * sum;
        }
    }

    // Copy results back to the original block
    for (u = 0; u < BLOCK_SIZE; u++)
    {
        for (v = 0; v < BLOCK_SIZE; v++)
        {
            block[u][v] = round(temp[u][v]);
        }
    }
}

// Quantization
__device__ void Quantize(float block[BLOCK_SIZE][BLOCK_SIZE])
{
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            block[i][j] = round(block[i][j] / d_qMatrix[i][j]);
        }
    }
}

// Rescaling (Inverse Quantization)
__device__ void Rescale(float block[BLOCK_SIZE][BLOCK_SIZE])
{
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            block[i][j] = round(block[i][j] * d_qMatrix[i][j]);
        }
    }
}

// Inverse Discrete Cosine Transform
__device__ void IDCT(float block[BLOCK_SIZE][BLOCK_SIZE])
{
    float alpha, beta;
    float sum;
    float temp[BLOCK_SIZE][BLOCK_SIZE];
    int u, v, x, y;

    for (x = 0; x < BLOCK_SIZE; x++)
    {
        for (y = 0; y < BLOCK_SIZE; y++)
        {
            sum = 0.0;
            for (u = 0; u < BLOCK_SIZE; u++)
            {
                for (v = 0; v < BLOCK_SIZE; v++)
                {
                    alpha = (u == 0) ? sqrt(1.0 / BLOCK_SIZE) : sqrt(2.0 / BLOCK_SIZE);
                    beta = (v == 0) ? sqrt(1.0 / BLOCK_SIZE) : sqrt(2.0 / BLOCK_SIZE);

                    sum += alpha * beta * block[u][v] *
                           cos((2 * x + 1) * u * M_PI / (2.0 * BLOCK_SIZE)) *
                           cos((2 * y + 1) * v * M_PI / (2.0 * BLOCK_SIZE));
                }
            }
            temp[x][y] = sum;
        }
    }

    // Copy results back to the original block
    for (x = 0; x < BLOCK_SIZE; x++)
    {
        for (y = 0; y < BLOCK_SIZE; y++)
        {
            block[x][y] = round(temp[x][y]);
        }
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

__device__ void blockManipulation(float block[BLOCK_SIZE][BLOCK_SIZE])
{
    // Apply DCT
    //DCT(block);

    // Apply Quantize
    Quantize(block);

    // Apply Rescale
    Rescale(block);

    // Apply IDCT
    //IDCT(block);
}

#endif