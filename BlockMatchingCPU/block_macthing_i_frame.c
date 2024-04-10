
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#define BLOCK_SIZE 8 // Assuming block size is 8x8
#define WIDTH 352 // Example video frame WIDTH
#define HEIGHT 288 // Example video frame HEIGHT
#define SEARCH_RANGE 2 // Search range
#define QP 6 // QP
#define IFRAME_INTERVAL 10 


const int RANGE_2[25][2] = {{0, 0}, {-1, 0}, {0, -1}, {0, 1}, {1, 0}, 
    {-2, 0}, {-1, -1}, {-1, 1}, {0, -2}, {0, 2}, {1, -1}, {1, 1}, {2, 0}, {-2, -1}, {-2, 1}, {-1, -2},
    {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}, {-2, -2}, {-2, 2}, {2, -2}, {2, 2}};

// Discrete Cosine Transform
void DCT(float block[BLOCK_SIZE][BLOCK_SIZE]) {
    float alpha, beta;
    float sum;
    float temp[BLOCK_SIZE][BLOCK_SIZE];
    int u, v, x, y;

    for (u = 0; u < BLOCK_SIZE; u++) {
        for (v = 0; v < BLOCK_SIZE; v++) {
            alpha = (u == 0) ? sqrt(1.0 / BLOCK_SIZE) : sqrt(2.0 / BLOCK_SIZE);
            beta = (v == 0) ? sqrt(1.0 / BLOCK_SIZE) : sqrt(2.0 / BLOCK_SIZE);

            sum = 0.0;
            for (x = 0; x < BLOCK_SIZE; x++) {
                for (y = 0; y < BLOCK_SIZE; y++) {
                    sum += block[x][y] *
                           cos((2 * x + 1) * u * M_PI / (2.0 * BLOCK_SIZE)) *
                           cos((2 * y + 1) * v * M_PI / (2.0 * BLOCK_SIZE));
                }
            }
            temp[u][v] = alpha * beta * sum;
        }
    }

    // Copy results back to the original block
    for (u = 0; u < BLOCK_SIZE; u++) {
        for (v = 0; v < BLOCK_SIZE; v++) {
            block[u][v] = round(temp[u][v]);
        }
    }
}

// Quantization
void Quantize(float block[BLOCK_SIZE][BLOCK_SIZE], float qMatrix[BLOCK_SIZE][BLOCK_SIZE]) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            block[i][j] = round(block[i][j] / qMatrix[i][j]);
        }
    }
}

// Rescaling (Inverse Quantization)
void Rescale(float block[BLOCK_SIZE][BLOCK_SIZE], const float qMatrix[BLOCK_SIZE][BLOCK_SIZE]) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            block[i][j] = round(block[i][j] * qMatrix[i][j]);
        }
    }
}

// Inverse Discrete Cosine Transform
void IDCT(float block[BLOCK_SIZE][BLOCK_SIZE]) {
    float alpha, beta;
    float sum;
    float temp[BLOCK_SIZE][BLOCK_SIZE];
    int u, v, x, y;

    for (x = 0; x < BLOCK_SIZE; x++) {
        for (y = 0; y < BLOCK_SIZE; y++) {
            sum = 0.0;
            for (u = 0; u < BLOCK_SIZE; u++) {
                for (v = 0; v < BLOCK_SIZE; v++) {
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
    for (x = 0; x < BLOCK_SIZE; x++) {
        for (y = 0; y < BLOCK_SIZE; y++) {
            block[x][y] = round(temp[x][y]);
        }
    }
}

void initQuantizationMatrix(float qMatrix[BLOCK_SIZE][BLOCK_SIZE]) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            if (i + j < BLOCK_SIZE - 1) {
                qMatrix[i][j] = pow(2, QP);
            }
            else if (i + j == BLOCK_SIZE - 1) {
                qMatrix[i][j] = pow(2, QP + 1);
            }
            else {
                qMatrix[i][j] = pow(2, QP + 2);
            }
        }
    }
}

// GeneratePredictedFrame
void GeneratePredictedFrame(unsigned char *predictedFrame, unsigned char *prevFrame, int *motionVectors) {
    int blocksInRow = WIDTH / BLOCK_SIZE;
    int blocksInCol = HEIGHT / BLOCK_SIZE;

    memset(predictedFrame, 0, WIDTH * HEIGHT);

    for (int row = 0; row < blocksInCol; row++) {
        for (int col = 0; col < blocksInRow; col++) {
            int vectorIndex = (row * blocksInRow + col) * 2;
            int dy = motionVectors[vectorIndex];
            int dx = motionVectors[vectorIndex + 1];

            for (int y = 0; y < BLOCK_SIZE; y++) {
                for (int x = 0; x < BLOCK_SIZE; x++) {
                    int srcX = col * BLOCK_SIZE + x + dx;
                    int srcY = row * BLOCK_SIZE + y + dy;
                    int destX = col * BLOCK_SIZE + x;
                    int destY = row * BLOCK_SIZE + y;

                    // if (srcX >= 0 && srcX < WIDTH && srcY >= 0 && srcY < HEIGHT) {}
                    predictedFrame[destY * WIDTH + destX] = prevFrame[srcY * WIDTH + srcX];
                    
                }
            }
        }
    }
}

void printMatrix(float matrix[BLOCK_SIZE][BLOCK_SIZE]) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            printf("%0.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void blockManipulation(float block[BLOCK_SIZE][BLOCK_SIZE], float qMatrix[BLOCK_SIZE][BLOCK_SIZE]) {
    // Apply DCT
    DCT(block);

    // Apply Quantize
    Quantize(block, qMatrix);

    // Apply Rescale
    Rescale(block, qMatrix);

    // Apply IDCT
    IDCT(block);
}

// GenerateReconstructedFrame
void generateReconstructedFrame(unsigned char *reconstructedFrame, unsigned char *predictedFrame, unsigned char *currFrame, float qMatrix[BLOCK_SIZE][BLOCK_SIZE]) {
    float block[BLOCK_SIZE][BLOCK_SIZE];

    for (int y = 0; y < HEIGHT; y += BLOCK_SIZE) {
        for (int x = 0; x < WIDTH; x += BLOCK_SIZE) {
            
            
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    block[i][j] = (float)(currFrame[(y + i) * WIDTH + (x + j)] - predictedFrame[(y + i) * WIDTH + (x + j)]);

                }
            }

            blockManipulation(block, qMatrix);

            // reconstructedFrame
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    int val = (int)block[i][j] + predictedFrame[(y + i) * WIDTH + (x + j)];
                    reconstructedFrame[(y + i) * WIDTH + (x + j)] = (unsigned char)fmax(0, fmin(255, val));
                }
            }
        }
    }
}


void saveYFrameToText(FILE *file, unsigned char *frame, int frameNumber) {
    if (file == NULL) {
        printf("File pointer is NULL.\n");
        return;
    }
    
    fprintf(file, "Frame %d:\n", frameNumber); 
    
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            fprintf(file, "%d ", frame[y * WIDTH + x]); 
        }
        fprintf(file, "\n"); 
    }
    
    fprintf(file, "\n\n"); 
}

// Calculate Mean Absolute Error (SAD) between two blocks
int calculateFrameSAD(unsigned char *prev_block, unsigned char *curr_block) {
    int sum = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += abs(prev_block[i * WIDTH + j] - curr_block[i * WIDTH + j]);
        }
    }
    return sum;
}

int calculateBlockSAD(unsigned char block[BLOCK_SIZE][BLOCK_SIZE], unsigned char *curr_block) {
    int sum = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += abs(block[i][j] - curr_block[i * WIDTH + j]);
        }
    }
    return sum;
}

// CPU-side block matching algorithm
void pFrameMontionEstimationCPU(unsigned char *prevFrame, unsigned char *currFrame, int *motionVectors) {
    int blocksInRow = WIDTH / BLOCK_SIZE;
    int blocksInCol = HEIGHT / BLOCK_SIZE;

    for (int row = 0; row < blocksInCol; row++) {
        for (int col = 0; col < blocksInRow; col++) {
            int minSAD = INT_MAX;
            int bestMatchX = 0;
            int bestMatchY = 0;

            // Starting position of the current block
            int blockStartX = col * BLOCK_SIZE;
            int blockStartY = row * BLOCK_SIZE;

            // Search in every position within the search range
            for (int i = 0; i < 25; i++) {
                int searchPosX = blockStartX + RANGE_2[i][1];
                int searchPosY = blockStartY + RANGE_2[i][0];

                // Ensure the search position is within the frame
                if (searchPosX < 0 || searchPosY < 0 || searchPosX + BLOCK_SIZE > WIDTH || searchPosY + BLOCK_SIZE > HEIGHT) {
                    continue;
                }

                // Calculate SAD
                int SAD = calculateFrameSAD(&prevFrame[searchPosY * WIDTH + searchPosX], &currFrame[blockStartY * WIDTH + blockStartX]);

                // Update the best match
                if (SAD < minSAD) {
                    minSAD = SAD;
                    bestMatchX = RANGE_2[i][1];
                    bestMatchY = RANGE_2[i][0];
                }
            }

            // Store the motion vector
            motionVectors[(row * blocksInRow + col) * 2] = bestMatchY;
            motionVectors[(row * blocksInRow + col) * 2 + 1] = bestMatchX;
        }
    }
}

void iFrameReconstCPU(unsigned char *currFrame, unsigned char *predictedFrame, unsigned char *reconstructedFrame, int *motionMode, float qMatrix[BLOCK_SIZE][BLOCK_SIZE]) {
    memset(predictedFrame, 0, WIDTH * HEIGHT);
    float block[BLOCK_SIZE][BLOCK_SIZE];
    
    for (int row = 0; row < HEIGHT; row += BLOCK_SIZE) {
        for (int col = 0; col < WIDTH; col += BLOCK_SIZE){
            // pad the 2 blocks
            unsigned char horizontalBlock[BLOCK_SIZE][BLOCK_SIZE];
            unsigned char verticalBlock[BLOCK_SIZE][BLOCK_SIZE];
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    if (col == 0)
                        horizontalBlock[i][j] = 0;
                    else
                        horizontalBlock[i][j] = reconstructedFrame[(row + i) * WIDTH + col - 1];
                    if (row == 0)
                        verticalBlock[i][j] = 0;
                    else
                        verticalBlock[i][j] = reconstructedFrame[(row - 1) * WIDTH + col + j];
                }
            }

            //compare the 2 blocks
            int horizontalSAD = calculateBlockSAD(horizontalBlock, &currFrame[row * WIDTH + col]);
            int verticalSAD = calculateBlockSAD(verticalBlock, &currFrame[row * WIDTH + col]);

            if(horizontalSAD <= verticalSAD) {
                motionMode[row * WIDTH / BLOCK_SIZE / BLOCK_SIZE + col / BLOCK_SIZE] = 0; // horizontal
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        predictedFrame[(row + i) * WIDTH + col + j] = horizontalBlock[i][j];
                        block[i][j] = (float)(currFrame[(row + i) * WIDTH + (col + j)] - horizontalBlock[i][j]);
                        
                    }
                }
            }
            else {
                motionMode[row * WIDTH / BLOCK_SIZE / BLOCK_SIZE + col / BLOCK_SIZE] = 1; // vertical
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        predictedFrame[(row + i) * WIDTH + col + j] = verticalBlock[i][j];
                        block[i][j] = (float)(currFrame[(row + i) * WIDTH + (col + j)] - verticalBlock[i][j]);
                    }
                }
            }

            blockManipulation(block, qMatrix);

            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    int val = (int)block[i][j] + predictedFrame[(row + i) * WIDTH + col + j];
                    reconstructedFrame[(row + i) * WIDTH + (col + j)] = (unsigned char)fmax(0, fmin(255, val));
                }
            }

        }
    }
}

int main() {
    FILE* file = fopen("foreman_cif-1.yuv", "rb");
    if (file == NULL) {
        printf("Error opening file.\n");
        return -1;
    }

    FILE* mvFile = fopen("motionVectors.txt", "w");
    if (mvFile == NULL) {
        printf("Error creating motion vectors file.\n");
        fclose(file);
        return -1;
    }

    FILE *yFrameFile = fopen("AllYFrames.txt", "w"); 
    if (yFrameFile == NULL) {
        printf("Error opening Y frame file.\n");
        return -1;
    }

    FILE *pred_yFrameFile = fopen("AllPredYFrames.txt", "w"); 
    if (pred_yFrameFile == NULL) {
        printf("Error opening Y frame file.\n");
        return -1;
    }

    FILE *recon_yFrameFile = fopen("AllReconYFrames.txt", "w"); 
    if (recon_yFrameFile == NULL) {
        printf("Error opening Y frame file.\n");
        return -1;
    }

    // YUV frame size
    int frameSize = WIDTH * HEIGHT + (WIDTH * HEIGHT) / 2; // YUV 4:2:0
    int yPlaneSize = WIDTH * HEIGHT;
    
    // Allocate memory
    unsigned char* prevFrame = (unsigned char*)malloc(yPlaneSize);
    unsigned char* currFrame = (unsigned char*)malloc(yPlaneSize);
    int* motionVectors = (int*)malloc(sizeof(int) * WIDTH * HEIGHT / (BLOCK_SIZE * BLOCK_SIZE) * 2);
    int* motionMode = (int*)malloc(sizeof(int) * WIDTH * HEIGHT / (BLOCK_SIZE * BLOCK_SIZE));

    //
    unsigned char* predictedFrame = (unsigned char*)malloc(yPlaneSize);
    unsigned char* reconstructedFrame = (unsigned char*)malloc(yPlaneSize);

    float qMatrix[BLOCK_SIZE][BLOCK_SIZE]; 
    initQuantizationMatrix(qMatrix);

    unsigned char* yuvBuffer = (unsigned char*)malloc(frameSize);
    int frameCount = 0;

    // Read and process each frame
    // while (fread(yuvBuffer, 1, frameSize, file) == frameSize) 
    while (frameCount < 15){
        fread(yuvBuffer, 1, frameSize, file);
        frameCount++;

        // Copy Y plane to currFrame
        memcpy(currFrame, yuvBuffer, yPlaneSize);

        if (frameCount % IFRAME_INTERVAL == 1) {
            iFrameReconstCPU(currFrame, predictedFrame, reconstructedFrame, motionMode, qMatrix);

            saveYFrameToText(pred_yFrameFile, predictedFrame, frameCount);
            saveYFrameToText(recon_yFrameFile, reconstructedFrame, frameCount);
        
            // Save motion vectors to file
            fprintf(mvFile, "Frame %d:\n", frameCount);
            for (int i = 0; i < WIDTH * HEIGHT / (BLOCK_SIZE * BLOCK_SIZE); ++i) {
                fprintf(mvFile, "%d\n", motionMode[i]);
            }
            fprintf(mvFile, "\n");

            printf("Frame %d (i frame) complete.\n", frameCount);
            
        }
        else {
            // Call block matching algorithm
            pFrameMontionEstimationCPU(prevFrame, currFrame, motionVectors);

            // Call 
            GeneratePredictedFrame(predictedFrame, prevFrame, motionVectors);
            generateReconstructedFrame(reconstructedFrame, predictedFrame, currFrame, qMatrix);

            // char reconstructedFilename[256];
            // sprintf(reconstructedFilename, "ReconstructedFrame_%d.txt", frameCount);
            saveYFrameToText(pred_yFrameFile, predictedFrame, frameCount);
            saveYFrameToText(recon_yFrameFile, reconstructedFrame, frameCount);
        
            // Save motion vectors to file
            fprintf(mvFile, "Frame %d:\n", frameCount);
            for (int i = 0; i < WIDTH * HEIGHT / (BLOCK_SIZE * BLOCK_SIZE); ++i) {
                fprintf(mvFile, "%d %d\n", motionVectors[i * 2], motionVectors[i * 2 + 1]);
            }
            fprintf(mvFile, "\n");
            printf("Frame %d (p frame) complete.\n", frameCount);
        }
    
        saveYFrameToText(yFrameFile, currFrame, frameCount);

        // Set current frame as the previous frame for the next iteration
        memcpy(prevFrame, reconstructedFrame, yPlaneSize);
        
    }
    

    // Clean up resources
    fclose(file);
    fclose(yFrameFile);
    fclose(pred_yFrameFile);
    fclose(recon_yFrameFile);
    fclose(mvFile);
    free(prevFrame);
    free(currFrame);
    free(reconstructedFrame);
    free(predictedFrame);
    free(motionVectors);
    free(yuvBuffer);

    return 0;
}
