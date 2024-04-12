Determining workspace structure

Deciding which workspace information to collect

Gathering workspace info

# CUDA Block Matching Project

This project implements a CUDA-based block matching algorithm for motion estimation. The project is structured as follows:

## [`BlockMatchingCPU`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fwad098%2FBlockMatchingOffical%2FBlockMatchingCPU%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/wad098/BlockMatchingOffical/BlockMatchingCPU")

This directory contains the CPU implementation of the block matching algorithm.

- [`block_macthing_i_frame.c`](command:_github.copilot.openSymbolInFile?%5B%22BlockMatchingCPU%2Fblock_macthing_i_frame.c%22%2C%22block_macthing_i_frame.c%22%5D "BlockMatchingCPU/block_macthing_i_frame.c"): This is the main file for the CPU implementation. It includes the main function that reads the input video file, performs block matching, and writes the output to a file.

## [`BlockMatchingGPU`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fwad098%2FBlockMatchingOffical%2FBlockMatchingGPU%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/wad098/BlockMatchingOffical/BlockMatchingGPU")

This directory contains the GPU implementation of the block matching algorithm.

### [`BlockMatchingGPU/iFrame`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fwad098%2FBlockMatchingOffical%2FBlockMatchingGPU%2FiFrame%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/wad098/BlockMatchingOffical/BlockMatchingGPU/iFrame")

This directory contains files related to the iFrame part of the GPU implementation.

- [`README.Md`](command:_github.copilot.openSymbolInFile?%5B%22BlockMatchingGPU%2FiFrame%2FREADME.Md%22%2C%22README.Md%22%5D "BlockMatchingGPU/iFrame/README.Md"): This file provides information about the files in the iFrame directory.

### [`BlockMatchingGPU/pFrame`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fwad098%2FBlockMatchingOffical%2FBlockMatchingGPU%2FpFrame%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/wad098/BlockMatchingOffical/BlockMatchingGPU/pFrame")

This directory contains files related to the pFrame part of the GPU implementation.

- [`util.cu`](command:_github.copilot.openSymbolInFile?%5B%22BlockMatchingGPU%2FpFrame%2Futil.cu%22%2C%22util.cu%22%5D "BlockMatchingGPU/pFrame/util.cu"): This file contains utility functions and constants used in the GPU implementation.

#### [`BlockMatchingGPU/pFrame/bug_but_fast`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fwad098%2FBlockMatchingOffical%2FBlockMatchingGPU%2FpFrame%2Fbug_but_fast%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/wad098/BlockMatchingOffical/BlockMatchingGPU/pFrame/bug_but_fast")

This directory contains an optimized version of the GPU implementation.

- [`full_search_stream_without_i_frame_2.cu`](command:_github.copilot.openSymbolInFile?%5B%22BlockMatchingGPU%2FpFrame%2Fbug_but_fast%2Ffull_search_stream_without_i_frame_2.cu%22%2C%22full_search_stream_without_i_frame_2.cu%22%5D "BlockMatchingGPU/pFrame/bug_but_fast/full_search_stream_without_i_frame_2.cu"): This file contains an optimized version of the block matching algorithm.

#### [`BlockMatchingGPU/pFrame/luck`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fwad098%2FBlockMatchingOffical%2FBlockMatchingGPU%2FpFrame%2Fluck%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/wad098/BlockMatchingOffical/BlockMatchingGPU/pFrame/luck")

This directory contains another version of the GPU implementation.

- [`luck_stream.cu`](command:_github.copilot.openSymbolInFile?%5B%22BlockMatchingGPU%2FpFrame%2Fluck%2Fluck_stream.cu%22%2C%22luck_stream.cu%22%5D "BlockMatchingGPU/pFrame/luck/luck_stream.cu"): This file contains another version of the block matching algorithm.

#### [BlockMatchingGPU/pFrame/naive approach](BlockMatchingGPU/pFrame/naive%20approach)

This directory contains a naive approach to the GPU implementation.

- `full_search_stream_naive.cu`: This file contains a naive approach to the block matching algorithm.

#### [`BlockMatchingGPU/pFrame/thread_padding`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fwad098%2FBlockMatchingOffical%2FBlockMatchingGPU%2FpFrame%2Fthread_padding%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/wad098/BlockMatchingOffical/BlockMatchingGPU/pFrame/thread_padding")

This directory contains files related to the thread padding part of the GPU implementation.

#### [BlockMatchingGPU/pFrame/with padding](BlockMatchingGPU/pFrame/with%20padding)

This directory contains files related to the padding part of the GPU implementation.

- `full_search_stream_pad.cu`: This file contains the block matching algorithm with padding.

## [`README.md`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fwad098%2FBlockMatchingOffical%2FREADME.md%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "/home/wad098/BlockMatchingOffical/README.md")

This is the main README file for the project. It provides an overview of the project, installation instructions, usage instructions, and other information.