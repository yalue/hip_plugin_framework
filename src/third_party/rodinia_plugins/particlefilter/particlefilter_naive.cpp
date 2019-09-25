/**
 * @file ex_particle_OPENMP_seq.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP
 */
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "plugin_hip_utilities.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"

#define PI 3.1415926535897932
#define BLOCK_X 16
#define BLOCK_Y 16

// The following values are taken from Rodinia's original "run" file.
#define DEFAULT_SIZE_X (128)
#define DEFAULT_SIZE_Y (128)
#define DEFAULT_FRAME_COUNT (10)

// I increased this by a factor of 10 from the original benchmark's "run" file.
#define DEFAULT_NPARTICLES (10000)

// The following values were just constants in the original code.
const int DISK_RADIUS = 5;
const int DISK_DIAMETER = DISK_RADIUS * 2 - 1;
const int THREADS_PER_BLOCK = 128;

typedef struct {
  hipStream_t stream;
  int stream_created;
  KernelTimes *kernel_times;
  int block_count;
  // We'll use drand48_r rather than the roll-your-own rng used in the original
  struct drand48_data rng;
  // Parameters and memory used by the original program are below:
  int Nparticles;
  int IszX;
  int IszY;
  int Nfr;
  int *I;
  // Used to an allocation during the original code's dilate_matrix function.
  int *tmp_dilate;
  int *disk;
  // This is set during AllocateMemory, it's just the number of points that lie
  // in a disk. It's needed during allocation.
  int countOnes;
  double *objxy;
  double *weights;
  double *likelihood;
  double *arrayX;
  double *arrayY;
  double *xj;
  double *yj;
  double *CDF;
  double *arrayX_GPU;
  double *arrayY_GPU;
  double *xj_GPU;
  double *yj_GPU;
  double *CDF_GPU;
  int *ind;
  double *u;
  double *u_GPU;

  // Holds block start and end times for the last kernel invocation.
  uint64_t *device_block_times;
} PluginState;

static void Cleanup(void *data) {
  int i;
  PluginState *s = (PluginState *) data;
  free(s->I);
  free(s->tmp_dilate);
  free(s->disk);
  free(s->objxy);
  free(s->ind);
  free(s->weights);
  free(s->likelihood);
  if (s->kernel_times) {
    for (i = 0; i < (s->Nfr - 1); i++) {
      hipHostFree(s->kernel_times[i].block_times);
    }
  }
  free(s->kernel_times);
  hipFree(s->device_block_times);
  hipHostFree(s->arrayX);
  hipHostFree(s->arrayY);
  hipHostFree(s->xj);
  hipHostFree(s->yj);
  hipHostFree(s->CDF);
  hipHostFree(s->u);
  hipFree(s->arrayX_GPU);
  hipFree(s->arrayY_GPU);
  hipFree(s->xj_GPU);
  hipFree(s->yj_GPU);
  hipFree(s->CDF_GPU);
  hipFree(s->u_GPU);
  if (s->stream_created) {
    CheckHIPError(hipStreamDestroy(s->stream));
  }
  hipHostFree(s);
}

/**
* Fills a radius x radius matrix representing the disk
* @param disk The pointer to the disk to be made
* @param radius  The radius of the disk to be made
*/
static void strelDisk(int *disk, int radius) {
  int diameter = radius * 2 - 1;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      double distance = sqrt(pow((double)(x - radius + 1), 2) +
                             pow((double)(y - radius + 1), 2));
      if (distance < radius) disk[x * diameter + y] = 1;
    }
  }
}

// Allocates all memory used by the plugin. Returns 0 on error.
static int AllocateMemory(PluginState *s) {
  int x, y, i;
  uint64_t *tmp = NULL;
  size_t size = s->IszX * s->IszY * s->Nfr * sizeof(int);
  s->I = (int *) malloc(size);
  if (!s->I) {
    printf("Failed allocating matrix.\n");
  }
  memset(s->I, 0, size);
  s->tmp_dilate = (int *) malloc(size);
  if (!s->tmp_dilate) {
    printf("Failed allocating tmp dilate matrix.\n");
    return 0;
  }
  memset(s->tmp_dilate, 0, size);
  s->disk = (int *) malloc(DISK_DIAMETER * DISK_DIAMETER * sizeof(int));
  if (!s->disk) {
    printf("Failed allocating disk buffer.\n");
    return 0;
  }
  memset(s->disk, 0, DISK_DIAMETER * DISK_DIAMETER * sizeof(int));
  // This only needs to be done once in the original code, so we'll do it here
  // rather than in Execute.
  strelDisk(s->disk, DISK_RADIUS);
  for (x = 0; x < DISK_DIAMETER; x++) {
    for (y = 0; y < DISK_DIAMETER; y++) {
      if (s->disk[x * DISK_DIAMETER + y] == 1) s->countOnes++;
    }
  }
  s->objxy = (double *) malloc(s->countOnes * 2 * sizeof(double));
  if (!s->objxy) {
    printf("Failed allocating the objxy array.\n");
    return 0;
  }
  memset(s->objxy, 0, s->countOnes * 2 * sizeof(double));
  s->ind = (int *) malloc(s->countOnes * sizeof(int));
  if (!s->ind) {
    printf("Failed allocating ind array.\n");
    return 0;
  }
  memset(s->ind, 0, s->countOnes * sizeof(int));
  s->kernel_times = (KernelTimes *) malloc((s->Nfr - 1) * sizeof(KernelTimes));
  if (!s->kernel_times) {
    printf("Failed allocating kernel times buffer.\n");
    return 0;
  }
  memset(s->kernel_times, 0, (s->Nfr - 1) * sizeof(KernelTimes));
  for (i = 0; i < (s->Nfr - 1); i++) {
    // Allocate enough space for start and end times of every block.
    tmp = NULL;
    if (!CheckHIPError(hipHostMalloc(&tmp, 2 * s->block_count *
      sizeof(uint64_t)))) {
      return 0;
    }
    s->kernel_times[i].block_times = tmp;
  }
  if (!CheckHIPError(hipMalloc(&s->device_block_times, 2 * s->block_count *
    sizeof(uint64_t)))) {
    return 0;
  }

  size = s->Nparticles * sizeof(double);
  s->weights = (double *) malloc(size);
  if (!s->weights) {
    printf("Failed allocating the weights array.\n");
    return 0;
  }
  memset(s->weights, 0, size);
  s->likelihood = (double *) malloc(size);
  if (!s->likelihood) {
    printf("Failed allocating the likelihood array.\n");
    return 0;
  }
  memset(s->likelihood, 0, size);
  if (!CheckHIPError(hipHostMalloc(&s->arrayX, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&s->arrayY, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&s->xj, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&s->yj, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&s->CDF, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&s->u, size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->arrayX_GPU, size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->arrayY_GPU, size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->xj_GPU, size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->yj_GPU, size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->CDF_GPU, size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->u_GPU, size))) return 0;
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  int random_seed, i;
  PluginState *s = NULL;
  if (!CheckHIPError(hipSetDevice(params->device_id))) {
    return NULL;
  }
  if (!CheckHIPError(hipHostMalloc(&s, sizeof(*s)))) return NULL;
  memset(s, 0, sizeof(*s));
  random_seed = ((int) (CurrentSeconds() * 1e7) & 0x7fffffff);
  srand48_r(random_seed, &(s->rng));

  // All of this stuff needs to be set before AllocateMemory.
  s->Nparticles = DEFAULT_NPARTICLES;
  s->IszX = DEFAULT_SIZE_X;
  s->IszY = DEFAULT_SIZE_Y;
  s->Nfr = DEFAULT_FRAME_COUNT;
  s->block_count = s->Nparticles / THREADS_PER_BLOCK;
  if ((s->Nparticles % THREADS_PER_BLOCK) != 0) s->block_count++;
  if (!AllocateMemory(s)) {
    Cleanup(s);
    return NULL;
  }

  if (!CheckHIPError(CreateHIPStreamWithMask(&(s->stream),
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(s);
    return NULL;
  }
  s->stream_created = 1;
  for (i = 0; i < (s->Nfr - 1); i++) {
    s->kernel_times[i].kernel_name = "ParticleFilter kernel";
    s->kernel_times[i].thread_count = THREADS_PER_BLOCK;
    s->kernel_times[i].block_count = s->block_count;
  }
  return s;
}

static double randu(PluginState *s) {
  double to_return;
  drand48_r(&(s->rng), &to_return);
  return to_return;
}

/**
* Generates a normally distributed random number using the Box-Muller
* transformation
*/
static double randn(PluginState *s) {
  double u = randu(s);
  double v = randu(s);
  double cosine = cos(2 * PI * v);
  double rt = -2 * log(u);
  return sqrt(rt) * cosine;
}

/**
* Takes in a double and returns an integer that approximates to that double
* @return if the mantissa < .5 => return value < input value; else return value
* > input value
*/
static double roundDouble(double value) {
  int newValue = (int)(value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}

/**
* Set values of the 3D array to a newValue if that value is equal to the
* testValue
* @param testValue The value to be replaced
* @param newValue The value to replace testValue with
* @param array3D The image vector
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
*/
static void setIf(int testValue, int newValue, int *array3D, int *dimX,
    int *dimY, int *dimZ) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
      }
    }
  }
}

/**
* Sets values of 3D matrix using randomly generated numbers from a normal
* distribution
* @param array3D The video to be modified
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param seed The seed array
*/
void addNoise(int *array3D, int *dimX, int *dimY, int *dimZ, PluginState *s) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        array3D[x * *dimY * *dimZ + y * *dimZ + z] =
            array3D[x * *dimY * *dimZ + y * *dimZ + z] +
            ((int) (5 * randn(s)));
      }
    }
  }
}

/**
* Dilates the provided video
* @param matrix The video to be dilated
* @param posX The x location of the pixel to be dilated
* @param posY The y location of the pixel to be dilated
* @param poxZ The z location of the pixel to be dilated
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param error The error radius
*/
void dilate_matrix(int *matrix, int posX, int posY, int posZ, int dimX,
                   int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0) startX++;
  int startY = posY - error;
  while (startY < 0) startY++;
  int endX = posX + error;
  while (endX > dimX) endX--;
  int endY = posY + error;
  while (endY > dimY) endY--;
  int x, y;
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      double distance =
          sqrt(pow((double)(x - posX), 2) + pow((double)(y - posY), 2));
      if (distance < error) matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}

/**
* Dilates the target matrix using the radius as a guide
* @param matrix The reference matrix
* @param dimX The x dimension of the video
* @param dimY The y dimension of the video
* @param dimZ The z dimension of the video
* @param error The error radius to be dilated
* @param newMatrix The target matrix
*/
void imdilate_disk(int *matrix, int dimX, int dimY, int dimZ, int error,
                   int *newMatrix) {
  int x, y, z;
  for (z = 0; z < dimZ; z++) {
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}

/**
* The synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
* @param I The video itself
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames of the video
* @param seed The seed array used for number generation
*/
static void videoSequence(PluginState *s) {
  int *I = s->I;
  int IszX = s->IszX;
  int IszY = s->IszY;
  int Nfr = s->Nfr;
  int k;
  int max_size = IszX * IszY * Nfr;
  int *newMatrix = s->tmp_dilate;

  /*get object centers*/
  int x0 = (int) roundDouble(IszY / 2.0);
  int y0 = (int) roundDouble(IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  /*move point*/
  int xk, yk, pos;
  for (k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size) pos = 0;
    I[pos] = 1;
  }

  /*dilate matrix*/
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (k = 0; k < Nfr; k++) {
        I[x * IszY * Nfr + y * Nfr + k] =
            newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }

  /*define background, add noise*/
  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);
  addNoise(I, &IszX, &IszY, &Nfr, s);
}

// Even though this benchmark does a bunch of copying, it does multiple times
// (once per "frame") during execution, so doing the copies in Execute(...)
// fits better.
static int CopyIn(void *data) {
  return 1;
}

__device__ int findIndexSeq(double *CDF, int lengthCDF, double value) {
  int index = -1;
  int x;
  for (x = 0; x < lengthCDF; x++) {
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  }
  if (index == -1) return lengthCDF - 1;
  return index;
}

__device__ int findIndexBin(double *CDF, int beginIndex, int endIndex,
                            double value) {
  if (endIndex < beginIndex) return -1;
  int middleIndex;
  while (endIndex > beginIndex) {
    middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
    if (CDF[middleIndex] >= value) {
      if (middleIndex == 0)
        return middleIndex;
      else if (CDF[middleIndex - 1] < value)
        return middleIndex;
      else if (CDF[middleIndex - 1] == value) {
        while (CDF[middleIndex] == value && middleIndex >= 0) middleIndex--;
        middleIndex++;
        return middleIndex;
      }
    }
    if (CDF[middleIndex] > value)
      endIndex = middleIndex - 1;
    else
      beginIndex = middleIndex + 1;
  }
  return -1;
}

/*****************************
* CUDA Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param3: CDF
* param4: u
* param5: xj
* param6: yj
* param7: Nparticles
*****************************/
__global__ void kernel(double *arrayX, double *arrayY, double *CDF, double *u,
                       double *xj, double *yj, int Nparticles,
                       uint64_t *block_times) {
  uint64_t start_clock = clock64();
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_clock;
  }
  int block_id = blockIdx.x;  // + gridDim.x * blockIdx.y;
  int i = blockDim.x * block_id + threadIdx.x;

  if (i < Nparticles) {
    int index = -1;
    int x;

    for (x = 0; x < Nparticles; x++) {
      if (CDF[x] >= u[i]) {
        index = x;
        break;
      }
    }
    if (index == -1) {
      index = Nparticles - 1;
    }

    xj[i] = arrayX[index];
    yj[i] = arrayY[index];
  }
  block_times[blockIdx.x * 2 + 1] = clock64();
}

/**
* Fills a 2D array describing the offsets of the disk object
* @param se The disk object
* @param numOnes The number of ones in the disk
* @param neighbors The array that will contain the offsets
* @param radius The radius used for dilation
*/
void getneighbors(int *se, int numOnes, double *neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int)(y - center);
        neighbors[neighY * 2 + 1] = (int)(x - center);
        neighY++;
      }
    }
  }
}

/**
* Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 -
* (IK[IND] - 228)^2)/ 100
* @param I The 3D matrix
* @param ind The current ind array
* @param numOnes The length of ind array
* @return A double representing the sum
*/
double calcLikelihoodSum(int *I, int *ind, int numOnes) {
  double likelihoodSum = 0.0;
  int y;
  for (y = 0; y < numOnes; y++)
    likelihoodSum += (pow((double)(I[ind[y]] - 100), 2) -
                      pow((double)(I[ind[y]] - 228), 2)) /
                     50.0;
  return likelihoodSum;
}

/**
* Finds the first element in the CDF that is greater than or equal to the
* provided value and returns that index
* @note This function uses sequential search
* @param CDF The CDF
* @param lengthCDF The length of CDF
* @param value The value to be found
* @return The index of value in the CDF; if value is never found, returns the
* last index
*/
int findIndex(double *CDF, int lengthCDF, double value) {
  int index = -1;
  int x;
  for (x = 0; x < lengthCDF; x++) {
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  }
  if (index == -1) {
    return lengthCDF - 1;
  }
  return index;
}

/**
* The implementation of the particle filter using OpenMP for many frames
* @see http://openmp.org/wp/
* @note This function is designed to work with a video of several frames. In
* addition, it references a provided MATLAB function which takes the video, the
* objxy matrix and the x and y arrays as arguments and returns the likelihoods
*
* NOTE: I modified this to take all of the parameters from the PluginState
* instance. Also, it now returns 0 on failure and nonzero on success.
*
* Old parameters explanation:
* @param I The video to be run
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames
* @param Nparticles The number of particles to be used
*/
static int particleFilter(PluginState *s) {
  int *I = s->I;
  int IszX = s->IszX;
  int IszY = s->IszY;
  int Nfr = s->Nfr;
  int Nparticles = s->Nparticles;
  int max_size = IszX * IszY * Nfr;
  // original particle centroid
  double xe = roundDouble(IszY / 2.0);
  double ye = roundDouble(IszX / 2.0);

  // expected object locations, compared to center
  int x, y;
  int countOnes = s->countOnes;
  int radius = DISK_RADIUS;
  int *disk = s->disk;
  double *objxy = s->objxy;
  getneighbors(disk, countOnes, objxy, radius);

  // initial weights are all equal (1/Nparticles)
  double *weights = s->weights;
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1 / ((double)(Nparticles));
  }

  // initial likelihood to 0.0
  double *likelihood = s->likelihood;
  double *arrayX = s->arrayX;
  double *arrayY = s->arrayY;
  double *xj = s->xj;
  double *yj = s->yj;
  double *CDF = s->CDF;

  // GPU copies of arrays
  double *arrayX_GPU = s->arrayX_GPU;
  double *arrayY_GPU = s->arrayY_GPU;
  double *xj_GPU = s->xj_GPU;
  double *yj_GPU = s->yj_GPU;
  double *CDF_GPU = s->CDF_GPU;

  int *ind = s->ind;
  double *u = s->u;
  double *u_GPU = s->u_GPU;

  for (x = 0; x < Nparticles; x++) {
    arrayX[x] = xe;
    arrayY[x] = ye;
  }

  int k;
  int indX, indY;
  for (k = 1; k < Nfr; k++) {
    // apply motion model
    // draws sample from motion model (random walk). The only prior information
    // is that the object moves 2x as fast as in the y direction

    for (x = 0; x < Nparticles; x++) {
      arrayX[x] = arrayX[x] + 1.0 + 5.0 * randn(s);
      arrayY[x] = arrayY[x] - 2.0 + 2.0 * randn(s);
    }
    // particle filter likelihood
    for (x = 0; x < Nparticles; x++) {
      // compute the likelihood: remember our assumption is that you know
      // foreground and the background image intensity distribution.
      // Notice that we consider here a likelihood ratio, instead of
      // p(z|x). It is possible in this case. why? a hometask for you.
      // calc ind
      for (y = 0; y < countOnes; y++) {
        indX = roundDouble(arrayX[x]) + objxy[y * 2 + 1];
        indY = roundDouble(arrayY[x]) + objxy[y * 2];
        ind[y] = fabs(indX * IszY * Nfr + indY * Nfr + k);
        if (ind[y] >= max_size) ind[y] = 0;
      }
      likelihood[x] = calcLikelihoodSum(I, ind, countOnes);
      likelihood[x] = likelihood[x] / countOnes;
    }

    // update & normalize weights
    // using equation (63) of Arulampalam Tutorial
    for (x = 0; x < Nparticles; x++) {
      weights[x] = weights[x] * exp(likelihood[x]);
    }

    double sumWeights = 0;
    for (x = 0; x < Nparticles; x++) {
      sumWeights += weights[x];
    }

    for (x = 0; x < Nparticles; x++) {
      weights[x] = weights[x] / sumWeights;
    }

    xe = 0;
    ye = 0;
    // estimate the object location by expected values
    for (x = 0; x < Nparticles; x++) {
      xe += arrayX[x] * weights[x];
      ye += arrayY[x] * weights[x];
    }
    // resampling
    CDF[0] = weights[0];
    for (x = 1; x < Nparticles; x++) {
      CDF[x] = weights[x] + CDF[x - 1];
    }

    double u1 = (1 / ((double)(Nparticles))) * randu(s);
    for (x = 0; x < Nparticles; x++) {
      u[x] = u1 + x / ((double)(Nparticles));
    }

    // CUDA memory copying from CPU memory to GPU memory
    size_t copy_size = sizeof(double) * Nparticles;
    if (!CheckHIPError(hipMemcpyAsync(arrayX_GPU, arrayX, copy_size,
      hipMemcpyHostToDevice, s->stream))) {
      return 0;
    }
    if (!CheckHIPError(hipMemcpyAsync(arrayY_GPU, arrayY, copy_size,
      hipMemcpyHostToDevice, s->stream))) {
      return 0;
    }
    if (!CheckHIPError(hipMemcpyAsync(xj_GPU, xj, copy_size,
      hipMemcpyHostToDevice, s->stream))) {
      return 0;
    }
    if (!CheckHIPError(hipMemcpyAsync(yj_GPU, yj, copy_size,
      hipMemcpyHostToDevice, s->stream))) {
      return 0;
    }
    if (!CheckHIPError(hipMemcpyAsync(CDF_GPU, CDF, copy_size,
      hipMemcpyHostToDevice, s->stream))) {
      return 0;
    }
    if (!CheckHIPError(hipMemcpyAsync(u_GPU, u, copy_size,
      hipMemcpyHostToDevice, s->stream))) {
      return 0;
    }
    if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;

    // KERNEL FUNCTION CALL
    s->kernel_times[k - 1].kernel_launch_times[0] = CurrentSeconds();
    hipLaunchKernelGGL((kernel), dim3(s->block_count), dim3(THREADS_PER_BLOCK),
      0, 0, arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, Nparticles,
      s->device_block_times);
    s->kernel_times[k - 1].kernel_launch_times[1] = CurrentSeconds();
    if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
    s->kernel_times[k - 1].kernel_launch_times[2] = CurrentSeconds();

    // CUDA memory copying back from GPU to CPU memory
    if (!CheckHIPError(hipMemcpyAsync(yj, yj_GPU, copy_size,
      hipMemcpyDeviceToHost, s->stream))) {
      return 0;
    }
    if (!CheckHIPError(hipMemcpyAsync(xj, xj_GPU, copy_size,
      hipMemcpyDeviceToHost, s->stream))) {
      return 0;
    }
    if (!CheckHIPError(hipMemcpyAsync(s->kernel_times[k - 1].block_times,
      s->device_block_times, 2 * s->block_count * sizeof(uint64_t),
      hipMemcpyDeviceToHost, s->stream))) {
      return 0;
    }
    if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;

    for (x = 0; x < Nparticles; x++) {
      // reassign arrayX and arrayY
      arrayX[x] = xj[x];
      arrayY[x] = yj[x];
      weights[x] = 1 / ((double)(Nparticles));
    }
  }

  return 1;
}

static int Execute(void *data) {
  PluginState *s = (PluginState *) data;
  videoSequence(s);
  if (!particleFilter(s)) return 0;
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  PluginState *s = (PluginState *) data;
  // All of the "actual" copying took place in Execute(..), basically by
  // necessity, so all this function needs to do is provide the kernel times.
  times->kernel_count = s->Nfr - 1;
  times->kernel_times = s->kernel_times;
  times->resulting_data_size = 0;
  times->resulting_data = NULL;
  return 1;
}

static const char* GetName(void) {
  return "ParticleFilter-Naive (Rodinia)";
}

int RegisterPlugin(PluginFunctions *functions) {
  functions->get_name = GetName;
  functions->cleanup = Cleanup;
  functions->initialize = Initialize;
  functions->copy_in = CopyIn;
  functions->execute = Execute;
  functions->copy_out = CopyOut;
  return 1;
}

