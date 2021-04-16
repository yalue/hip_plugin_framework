/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <hip/hip_runtime_api.h>
#include "plugin_hip_utilities.h"

#include "backprop.h"
#include "backprop_state.h"

#define ABS(x) (((x) > 0.0) ? (x) : (-(x)))

/*** The squashing function.  Currently, it's a sigmoid. ***/

float squash(float x) {
  return (1.0 / (1.0 + exp(-x)));
}

int AllocateFloatArray(float **ptr, int n) {
  if (!CheckHIPError(hipHostMalloc(ptr, n * sizeof(float)))) return 0;
  return 1;
}

// Allocates a 2D array of m by n floats. Returns 0 on error, and nonzero
// otherwise. Sets ptr to be a pointer to an array of float pointers.
static int AllocateFloat2D(float ***ptr, int m, int n) {
  int i;
  float **result = NULL;
  float *tmp = NULL;
  if (!CheckHIPError(hipHostMalloc(&result, m * sizeof(float*)))) {
    return 0;
  }
  memset(result, 0, m * sizeof(float*));
  // We'll allocate all of the underlying float arrays in a single large chunk.
  if (!CheckHIPError(hipHostMalloc(&tmp, n * m * sizeof(float)))) {
    free(result);
    return 0;
  }
  // Grab the pointers to subsequent rows--each row contains n floats, so row
  // i starts at position i * n.
  for (i = 0; i < m; i++) {
    result[i] = tmp + i * n;
  }
  *ptr = result;
  return 1;

}

static void bpnn_randomize_weights(PluginState *s, float **w, int m, int n) {
  int i, j;
  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = RandomFloat(s);
    }
  }
}

static void bpnn_randomize_row(PluginState *s, float *w, int m) {
  int i;
  for (i = 0; i <= m; i++) {
    // Note from HIP porting: This is how this function was in the original
    // Rodinia 3.1 release (i.e. not randomized at all). "Determined by a fair
    // dice roll" I guess, lol! If someone changes it back to be actually
    // random, remember to use the RandomFloat function instead.
    //w[i] = (float) rand()/RAND_MAX;
    w[i] = 0.1;
  }
}

static void bpnn_zero_weights(float **w, int m, int n) {
  int i, j;
  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}

// Frees a 2D array allocated by AllocateFloat2D
static void Free2D(float **ptr, int n) {
  if (!ptr) return;
  // Free the big underlying chunk of data.
  hipHostFree(ptr[0]);
  // Now free the array of pointers.
  hipHostFree(ptr);
}

void BPNNFree(BPNN *net) {
  int n1, n2;
  n1 = net->input_n;
  n2 = net->hidden_n;
  hipHostFree(net->input_units);
  hipHostFree(net->hidden_units);
  hipHostFree(net->output_units);
  hipHostFree(net->hidden_delta);
  hipHostFree(net->output_delta);
  hipHostFree(net->target);
  if (net->input_weights) {
    Free2D(net->input_weights, n1);
  }
  if (net->input_prev_weights) {
    Free2D(net->input_prev_weights, n1);
  }
  if (net->hidden_weights) {
    Free2D(net->hidden_weights, n2);
  }
  if (net->hidden_prev_weights) {
    Free2D(net->hidden_prev_weights, n2);
  }
  hipHostFree(net);
}

static BPNN* BPNNInternalCreate(int n_in, int n_hidden, int n_out) {
  BPNN *newnet;
  if (!CheckHIPError(hipHostMalloc(&newnet, sizeof(*newnet)))) return NULL;
  memset(newnet, 0, sizeof(*newnet));

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  if (!AllocateFloatArray(&newnet->input_units, n_in + 1)) goto error;
  if (!AllocateFloatArray(&newnet->hidden_units, n_hidden + 1)) goto error;
  if (!AllocateFloatArray(&newnet->output_units, n_out + 1)) goto error;
  if (!AllocateFloatArray(&newnet->hidden_delta, n_hidden + 1)) goto error;
  if (!AllocateFloatArray(&newnet->output_delta, n_out + 1)) goto error;
  if (!AllocateFloatArray(&newnet->target, n_out + 1)) goto error;
  if (!AllocateFloat2D(&newnet->input_weights, n_in + 1, n_hidden + 1)) {
    goto error;
  }
  if (!AllocateFloat2D(&newnet->hidden_weights, n_hidden + 1, n_out + 1)) {
    goto error;
  }
  if (!AllocateFloat2D(&newnet->input_prev_weights, n_in + 1, n_hidden + 1)) {
    goto error;
  }
  if (!AllocateFloat2D(&newnet->hidden_prev_weights, n_hidden + 1,
    n_out + 1)) {
    goto error;
  }
  return newnet;

error:
  BPNNFree(newnet);
  return NULL;
}

/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/
BPNN* BPNNCreate(int n_in, int n_hidden, int n_out) {
  return BPNNInternalCreate(n_in, n_hidden, n_out);
}

void BPNNInitializeValues(PluginState *s) {
  BPNN *net = s->net;
  int n_in = net->input_n;
  int n_hidden = net->hidden_n;
  int n_out = net->output_n;
  bpnn_randomize_weights(s, net->input_weights, n_in, n_hidden);
  bpnn_randomize_weights(s, net->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(net->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(net->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(s, net->target, n_out);
}

void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2) {
  float sum;
  int j, k;

  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
  /*** For each unit in second layer ***/
  for (j = 1; j <= n2; j++) {
    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k <= n1; k++) {
      sum += conn[k][j] * l1[k];
    }
    l2[j] = squash(sum);
  }
}

void bpnn_output_error(float *delta, float *target, float *output, int nj,
    float *err) {
  int j;
  float o, t, errsum;
  errsum = 0.0;
  for (j = 1; j <= nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += ABS(delta[j]);
  }
  *err = errsum;
}

void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no,
    float **who, float *hidden, float *err) {
  int j, k;
  float h, sum, errsum;
  errsum = 0.0;
  for (j = 1; j <= nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += ABS(delta_h[j]);
  }
  *err = errsum;
}

void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
    float **w, float **oldw) {
  float new_dw;
  int k, j;
  ly[0] = 1.0;
  for (j = 1; j <= ndelta; j++) {
    for (k = 0; k <= nly; k++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;
    }
  }
}

