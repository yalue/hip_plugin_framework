# This script tests performance under various compute unit mask settings.
import argparse
import json
import subprocess

def generate_config(cu_mask, stripe_width):
    """ Returns a JSON string containing a config. The config will use the
    Matrix Multiply plugin with a 1024x1024 matrix, using 32x32 thread blocks.
    Only the compute unit mask is varied.
    """
    active_cu_count = 0
    for b in cu_mask:
        if b:
            active_cu_count += 1
    hex_mask = cu_mask_to_hex_string(cu_mask)
    plugin_config = {
        "label": str(active_cu_count),
        "log_name": "results/cu_mask_sw%d_%s.json" % (stripe_width, hex_mask),
        "filename": "./bin/matrix_multiply.so",
        "thread_count": [32, 32],
        "block_count": 1,
        "compute_unit_mask": cu_mask,
        "additional_info": {
            "matrix_width": 1024,
            "skip_copy": True
        }
    }
    name = "Compute Unit Count vs. Performance"
    # Indicate the stripe width if it isn't the default
    if stripe_width != 1:
        name += " (stripe width: %d)" % (stripe_width,)
    overall_config = {
        "name": name,
        "max_iterations": 100,
        "max_time": 0,
        "gpu_device_id": 0,
        "pin_cpus": True,
        "do_warmup": True,
        "omit_block_times": True,
        "plugins": [plugin_config]
    }
    return json.dumps(overall_config)

def cu_mask_to_int(cu_mask):
    """ A utility function that takes an array of booleans and returns an
    integer with 1s wherever there was a "True" in the array. The value at
    index 0 is the least significant bit. """
    n = 0
    for b in reversed(cu_mask):
        n = n << 1
        if b:
            n |= 1
    return n

def cu_mask_to_binary_string(cu_mask):
    """ A utility function that takes an array of booleans and returns a string
    of 0s and 1s. """
    return format(cu_mask_to_int(cu_mask), "064b")

def cu_mask_to_hex_string(cu_mask):
    return format(cu_mask_to_int(cu_mask), "08x")

def run_process(cu_mask, stripe_width):
    """ This function starts a process that will run the plugin with the given
    compute unit mask. Also requires a stripe width to include to use in the
    labeling of output files. """
    config = generate_config(cu_mask, stripe_width)
    print("Starting test with CU mask " + cu_mask_to_binary_string(cu_mask))
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

def get_cu_mask(active_cus, total_cus, stripe_width):
    """ Returns a CU mask (represented as a boolean array) with the active
    number of CUs specified by active_cus, using the given stripe_width. """
    to_return = [False] * total_cus
    i = 0
    for n in range(active_cus):
        if i >= total_cus:
            i = n / (total_cus / stripe_width)
        to_return[i] = True
        i += stripe_width
    return to_return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cu_count", type=int, default=60,
        help="The total number of CUs on the GPU.")
    parser.add_argument("--start_count", type=int, default=0,
        help="The number of CUs to start testing from. Can be used to resume "+
            "tests if one hung.")
    parser.add_argument("--stripe_width", type=int, default=1,
        help="The width, in CUs, of the \"stripe\" used to assign subsequent "+
            "CUs. Must evenly divide the cu_count.")
    args = parser.parse_args()
    cu_count = args.cu_count
    if (cu_count <= 0):
        print("The CU count must be positive.")
    stripe_width = args.stripe_width
    if (stripe_width <= 0) or (stripe_width >= cu_count):
        print("The stripe width must be at least 1 and less than the CU count")
        exit(1)
    if (cu_count % stripe_width) != 0:
        print("Invalid stripe width: it must evenly divide the CU count.")
        exit(1)
    for active_cus in range(args.start_count, cu_count):
        print("Running test for %d (+ 1) active CUs." % (active_cus))
        cu_mask = get_cu_mask(active_cus + 1, cu_count, stripe_width)
        run_process(cu_mask, stripe_width)
        print("\n")

