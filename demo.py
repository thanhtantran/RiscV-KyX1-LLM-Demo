import numpy as np
import argparse
import os
import time
import onnxruntime as ort
import ky_ort


parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--name",
    required=False,
    default="demo",
    type=str,
    help="Name to the Test Model.",
)
parser.add_argument(
    "-m",
    "--model_path",
    required=True,
    default="resnet18.q.onnx",
    help="Path to the Test Model.",
)
parser.add_argument(
    "-t",
    "--threads",
    required=False,
    default=1,
    type=int,
    help="Set Intra num threads",
)
parser.add_argument(
    "-p",
    "--profile_prefix",
    required=False,
    default=None,
    type=str,
    help="profile_prefix",
)
parser.add_argument(
    "-l",
    "--loop_count",
    required=False,
    default=1,
    type=int,
    help="",
)
parser.add_argument(
    "--log_level",
    required=False,
    default=-1,
    type=int,
    help="",
)

tensor_type_to_np_type = {
    "tensor(float)": "float32",
    "tensor(int8)": "int8",
    "tensor(uint8)": "uint8",
    "tensor(int16)": "int16",
    "tensor(uint16)": "uint16",
    "tensor(int32)": "int32",
    "tensor(uint32)": "uint32",
    "tensor(int64)": "int64",
    "tensor(uint64)": "uint64",
}

if __name__ == "__main__":
    args = parser.parse_args()
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.threads
    sess_options.log_severity_level = args.log_level
    if isinstance(args.profile_prefix, str):
        sess_options.profile_file_prefix = args.profile_prefix + args.name
        sess_options.optimized_model_filepath = (
            args.profile_prefix + args.name + "_opt.onnx"
        )
        sess_options.enable_profiling = True

    start = time.time()
    session = ort.InferenceSession(
        args.model_path,
        sess_options,
        providers=["KyExecutionProvider"],
    )
    print("init time cost: {} ms.".format((time.time() - start) * 1000))

    output_names = [o.name for o in session.get_outputs()]
    feed_dict = {}
    # ort_value_feed_dict = {}
    for in_var in session.get_inputs():
        shape = in_var.shape
        dtype = tensor_type_to_np_type.get(in_var.type)
        if dtype in {"int64", "int32", "uint64", "uint32"}:
            feed_dict[in_var.name] = np.zeros(shape, dtype)
        else:
            feed_dict[in_var.name] = np.ones(shape, dtype)
        # ort_value_feed_dict[in_var.name] = ort.OrtValue.ortvalue_from_numpy(
        #    feed_dict[in_var.name]
        # )

    during_time = []
    start = time.time()
    outputs = session.run(output_names, feed_dict)
    # session.run_with_ort_values(output_names, ort_value_feed_dict)
    during_time.append((time.time() - start) * 1000)

    print("inference time cost: {} ms.".format(during_time[-1]))

    if args.loop_count > 1:
        for i in range(args.loop_count):
            start = time.time()
            outputs = session.run(output_names, feed_dict)
            # session.run_with_ort_values(output_names, ort_value_feed_dict)
            during_time.append((time.time() - start) * 1000)

            if i % 10 == 0:
                print("inference time cost: {} ms.".format(during_time[-1]))

    sorted(during_time)

    during_time = during_time[2:-2]

    if len(during_time) > 0:
        avg_during_time = sum(during_time) / len(during_time)
        print(
            "inference time cost avg: {} ms. min: {} ms. max: {} ms.".format(
                avg_during_time, during_time[0], during_time[-1]
            )
        )
