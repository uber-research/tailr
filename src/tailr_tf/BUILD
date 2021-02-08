load("@rules_python//python:defs.bzl", "py_binary")
load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_extra_deps//:requirements.bzl", extra_requirement="requirement")
load("@python3_project_deps//:requirements.bzl", project_requirement="requirement")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@io_bazel_rules_docker//python:image.bzl", "py_layer")
load("@brezel//rules/doe:gke.bzl", "doe_gke")
load("@io_bazel_rules_docker//container:container.bzl", "container_image")
load("@brezel//rules:images.bzl", "extended_container_image")

ML_DEPS = [
    requirement("numpy"),
    requirement("tqdm"),
    requirement("dash"),
    requirement("memory_profiler"),
    requirement("pyyaml"),
    extra_requirement("pandas"),
    extra_requirement("scipy"),
    extra_requirement("scikit-learn"),
    extra_requirement("tensorboard"),
    extra_requirement("torch"),
    extra_requirement("torchvision"),
    project_requirement("absl-py"),
    project_requirement("dm-sonnet"),
    project_requirement("matplotlib"),
    project_requirement("profilehooks"),
    project_requirement("seaborn"),
    project_requirement("tensorflow-datasets"),
    project_requirement("tensorflow-gpu"),
    project_requirement("tensorflow-probability"),
]

py_library(
    name = "load_model_lib",
    srcs = glob(["*.py"], exclude=["load_model_main.py"]),
    deps = ML_DEPS,
    visibility = ["//visibility:public"],
)

py_library(
    name = "perfect_clf",
    srcs = glob(["oracle_classifier/*.py"], exclude=["oracle_classifier/main.py"]),
    deps = ML_DEPS,
    visibility = ["//visibility:public"],
)

py_binary(
    name = "load_model",
    main = "load_model_main.py",
    data = [
            "oracle_classifier/oracle_classifier.pth"
        ],
    srcs = ["load_model_main.py"],
    deps = [":load_model_lib",  "//src/utils"],
)

py_binary(
    name = "train_clf",
    main = "oracle_classifier/main.py",
    srcs = ["oracle_classifier/main.py"],
    deps = [":perfect_clf"],
)

py_library(
    name = "tailr2",
    srcs = glob(["*.py"], exclude=["train_main.py"]),
    deps = ML_DEPS,
    visibility = ["//visibility:public"],
)

py_binary(
    name = "short_test",
    main = "train_main.py",
    srcs = ["train_main.py"],
    deps = [":tailr2", "//src/utils"],
    data = [
            "oracle_classifier/oracle_classifier.pth"
        ],
    args = ["--dataset", "mnist", "--log_dir", "./", "--clfmode", "cluster_init",
            "--n_steps", "150", "--max_gen_batches", "100", "--cluster_wait_steps", "10", "--batch_mix", "semi_combined",
            "--experiment_name", "short_test"],
)

py_binary(
    name = "short_test_order",
    main = "train_main.py",
    srcs = ["train_main.py"],
    deps = [":tailr2", "//src/utils"],
    data = [
            "oracle_classifier/oracle_classifier.pth"
        ],
    args = ["--dataset", "mnist", "--log_dir", "./", "--clfmode", "cluster_init",
            "--n_steps", "150", "--max_gen_batches", "100", "--cluster_wait_steps", "10", "--class_order", '''"[1,4,5,7,2,6,3,0,9,8]"'''],
)

py_binary(
    name = "short_test_experimental",
    main = "train_main.py",
    srcs = ["train_main.py"],
    deps = [":tailr2", "//src/utils"],
    data = [
            "oracle_classifier/oracle_classifier.pth"
        ],
    args = ["--dataset", "mnist", "--log_dir", "./", "--clfmode", "loss_init",
            "--n_steps", "150", "--max_gen_batches", "100", "--clf_thresh", "1.7",
            "--encoder_type", "mixed" ,"--class_conditioned"],
)

py_binary(
    name = "main_test",
    main = "train_main.py",
    srcs = ["train_main.py"],
    deps = [":tailr2", "//src/utils"],
    data = [
            "config_training.yaml",
            "oracle_classifier/oracle_classifier.pth"
        ],
    args = ["--dataset", "mnist", "--log_dir", "./", "--clfmode", "cluster_init",
            "--n_steps", "500", "--max_gen_batches", "100"],
)

py_layer(name="deps_layer", deps = ML_DEPS)
LAYERS = [":deps_layer"]

py3_image(
    name = "tailr2-train",
    main = "train_main.py",
    srcs = glob(["*.py"]),
    data = [
            "config_training.yaml",
            "oracle_classifier/oracle_classifier.pth"
        ],
    layers = LAYERS,
    deps = ["//src/utils"],
    base = ":python3_cuda10",
)

py3_image(
    name = "logcuda",
    main = "logcuda.py",
    srcs = ["logcuda.py"],
    layers = LAYERS,
    base = ":python3_cuda10",
)

doe_gke(
    name = "logcuda-gke",
    image = {"eu.gcr.io/atcp-testing/tailr2/debug:cuda10.0": ":logcuda"},
    gcs_upload = {"/results": "gs://atcp-data/experiments/tailr_result/"},
    nodepool = "pool-gpu",
    jobs = {"tailr2-logcuda": "a"},
    visibility = ["//visibility:public"],
)

doe_gke(
    name = "tailr2-gke",
    image = {"eu.gcr.io/atcp-testing/tailr2/train:cuda10.0": ":tailr2-train"},
    gcs_upload = {"/results": "gs://atcp-data/experiments/tailr_result/"},
    nodepool = "pool-gpu-highmem",
    jobs = {
        "tailr2-exp-300-0.2": "--dataset mnist --log_dir ./ --clfmode loss_init --clf_thresh 0.2 --n_steps 25000 --max_gen_batches 2000 --encoder_type mixed --batch_mix semi_combined --cluster_wait_steps 300 --experiment_name exp-wait-300-0.2",
        "tailr2-exp-300-0.5": "--dataset mnist --log_dir ./ --clfmode loss_init --clf_thresh 0.5 --n_steps 25000 --max_gen_batches 2000 --encoder_type mixed --batch_mix semi_combined --cluster_wait_steps 300 --experiment_name exp-wait-300-0.5",
        "tailr2-exp-300-0.8": "--dataset mnist --log_dir ./ --clfmode loss_init --clf_thresh 0.8 --n_steps 25000 --max_gen_batches 2000 --encoder_type mixed --batch_mix semi_combined --cluster_wait_steps 300 --experiment_name exp-wait-300-0.8",
        "tailr2-exp-300-1.1": "--dataset mnist --log_dir ./ --clfmode loss_init --clf_thresh 1.1 --n_steps 25000 --max_gen_batches 2000 --encoder_type mixed --batch_mix semi_combined --cluster_wait_steps 300 --experiment_name exp-wait-300-1.1",
        "tailr2-exp-300-1.4": "--dataset mnist --log_dir ./ --clfmode loss_init --clf_thresh 1.4 --n_steps 25000 --max_gen_batches 2000 --encoder_type mixed --batch_mix semi_combined --cluster_wait_steps 300 --experiment_name exp-wait-300-1.4",
        "tailr2-exp-300-2.1": "--dataset mnist --log_dir ./ --clfmode loss_init --clf_thresh 2.1 --n_steps 25000 --max_gen_batches 2000 --encoder_type mixed --batch_mix semi_combined --cluster_wait_steps 300 --experiment_name exp-wait-300-2.1",
    },
    visibility = ["//visibility:public"],
)

extended_container_image(
    name = "python3_cuda10",
    base = "@cuda10//image",
    packages = ["python", "python3", "python3-pip"],
    env = {
        "LD_LIBRARY_PATH":"/usr/local/cuda/lib64:/usr/local/nvidia/lib64",
        "PATH": "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    },
    visibility = ["//visibility:public"]
)
