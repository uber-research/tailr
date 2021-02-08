load("@rules_python//python:defs.bzl", "py_binary")
load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_extra_deps//:requirements.bzl", extra_requirement="requirement")
load("@python3_project_deps//:requirements.bzl", project_requirement="requirement")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@io_bazel_rules_docker//python:image.bzl", "py_layer")
load("@brezel//rules/doe:gke.bzl", "doe_gke")

ML_DEPS = [
    requirement("numpy"),
    requirement("tqdm"),
    requirement("dash"),
    extra_requirement("pandas"),
    extra_requirement("scipy"),
    extra_requirement("scikit-learn"),
    extra_requirement("tensorboard"),
    extra_requirement("torch"),
    project_requirement("absl-py"),
    project_requirement("dm-sonnet"),
    project_requirement("matplotlib"), 
    project_requirement("seaborn"),
    project_requirement("tensorflow"),
    project_requirement("tensorflow-datasets"),
    project_requirement("tensorflow-gpu"),
    project_requirement("tensorflow-probability"),
]

py_library(
    name = "curl",
    srcs = glob(["*.py"], exclude=["train_main.py"]),
    deps = ML_DEPS,
    visibility = ["//visibility:public"],
)

py_binary(
    name = "train_main",
    srcs = ["train_main.py"],
    data = [
        "oracle_classifier.pth"
    ],
    deps = [":curl", "//src/utils"],
    args = ["--dataset", "mnist", "--log_dir", "./"],
)

py_layer(name="deps_layer", deps = ML_DEPS)
LAYERS = [":deps_layer"]

py3_image(
    name = "curl-train",
    main = "train_main.py",
    srcs = glob(["*.py"]),
    data = [
        "oracle_classifier.pth"
    ],
    layers = LAYERS,
    deps = ["//src/utils"],
    base = "@brezel//docker:python3_gpu_gke_base",
)

doe_gke(
    name = "curl-gke",
    image = {"eu.gcr.io/atcp-testing/curl_image": ":curl-train"},
    gcs_upload = {"/results/": "gs://atcp-data/experiments/curl_result/"},
    matrix = ":curl.mat",
    nodepool = "pool-gpu-highmem",
    visibility = ["//visibility:public"],
)

