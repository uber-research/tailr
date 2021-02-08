load("@rules_python//python:defs.bzl", "py_binary")
load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_extra_deps//:requirements.bzl", extra_requirement="requirement")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@io_bazel_rules_docker//python:image.bzl", "py_layer")
load("@brezel//rules/doe:gke.bzl", "doe_gke")

ML_DEPS = [
    requirement("numpy"),
    requirement("tqdm"),
    requirement("dash"),
    extra_requirement("torch"),
    extra_requirement("scipy"),
    extra_requirement("tensorboard"),
    extra_requirement("pandas"),
    extra_requirement("tensorboardX"),
    extra_requirement("torchvision"),
]

py_library(
    name = "tailr",
    srcs = glob(["*.py"], exclude=["main.py"]),
    deps = ML_DEPS,
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [":tailr"],
)

py_layer(name="deps_layer", deps = ML_DEPS)
LAYERS = [":deps_layer"]

py3_image(
    name = "tailr-train",
    main = "main.py",
    srcs = glob(["*.py"]),
    deps = ML_DEPS, 
    layers = LAYERS,
    base = "@brezel//docker:python3_gpu_gke_base",
)

doe_gke(
    name = "tailr-gke",
    image = {"eu.gcr.io/atcp-testing/tailr_image": ":tailr-train"},
    bucket = "gs://atcp-data/experiments/tailr_result/",
    output = "/results/",
    matrix = ":tailr.mat",
    nodepool = "pool-gpu",
    visibility = ["//visibility:public"],
)

[py_test(
    name = test_file[:-3],
    srcs = [test_file],
    deps = [":tailr"],
) for test_file in glob(["*_test.py"])]
