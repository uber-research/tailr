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
    extra_requirement("tensorboard"),
    extra_requirement("torch"),
    extra_requirement("torchvision"),
    project_requirement("visdom"),
]

py_binary(
    name = "main",
    main = "main.py",
    srcs = glob(["*.py", "dgr/*.py", "generator/*.py", "training/*.py", "utils/*.py"]),
    args = ["--train", "--experiment=fashion_mnist", "--replay-mode=generative-replay", "--solver-iterations=1000", "--batch-size=128", "--lr=0.0001", "--importance-of-new-task=0.5", "--vis-show=0"],
    deps = ML_DEPS,
)

py_layer(name="deps_layer", deps = ML_DEPS)
LAYERS = [":deps_layer"]

py3_image(
    name = "dgr-train",
    main = "main.py",
    srcs = glob(["*.py", "dgr/*.py", "generators/*.py", "training/*.py", "utils/*.py"]), 
    layers = LAYERS,
    base = "@brezel//docker:python3_gpu_gke_base",
)

doe_gke(
    name = "dgr-gke",
    image = {"eu.gcr.io/atcp-testing/dgr_image": ":dgr-train"},
    bucket = "gs://atcp-data/experiments/dgr_result/", 
    matrix = ":dgr.mat",
    output = "/results", 
    nodepool = "pool-gpu",
)