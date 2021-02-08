load("@rules_python//python:defs.bzl", "py_binary")
load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_extra_deps//:requirements.bzl", extra_requirement="requirement")

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
            extra_requirement("tensorboard"),
            requirement("click"),
           ],
)
