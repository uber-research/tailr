#!/bin/bash

print_bazel_info () {
    echo '=== BAZEL INFO BEGIN ==='
    set -x
    apt list bazel -a
    : '---'
    bazel version
    : '---'
    bazel info
    set +x
    echo '=== BAZEL INFO END ==='
}

print_bazel_info
#bazel test '//...:all'
