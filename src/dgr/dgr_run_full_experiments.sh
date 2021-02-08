#!/usr/bin/env bash

# ================
# Permutated MNIST
# ================

#PERM_MNIST_GENERATOR_ITERATIONS=3000
#PERM_MNIST_SOLVER_ITERATIONS=1000
PERM_MNIST_GENERATOR_ITERATIONS=5000
PERM_MNIST_SOLVER_ITERATIONS=5000
PERM_MNIST_LR=0.0001
PERM_MNIST_IMPORTANCE_OF_NEW_TASK=0.3

usage () {
  echo "Usage:"
  echo "  -h              Display this message"
  echo "  -s server       Set visualisation server to server"
  echo "  -p port         Set visualisation port to server"
  echo "  -i 1            Creates visualization server if 1, if 0 no visualisation"
  exit 0
}

while getopts ":hs:p:i:" opt; do
  case ${opt} in 
    h ) 
      usage 
      ;;
    s )
      server=${OPTARG}
      ;;
    p )
      port=${OPTARG}
      ;;
    i )
      vis=${OPTARG}
      ((vis == 1 || vis == 0)) || usage
      ;;
  esac
done      
shift $((OPTIND-1))

server=${server:-http://localhost}
port=${port:-8097}
vis=${vis:-1}

echo "You choose to use visualisation: $vis"
if [ $vis == 1 ]; then
  echo "You choose the port: $port"
  echo "On the server: $server"
fi

bazel run //src:main -- --train \
  --experiment=permutated-mnist \
  --replay-mode=exact-replay \
  --solver-iterations=$PERM_MNIST_SOLVER_ITERATIONS \
  --lr=$PERM_MNIST_LR \
  --importance-of-new-task=$PERM_MNIST_IMPORTANCE_OF_NEW_TASK \
  --server $server \
  --port $port \
  --show-vis $vis 

bazel run //src:main -- --train \
  --experiment=permutated-mnist \
  --replay-mode=generative-replay \
  --generator-iterations=$PERM_MNIST_GENERATOR_ITERATIONS \
  --solver-iterations=$PERM_MNIST_SOLVER_ITERATIONS \
  --lr=$PERM_MNIST_LR \
  --importance-of-new-task=$PERM_MNIST_IMPORTANCE_OF_NEW_TASK \
  --server $server \
  --port $port \
  --show-vis $vis 

bazel run //src:main -- --train \
  --experiment=permutated-mnist \
  --replay-mode=none \
  --solver-iterations=$PERM_MNIST_SOLVER_ITERATIONS \
  --lr=$PERM_MNIST_LR \
  --server $server \
  --port $port \
  --show-vis $vis 

# ==========
# MNIST-SVHN
# ==========

MNIST_SVHN_GENERATOR_ITERATIONS=20000
MNIST_SVHN_SOLVER_ITERATIONS=4000
MNIST_SVHN_LR=0.00003
MNIST_SVHN_IMPORTANCE_OF_NEW_TASK=0.4

bazel run //src:main -- --train \
  --experiment=mnist-svhn \
  --replay-mode=exact-replay \
  --solver-iterations=$MNIST_SVHN_SOLVER_ITERATIONS \
  --importance-of-new-task=$MNIST_SVHN_IMPORTANCE_OF_NEW_TASK \
  --lr=$MNIST_SVHN_LR \
  --server $server \
  --port $port \
  --show-vis $vis 

bazel run //src:main -- --train \
  --experiment=mnist-svhn \
  --replay-mode=generative-replay \
  --generator-iterations=$MNIST_SVHN_GENERATOR_ITERATIONS \
  --solver-iterations=$MNIST_SVHN_SOLVER_ITERATIONS \
  --importance-of-new-task=$MNIST_SVHN_IMPORTANCE_OF_NEW_TASK \
  --lr=$MNIST_SVHN_LR \
  --sample-log \
  --server $server \
  --port $port \
  --show-vis $vis 

bazel run //src:main -- --train \
  --experiment=mnist-svhn \
  --replay-mode=none \
  --generator-iterations=$MNIST_SVHN_GENERATOR_ITERATIONS \
  --solver-iterations=$MNIST_SVHN_SOLVER_ITERATIONS \
  --lr=$MNIST_SVHN_LR \
  --sample-log \
  --server $server \
  --port $port \
  --show-vis $vis 

# ==========
# SVHN-MNIST
# ==========

SVHN_MNIST_GENERATOR_ITERATIONS=20000
SVHN_MNIST_SOLVER_ITERATIONS=4000
SVHN_MNIST_LR=0.00003
SVHN_MNIST_IMPORTANCE_OF_NEW_TASK=0.4

bazel run //src:main -- --train \
  --experiment=svhn-mnist \
  --replay-mode=exact-replay \
  --solver-iterations=$SVHN_MNIST_SOLVER_ITERATIONS \
  --importance-of-new-task=$SVHN_MNIST_IMPORTANCE_OF_NEW_TASK \
  --lr=$SVHN_MNIST_LR \
  --server $server \
  --port $port \
  --show-vis $vis 

bazel run //src:main -- --train \
  --experiment=svhn-mnist \
  --replay-mode=generative-replay \
  --generator-iterations=$SVHN_MNIST_GENERATOR_ITERATIONS \
  --solver-iterations=$SVHN_MNIST_SOLVER_ITERATIONS \
  --importance-of-new-task=$SVHN_MNIST_IMPORTANCE_OF_NEW_TASK \
  --lr=$SVHN_MNIST_LR \
  --sample-log \
  --server $server \
  --port $port \
  --show-vis $vis 

bazel run //src:main -- --train \
  --experiment=svhn-mnist \
  --replay-mode=none \
  --generator-iterations=$SVHN_MNIST_GENERATOR_ITERATIONS \
  --solver-iterations=$SVHN_MNIST_SOLVER_ITERATIONS \
  --lr=$SVHN_MNIST_LR \
  --sample-log \
  --server $server \
  --port $port \
  --show-vis $vis 
