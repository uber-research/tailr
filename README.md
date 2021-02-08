# TAILR

## Run and enter the development container
You need Docker and make installed on your machine.
From the project root run the following:
```
make build
make run-k8s-secrets
```

## Run TAILR on local
Inside the container you can run 
```
bazel run //src/tailr:train_main
```

## Run all experiments of TAILR on GCloud
Inside the container you can run 
```
bazel run //src/tailr:tailr2-gke
```
This will run all experiments described in the `jobs` variable of the `doe_gke` in the `BUILD` file. 
You can pick the experiments you want to run by changing the values of the `jobs` variable. 

## Visualize TB from inside a docker container
Enter the container with the command
```
make run-p<local_port>_<container_port>
```
This will create a link between the chosen local port and container port.

From the container now run
```
bazel run //src/tb_vis:main <log_dir>
```
to visualize the TensorBoard logs present in `<log_dir>`, where `<log_dir>` is an absolute path.
