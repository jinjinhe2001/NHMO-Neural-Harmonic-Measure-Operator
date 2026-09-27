# Paths used by the scripts in this directory. Source it after editing:
#   source scripts/env.sh
export MCB_ROOT=${MCB_ROOT:-$PWD/data/MCB_benchmark}                    # splits/ + ngf_solutions/
export MNIST_ROOT=${MNIST_ROOT:-$PWD/data/MNIST}                        # raw/{train,t10k}-*-idx*-ubyte.gz
export NHMO_DATA_2D=${NHMO_DATA_2D:-$PWD/data/mnist_pde_2d_paramBC_lf}  # train/ test/ test_ood/
export NHMO_CKPT_DIR=${NHMO_CKPT_DIR:-$PWD/checkpoints}                 # 3d/ 2d/
export RESULTS_DIR=${RESULTS_DIR:-$PWD/results}
mkdir -p "$RESULTS_DIR"
