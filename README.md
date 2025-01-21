<div align=center>
<img src="https://github.com/ChengYuHan0406/cuHAL/blob/main/logo.png" width="200" height="200">
</div>

# What is cuHAL
Roughly Speaking, **cuHAL** is a CUDA-accelerated implementation of the nonparametric regression method known as [*Highly Adaptive Lasso (HAL)*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5662030/).

But what exactly is HAL? And why do we need yet another nonparametric method (or machine learning algorithm, if you prefer) 
when there’s already an abundance of options like Random Forest, XGBoost, LightGBM, and more?

## HAL and its Advantages
HAL is primarily used in [Targeted Learning](https://onlinelibrary.wiley.com/doi/full/10.1155/2014/502678)—a fascinating branch of causal machine learning—to estimate nuisance parameters (a regression task) of causal estimands, 
such as the average treatment effect, under realistic assumptions. To achieve this, HAL offers the following advantages:
- **Assumption Lean:** HAL assumes only that the underlying regression function is right-hand continuous with left-hand limits and has a finite variation norm.
  **It's hard to imagine a non-pathological scenario where these conditions would be violated**. 
- **Theoretically Guaranteed Fast Convergence Rate:** HAL guarantees convergence to the true regression function at a rate of at least $o_p(n^{-\frac{1}{4}})$,
  which is **dimension-free**—a significant achievement in nonparametric regression.
  
While this convergence result is asymptotic, HAL has demonstrated strong finite-sample performance
and has been shown to be [competitive with state-of-the-art machine learning algorithms across various datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC5662030/).

**Even if you're only concerned with prediction tasks rather than causal inference**, HAL provides distinct advantages:
- **Interpretability:**  The HAL estimator is simply a sparse linear combination of products of indicator functions, making it more interpretable compared
   to ensemble methods like XGBoost. Additionally, HAL can be [converted into an equivalent decision tree](https://par.nsf.gov/servlets/purl/10455952) (though this feature is not implemented in this project),
   further enhancing its interpretability.


## Downside of HAL
While HAL enjoys many great properties, these advantages do not come without a cost. Without some form of approximation,
training HAL can quickly become computationally intractable as the data size increases. This is because HAL generates a design matrix of size
$n \times n(2^{d} - 1)$ as an intermediate step, where $n$ is the sample size and $d$ is the number of features.

To address this issue, the R package [hal9001](https://joss.theoj.org/papers/10.21105/joss.02526.pdf) ([GitHub link](https://github.com/tlverse/hal9001))
allows users to customize the number of knot points and the maximum order of interactions between variables. 
By doing so, users can reduce the number of basis functions to better suit their computational and practical needs.

## Some Design Considerations of cuHAL
Although *hal9001* is already an excellent package, it would be beneficial to harness the power of GPUs to further boost performance,
especially when working with datasets at the scale of those commonly found on Kaggle.

### DesignMatrix
While *cuHAL* aims to mimic the behavior of *hal9001* (e.g., strategies for reducing basis functions), it introduces its own design choices to ensure high GPU utilization.
For instance, during the initialization of the `DesignMatrix` object, *cuHAL* allocates memory on the GPU to store the dataframe and all information needed to construct the
design matrix. This approach minimizes the need to frequently transfer data between the host and the device.

Additionally, instead of precomputing the design matrix, *cuHAL* employs a custom CUDA kernel that fuses the construction of the design matrix with matrix-vector multiplication,
ensuring that the design matrix is **never explicitly constructed**. This design is motivated by two key considerations:

- **Minimized Memory Usage:** Given the sheer size of the design matrix, explicitly generating it for large datasets is infeasible.
- **Reduced Memory Access Overhead:** Even if the design matrix were precomputed, performing matrix operations on it would
  incur significant memory access overhead, which could dominate computational time.

### Optimizer
For optimization, *cuHAL* introduces `SRTrainer`, which implements the [strong rule](https://www.jstatsoft.org/article/view/v106i01)
used in [glmnet](https://github.com/cran/glmnet)
but incorporates an Adam-like update rule. The design of this optimizer is,
admittedly, based on trial and error and still suffers from slow convergence on some large-scale datasets. There is potential for exploring better alternatives in future iterations.

### Custom Loss Function Support
Another notable feature of *cuHAL* is its support for user-defined loss functions with minimal effort. To add a custom loss function, users need only implement it in `Loss.hpp`
and register it in `LossRegister.hpp`. Once registered, the custom loss can be specified in the configuration.

# Install Dependencies & Build cuHAL
Install [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/), the CUDA compiler driver required to compile CUDA programs:
```
sudo apt install -y nvidia-cuda-toolkit
```
Install [NumCpp](https://dpilger26.github.io/NumCpp/doxygen/html/index.html), a C++ library that provides functionality similar to NumPy:
```
sudo apt-get install libboost-all-dev
git clone https://github.com/dpilger26/NumCpp.git
cd NumCpp
mkdir build
cd build
cmake ..
sudo cmake --build . --target install
```
Install [xmake](https://xmake.io/#/getting_started), a lightweight build system used to compile *cuHAL*:
```
curl -fsSL https://xmake.io/shget.text | bash
```
With all dependencies installed, clone the *cuHAL* repository and build it
```
git clone git@github.com:ChengYuHan0406/cuHAL.git
cd cuHAL
xmake
```
# Examples
The `examples` directory contains examples demonstrating three common loss types: `mse`, `wmae`, and `coxloss`.
- **Set Library Path:** Before running examples, ensure the library path is set correctly, as the executable `build/cuHAL` relies on `build/lib`:
  ```
  source set_lib_path.sh
  ```
- **Training:** To train a model, `cuHAL` uses a JSON file to specify configurations such as data paths, losses, hyperparameters, and more:
  ```
  cd build
  ./cuHAL ../examples/mse/config.json
  ```
  ![image](https://github.com/ChengYuHan0406/cuHAL/blob/main/train_hal.gif)
  
- **Configuration:** Below is an example configuration file (`examples/mse/config.json`)
  ```json
  {
    "num_features": 10, 
    "train_size": 1000,
    "val_size": 500,
    "loss": "mse",
    "path_X_train": "../examples/mse/X_train.csv",
    "path_y_train": "../examples/mse/y_train.csv",
    "path_X_val": "../examples/mse/X_val.csv",
    "path_y_val": "../examples/mse/y_val.csv",
    "max_order": 2,
    "sample_ratio": 0.5,
    "reduce_epsilon": 0.1,
    "step_size": 0.001,
    "max_iter": 500
  }
  ```
  | Param | Meaning |
  | :----: | :----: |
  | `max_order` | Upper limit on the order of interactions between variables. |
  | `sample ratio` | Ratio of samples used as knot points. For example, if `train_size=1000` and `sample_ratio=0.5`, then 500 samples will be randomly selected as knot points. |
  | `reduce_epsilon` | Columns of the `DesignMatrix` are filtered based on the proportion of ones. Columns with proportions below `(1 + reduce_epsilon) × min_prop_ones`, or all 0/1 columns, are removed. |
  | `step_size` | Step size for the `SRTrainer` |
  | `max_iter` | Maximum number of iterations for each `lambda` (regularization rate) |

- **Inference:** To make predictions using a trained model, run the `inference.py` script:
  ```
  cd scripts
  python inference.py ../build/best_model.json ../examples/mse/X_test.csv ./y_hat.csv
  ```
- **Feature Importance:** To compute and display feature importance, use the `feature_importance.py` script:
  ```
  cd scripts
  python feature_importance.py ../build/best_model.json ../examples/mse/col_names.csv 20
  ```
# License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Third-Party Licenses
This project uses the following third-party libraries:
- [nlohmann/json](https://github.com/nlohmann/json), licensed under the MIT License.
- [dpilger26/NumCpp](https://github.com/dpilger26/NumCpp), , licensed under the MIT License.
- [google/googletest](https://github.com/google/googletest), licensed under the BSD 3-Clause License.
