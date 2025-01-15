#include "HAL.hpp"
#include "Loss.hpp"
#include "nlohmann/json.hpp"
#include "omp.h"
#include <NumCpp.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unordered_map>

using json = nlohmann::json;
typedef void *(*func_ptr)(void);

static std::unordered_map<std::string, func_ptr> loss_map;
static void register_loss(const std::string &loss_name, func_ptr func) {
  loss_map[loss_name] = func;
}

#define REGISTER(LOSS_NAME, CLASS_NAME)                                        \
  register_loss(LOSS_NAME, []() { return (void *)(new CLASS_NAME()); })

int main(int argc, char **argv) {

  /****** Register Losses ******/
  #include "LossRegister.hpp"
  /*****************************/

  if (argc != 2) {
    std::cerr << "Usage: ./cuHAL [FILENAME.json]" << std::endl;
    return 0;
  }

  std::ifstream f(argv[1]);
  json config = json::parse(f);

  /************************* Configurations ********************************/

  auto num_features = config["num_features"];
  auto train_size = config["train_size"];
  auto val_size = config["val_size"];
  auto loss_name = config["loss"];

  auto path_X_train = config["path_X_train"];
  auto path_y_train = config["path_y_train"];
  auto path_X_val = config["path_X_val"];
  auto path_y_val = config["path_y_val"];

  auto max_order = config["max_order"];
  auto sample_ratio = config["sample_ratio"];
  auto reduce_epsilon = config["reduce_epsilon"];

  auto step_size = config["step_size"];
  auto max_iter = config["max_iter"];

  /**************************** Prepare Data *******************************/

  auto label_dim = 1;
  if (config.contains("label_dim")) {
    label_dim = config["label_dim"];
  }

  std::cout << "Reading Data..." << std::endl;

  auto X_train =
      nc::fromfile<float>(path_X_train, ',').reshape(train_size, num_features);
  auto y_train =
      nc::fromfile<float>(path_y_train, ',').reshape(train_size, label_dim);

  auto X_val =
      nc::fromfile<float>(path_X_val, ',').reshape(val_size, num_features);
  auto y_val =
      nc::fromfile<float>(path_y_val, ',').reshape(val_size, label_dim);

  auto weight_train = nc::empty<float>(0, 0);
  auto weight_val = nc::empty<float>(0, 0);
  if (config.contains("path_weight_train")) {
    weight_train = nc::fromfile<float>(config["path_weight_train"], ',')
                       .reshape(train_size, 1);
  }
  if (config.contains("path_weight_val")) {
    weight_val = nc::fromfile<float>(config["path_weight_val"], ',')
                     .reshape(val_size, 1);
  }

  /**************************** Training ******************************/

  auto hal = HAL(X_train, y_train, max_order, sample_ratio, reduce_epsilon);

  auto loss = (Loss *)loss_map[loss_name]();

  auto trainer = SRTrainer(hal, step_size, *loss, weight_train, max_iter);

  auto start = omp_get_wtime();
  trainer.run(X_val, y_val, weight_val);
  auto end = omp_get_wtime();
  std::cout << "Time: " << end - start << "(s)" << std::endl;

  return 0;
}
