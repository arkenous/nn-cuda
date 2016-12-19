#include <iostream>
#include <vector>
#include <algorithm>
#include "MultiLayerPerceptron.h"
#include "Data.h"
#include "Data_me.h"
#include "Data_other.h"
#include "StackedDenoisingAutoencoder.h"
#include "Normalize.h"
#include "AddNoise.h"

using std::vector;
using std::string;
using std::random_device;
using std::mt19937;
using std::cout;
using std::endl;
using std::shuffle;

int main() {
  double dropout_rate = 0.5;
  unsigned long num_sda_layer = 1;
  float sda_compression_rate = 0.6;
  cout << dropout_rate << " " << num_sda_layer << " " << sda_compression_rate << endl;

  random_device rnd;
  mt19937 mt;
  mt.seed(rnd());

  vector<vector<double>> noised = add_noise(me, 0.3);

  for (unsigned long i = 0, size = me.size(); i < size; ++i)
    normalize(&me[i]);
  for (unsigned long i = 0, size = noised.size(); i < size; ++i)
    normalize(&noised[i]);
  for (unsigned long i = 0, size = other.size(); i < size; ++i)
    normalize(&other[i]);

  // test_successのシャッフル
  shuffle(me.begin(), me.end(), mt);
  shuffle(noised.begin(), noised.end(), mt);
  shuffle(other.begin(), other.end(), mt);

  vector<vector<double>> train;
  train.push_back(me[0]);
  train.push_back(me[1]);
  train.push_back(me[2]);
  train.push_back(other[0]);
  train.push_back(other[1]);
  train.push_back(other[2]);

  StackedDenoisingAutoencoder stackedDenoisingAutoencoder;
  string sda_params = stackedDenoisingAutoencoder.learn(train, num_sda_layer,
                                                        sda_compression_rate, dropout_rate);
  unsigned long mlp_input_size = stackedDenoisingAutoencoder.getNumMiddleNeuron();

  MultiLayerPerceptron mlp = MultiLayerPerceptron(mlp_input_size, mlp_input_size,
                                                  answer[0].size(), 1, 1,
                                                  dropout_rate, sda_params);
  mlp.learn(train, answer);

  cout << "--- NaN check ---" << endl;
  while (isnan(mlp.out(train[0], true)[0])) {
    cout << "is NaN" << endl;
    mlp = MultiLayerPerceptron(mlp_input_size, mlp_input_size, answer[0].size(), 1, 1,
                               dropout_rate, sda_params);
    mlp.learn(train, answer);
  }

  cout << "----------     Success     ----------" << endl;
  for (unsigned long i = 0, size = me.size(); i < size; ++i)
    mlp.out(me[i], true);

  cout << "----------     Fail[other]     ----------" << endl;
  for (unsigned long i = 0, size = other.size(); i < size; ++i)
    mlp.out(other[i], true);

//    cout << "----------     Fail[noised]     ----------" << endl;
//    for (unsigned long i = 0, size = noised.size(); i < size; ++i)
//      mlp.out(noised[i], true);

  return 0;
}

