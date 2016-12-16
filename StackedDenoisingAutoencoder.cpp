
#include <random>
#include <sstream>
#include <iostream>
#include "StackedDenoisingAutoencoder.h"
#include "DenoisingAutoencoder.h"
#include "AddNoise.h"

using std::string;
using std::vector;
using std::stringstream;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;

StackedDenoisingAutoencoder::StackedDenoisingAutoencoder() {}

string StackedDenoisingAutoencoder::learn(const vector<vector<double>> &input,
                                          const unsigned long result_num_layer,
                                          const float compression_rate,
                                          const double dropout_rate) {
  unsigned long num_sda_layer = 0;
  stringstream ss;

  vector<vector<double>> answer(input);
  vector<vector<double>> noisy_input(add_noise(input, 0.1));

  DenoisingAutoencoder denoisingAutoencoder(noisy_input[0].size(), compression_rate, dropout_rate);
  ss << denoisingAutoencoder.learn(answer, noisy_input) << "$";

  num_sda_layer++;

  while (num_sda_layer < result_num_layer) {
    answer = vector<vector<double>>(noisy_input);
    noisy_input = add_noise(denoisingAutoencoder.getMiddleOutput(noisy_input), 0.1);

    denoisingAutoencoder = DenoisingAutoencoder(noisy_input[0].size(), compression_rate,
                                                dropout_rate);
    ss << denoisingAutoencoder.learn(answer, noisy_input) << "$";

    num_sda_layer++;
  }

  string result = ss.str();
  result.pop_back();
  ss.str("");
  ss.clear(stringstream::goodbit);

  num_middle_neurons = denoisingAutoencoder.getCurrentMiddleNeuronNum();

  return result;
}

unsigned long StackedDenoisingAutoencoder::getNumMiddleNeuron() {
  return num_middle_neurons;
}