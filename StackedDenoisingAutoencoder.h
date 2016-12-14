
#ifndef NN_CUDA_STACKEDDENOISINGAUTOENCODER_H
#define NN_CUDA_STACKEDDENOISINGAUTOENCODER_H

#include <vector>
#include <string>

using std::string;
using std::vector;

class StackedDenoisingAutoencoder {
public:
  StackedDenoisingAutoencoder();

  string learn(const vector<vector<double>> &input, const unsigned long result_num_dimen,
                    const float compression_rate);
  unsigned long getNumMiddleNeuron();

private:
  unsigned long num_middle_neurons;
};

#endif //NN_CUDA_STACKEDDENOISINGAUTOENCODER_H
