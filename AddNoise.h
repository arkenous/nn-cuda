
#ifndef NN_CUDA_ADDNOISE_H
#define NN_CUDA_ADDNOISE_H

#include <vector>

using std::vector;

vector<vector<double>> add_noise(const vector<vector<double>> &input, const float rate);
vector<vector<double>> add_random_noise(const vector<vector<double>> &input, const float rate);

#endif //NN_CUDA_ADDNOISE_H
