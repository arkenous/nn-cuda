
#ifndef NN_CUDA_NEURON_H
#define NN_CUDA_NEURON_H


#include <vector>
#include <random>

#include <thrust/device_vector.h>

using std::vector;
using std::string;

class Neuron {
public:
  Neuron();
  Neuron(const unsigned long num_input, const vector<double> &weight,
         const vector<double> &m, const vector<double> &nu,
         const unsigned long iteration, const double bias, const int activation_type,
         const double dropout_rate);

  void dropout(const double random_value);
  void learn(const double delta, const vector<double> &inputValues);
  double learn_output(const vector<double> &inputValues);
  double output(const vector<double> &inputValues);

  double getInputWeightIndexOf(const int i);
  double getDelta();
  double getBias();
  double getMIndexOf(const int i);
  double getNuIndexOf(const int i);

  unsigned long getIteration();

  string toString();

private:
  unsigned long num_input = 0;
  int activation_type = 0;
  double delta = 0.0; // 修正量
  double bias = 0.0; // ニューロンのバイアス // -threshold
  double alpha = 0.01;
  double epsilon = 0.00000001;
  double rambda = 0.00001; // 荷重減衰の定数．正の小さな定数にしておくことで勾配がゼロでも重みが減る
  double activation_identity(const double x); // 0
  double activation_sigmoid(const double x); // 1
  double activation_tanh(const double x); // 2
  double activation_relu(const double x); // 3

  double beta_one = 0.9;
  double beta_two = 0.999;
  unsigned long iteration;

  double dropout_rate; // どれくらいの割合で中間層ニューロンをDropoutさせるか
  double dropout_mask; // Dropoutのマスク値．0.0で殺して1.0で生かす

  vector<double> h_inputWeights;
  vector<double> h_m;
  vector<double> h_nu;

  thrust::device_vector<double> d_inputValues;
  thrust::device_vector<double> d_m;
  thrust::device_vector<double> d_nu;
  thrust::device_vector<double> d_inputWeights;
  thrust::device_vector<double> d_adam_result;
  thrust::device_vector<double> d_output_result;
  thrust::device_vector<double> d_learn_output_result;
};

#endif //NN_CUDA_NEURON_H
