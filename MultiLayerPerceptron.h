
#ifndef NN_CUDA_MULTILAYERPERCEPTRON_H
#define NN_CUDA_MULTILAYERPERCEPTRON_H


#include <thread>
#include <zconf.h>
#include "Neuron.cuh"

using std::vector;
using std::string;
using std::thread;

class MultiLayerPerceptron {
public:
  MultiLayerPerceptron(const unsigned long input, const unsigned long middle,
                       const unsigned long output, const unsigned long middle_layer,
                       const int middle_layer_type, const double dropout_rate, 
                       const string &sda_params);

  void learn(const vector<vector<double>> &x,
             const vector<vector<double>> &answer);

  string toString();

  vector<double> out(const vector<double> &input, const bool showResult);

private:
  static const unsigned int MAX_TRIAL = 100000; // 学習上限回数
  constexpr static const double MAX_GAP = 0.01; // 許容する誤差の域値
  unsigned long num_thread = (unsigned long) sysconf(_SC_NPROCESSORS_ONLN); // プロセッサのコア数
  
  // ニューロン数
  unsigned long input_neuron_num = 0;
  unsigned long middle_neuron_num = 0;
  unsigned long output_neuron_num = 0;

  unsigned long middle_layer_number = 0; // 中間層の層数

  unsigned long sda_layer_size = 0;

  int middle_layer_type = 0; // 中間層の活性化関数の種類指定．0: identity 1: sigmoid 2: tanh 3: ReLU

  bool successFlg = true;

  vector<thread> threads;

  vector<double> in;
  vector<double> ans;

  vector<vector<Neuron>> sda_neurons;
  vector<vector<double>> sda_out;

  vector<vector<Neuron>> middle_neurons; // 中間層は複数層用意する
  vector<Neuron> outputNeurons;
  vector<vector<double>> h;
  vector<double> o;

  vector<vector<double>> learnedH;
  vector<double> learnedO;

  void setupSdA(const string &sda_params, const double dropout_rate);

  vector<double> separate_by_camma(const string &input);

  void middleFirstLayerForwardThread(const int begin, const int end);
  void middleLayerForwardThread(const int layer, const int begin, const int end);
  void outForwardThread(const int begin, const int end);

  void outLearnThread(const int begin, const int end);
  void middleLastLayerLearnThread(const int begin, const int end);
  void middleMiddleLayerLearnThread(const int layer, const int begin, const int end);
  void middleFirstLayerLearnThread(const int begin, const int end);

  void sdaFirstLayerOutThread(const int begin, const int end);
  void sdaOtherLayerOutThread(const int layer, const int begin, const int end);

  void middleFirstLayerOutThread(const int begin, const int end);
  void middleLayerOutThread(const int layer, const int begin, const int end);
  void outOutThread(const int begin, const int end);

  double crossEntropy(const double output, const double answer);
};


#endif //NN_CUDA_MULTILAYERPERCEPTRON_H
