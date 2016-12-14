
#include <iostream>
#include <sstream>
#include "DenoisingAutoencoder.h"
#include <thread>

using std::vector;
using std::thread;
using std::string;
using std::stringstream;
using std::cout;
using std::endl;

DenoisingAutoencoder::DenoisingAutoencoder(const unsigned long num_input,
                                           const float compression_rate) {
  input_neuron_num = num_input;
  middle_neuron_num = (unsigned long) (num_input * (1.0 - compression_rate));
  output_neuron_num = num_input;
  middle_layer_type = 1;

  vector<double> emptyVector;
  middle_neurons.resize(middle_neuron_num);
  for (int neuron = 0; neuron < middle_neuron_num; ++neuron) {
    middle_neurons[neuron] = Neuron(input_neuron_num, emptyVector, emptyVector, emptyVector,
                                    emptyVector, emptyVector,
                                    0, 0.0, middle_layer_type, 0.0);
  }

  output_neurons.resize(output_neuron_num);
  for (int neuron = 0; neuron < output_neuron_num; ++neuron) {
    output_neurons[neuron] = Neuron(middle_neuron_num, emptyVector, emptyVector, emptyVector,
                                    emptyVector, emptyVector,
                                    0, 0.0, 0, 0.0);
  }

  h.resize(middle_neuron_num);
  o.resize(output_neuron_num);
  learnedH.resize(middle_neuron_num);
  learnedO.resize(output_neuron_num);
}

string DenoisingAutoencoder::learn(const vector<vector<double>> &input,
                                        const vector<vector<double>> &noisy_input) {
  int succeed = 0; // 連続正解回数のカウンタを初期化

  for (int trial = 0; trial < MAX_TRIAL; ++trial) {
    // Dropoutは無効にする
    for (int neuron = 0; neuron < middle_neuron_num; ++neuron)
      middle_neurons[neuron].dropout(1.0);
    for (int neuron = 0; neuron < output_neuron_num; ++neuron)
      output_neurons[neuron].dropout(1.0);

    // 使用する教師データを選択
    vector<double> in = noisy_input[trial % input.size()];
    vector<double> ans = input[trial % input.size()];

    //region Feed Forward
    vector<thread> threads(num_thread);
    unsigned long charge;
    threads.clear();
    if (middle_neuron_num <= num_thread) charge = 1;
    else charge = middle_neuron_num / num_thread;
    for (int i = 0; i < middle_neuron_num; i += charge)
      if (i != 0 && middle_neuron_num / i == 1)
        threads.push_back(thread(&DenoisingAutoencoder::middleForwardThread, this,
                                      ref(in), i, middle_neuron_num));
      else
        threads.push_back(thread(&DenoisingAutoencoder::middleForwardThread, this,
                                      ref(in), i, i + charge));
    for (thread &th : threads) th.join();

    threads.clear();
    if (output_neuron_num <= num_thread) charge = 1;
    else charge = output_neuron_num / num_thread;
    for (int i = 0; i < output_neuron_num; i += charge)
      if (i != 0 && output_neuron_num / i == 1)
        threads.push_back(thread(&DenoisingAutoencoder::outForwardThread, this,
                                      i, output_neuron_num));
      else
        threads.push_back(thread(&DenoisingAutoencoder::outForwardThread, this,
                                      i, i + charge));
    for (thread &th : threads) th.join();
    
    //endregion

    successFlg = true;

    //region Back Propagation (learn phase)
    threads.clear();
    if (output_neuron_num <= num_thread) charge = 1;
    else charge = output_neuron_num / num_thread;
    for (int i = 0; i < output_neuron_num; i += charge)
      if (i != 0 && output_neuron_num / i == 1)
        threads.push_back(thread(&DenoisingAutoencoder::outLearnThread, this,
                                      ref(in), ref(ans), i, output_neuron_num));
      else
        threads.push_back(thread(&DenoisingAutoencoder::outLearnThread, this,
                                      ref(in), ref(ans), i, i + charge));
    for (thread &th : threads) th.join();

    if (successFlg) {
      succeed++;
      if (succeed >= input.size()) break;
      else continue;
    } else succeed = 0;

    threads.clear();
    if (middle_neuron_num <= num_thread) charge = 1;
    else charge = middle_neuron_num / num_thread;
    for (int i = 0; i < middle_neuron_num; i += charge)
      if (i != 0 && middle_neuron_num / i == 1)
        threads.push_back(thread(&DenoisingAutoencoder::middleLearnThread, this,
                                      ref(in), i, middle_neuron_num));
      else
        threads.push_back(thread(&DenoisingAutoencoder::middleLearnThread, this,
                                      ref(in), i, i + charge));
    for (thread &th : threads) th.join();
    
    
  }

  // 全ての教師データで正解を出すか，収束限度回数を超えた場合に終了
  // エンコーダ部分である中間層ニューロンの各パラメータをstringに詰める
  stringstream ss;
  for (int neuron = 0; neuron < middle_neuron_num; ++neuron) {
    // 重みを詰める
    for (int weight_num = 0; weight_num < input_neuron_num; ++weight_num) {
      ss << middle_neurons[neuron].getInputWeightIndexOf(weight_num) << ',';
    }
    ss << '|';

    // Adamのmを詰める
    for (int mNum = 0; mNum < input_neuron_num; ++mNum) {
      ss << middle_neurons[neuron].getMIndexOf(mNum) << ',';
    }
    ss << '|';

    // Adamのnuを詰める
    for (int nuNum = 0; nuNum < input_neuron_num; ++nuNum) {
      ss << middle_neurons[neuron].getNuIndexOf(nuNum) << ',';
    }
    ss << '|';

    // Adamのm_hatを詰める
    for (int mHatNum = 0; mHatNum < input_neuron_num; ++mHatNum) {
      ss << middle_neurons[neuron].getMHatIndexOf(mHatNum) << ',';
    }
    ss << '|';

    // Adamのnu_hatを詰める
    for (int nuHatNum = 0; nuHatNum < input_neuron_num; ++nuHatNum) {
      ss << middle_neurons[neuron].getNuHatIndexOf(nuHatNum) << ',';
    }
    ss << '|';

    // Adamのiterationを詰める
    ss << middle_neurons[neuron].getIteration() << '|';

    // バイアスを入れ，最後に ' を入れる
    ss << middle_neurons[neuron].getBias() << '\'';
  }

  string neuron_params = ss.str();
  // 末尾の ' を削除する
  neuron_params.pop_back();
  ss.str("");
  ss.clear(stringstream::goodbit);

  return neuron_params;
}

vector<vector<double>>
DenoisingAutoencoder::getMiddleOutput(const vector<vector<double>> &noisy_input) {
  vector<vector<double>> middle_output(noisy_input.size());
  vector<thread> threads(num_thread);
  unsigned long charge;

  for (unsigned long set = 0, set_size = noisy_input.size(); set < set_size; ++set) {
    threads.clear();
    if (middle_neuron_num <= num_thread) charge = 1;
    else charge = middle_neuron_num / num_thread;
    for (int i = 0; i < middle_neuron_num; i += charge)
      if (i != 0 && middle_neuron_num / i == 1)
        threads.push_back(thread(&DenoisingAutoencoder::middleOutThread, this,
                                      ref(noisy_input[set]), i, middle_neuron_num));
      else
        threads.push_back(thread(&DenoisingAutoencoder::middleOutThread, this,
                                      ref(noisy_input[set]), i, i + charge));
    for (thread &th : threads) th.join();
    middle_output[set] = learnedH;
  }

  return middle_output;
}

void DenoisingAutoencoder::middleForwardThread(const vector<double> &in, const int begin,
                                               const int end) {
  for (int neuron = begin; neuron < end; ++neuron)
    h[neuron] = middle_neurons[neuron].learn_output(in);
}

void DenoisingAutoencoder::outForwardThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron)
    o[neuron] = output_neurons[neuron].learn_output(h);
}

void
DenoisingAutoencoder::outLearnThread(const vector<double> &in, const vector<double> &ans,
                                     const int begin, const int end) {
  // Dropoutを用いた学習済みNNの出力を得るようにする
  vector<double> output = out(in, false);
  for (int neuron = begin; neuron < end; ++neuron) {
    // 出力層ニューロンのdeltaの計算
    double delta = o[neuron] - ans[neuron];

    cout << "dA mse: " << mean_squared_error(o[neuron], ans[neuron]) << endl;

    // 教師データとの誤差が十分小さい場合は学習しない．そうでなければ正解フラグをfalseに
    if (mean_squared_error(o[neuron], ans[neuron]) < MAX_GAP) continue;
    else successFlg = false;

    // 出力層の学習
    output_neurons[neuron].learn(delta, h);
  }
}

void DenoisingAutoencoder::middleLearnThread(const vector<double> &in, const int begin,
                                             const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // 中間層ニューロンのdeltaを計算
    double sumDelta = 0.0;

    for (int k = 0; k < output_neuron_num; ++k) {
      Neuron n = output_neurons[k];
      sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
    }

    double delta;
    if (middle_layer_type == 0) delta = 1.0 * sumDelta;
    else if (middle_layer_type == 1) delta = (h[neuron] * (1.0 - h[neuron])) * sumDelta;
    else if (middle_layer_type == 2) delta = (1.0 - pow(h[neuron], 2)) * sumDelta;
    else {
      //ReLU
      if (h[neuron] > 0) delta = 1.0 * sumDelta;
      else delta = 0 * sumDelta;
    }

    // 学習
    middle_neurons[neuron].learn(delta, in);
  }
}

vector<double> DenoisingAutoencoder::out(const vector<double> &input,
                                              const bool showResult) {
  vector<thread> threads(num_thread);
  unsigned long charge = 1;
  threads.clear();
  if (middle_neuron_num <= num_thread) charge = 1;
  else charge = middle_neuron_num / num_thread;
  for (int i = 0; i < middle_neuron_num; i += charge)
    if (i != 0 && middle_neuron_num / i == 1)
      threads.push_back(thread(&DenoisingAutoencoder::middleOutThread, this,
                                    ref(input), i, middle_neuron_num));
    else
      threads.push_back(thread(&DenoisingAutoencoder::middleOutThread, this,
                                    ref(input), i, i + charge));
  for (thread &th : threads) th.join();

  threads.clear();
  if (output_neuron_num <= num_thread) charge = 1;
  else charge = output_neuron_num / num_thread;
  for (int i = 0; i < output_neuron_num; i += charge)
    if (i != 0 && output_neuron_num / i == 1)
      threads.push_back(thread(&DenoisingAutoencoder::outOutThread, this,
                                    i, output_neuron_num));
    else
      threads.push_back(thread(&DenoisingAutoencoder::outOutThread, this,
                                    i, i + charge));
  for (thread &th : threads) th.join();

  if (showResult) {
    for (int neuron = 0; neuron < output_neuron_num; ++neuron)
      cout << "output[" << neuron << "]: " << learnedO[neuron] << " ";
    cout << endl;
  }

  return learnedO;
}

void DenoisingAutoencoder::middleOutThread(const vector<double> &in,
                                           const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron)
    learnedH[neuron] = middle_neurons[neuron].output(in);
}

void DenoisingAutoencoder::outOutThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron)
    learnedO[neuron] = output_neurons[neuron].output(learnedH);
}

unsigned long DenoisingAutoencoder::getCurrentMiddleNeuronNum() {
  return middle_neuron_num;
}

double DenoisingAutoencoder::mean_squared_error(const double output, const double answer) {
  return (output - answer) * (output - answer) / 2;
}