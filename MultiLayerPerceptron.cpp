
#include "MultiLayerPerceptron.h"
#include "Neuron.cuh"
#include "iostream"
#include <thread>
#include <sstream>

using std::vector;
using std::string;
using std::stringstream;
using std::stoul;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;
using std::thread;
using std::cout;
using std::endl;

/**
 * MultiLayerPerceptronのコンストラクタ
 * @param input 入力層のニューロン数
 * @param middle 中間層のニューロン数
 * @param output 出力層のニューロン数
 * @param middle_layer 中間層の層数
 * @param middle_layer_type 中間層の活性化関数の種類指定．0: identity 1: sigmoid 2: tanh 3: ReLU
 * @param dropout_rate Dropout率
 * @param sda_params Stacked Denoising Autoencoderのパラメータ
 * @return
 */
MultiLayerPerceptron::MultiLayerPerceptron(const unsigned long input, 
                                           const unsigned long middle,
                                           const unsigned long output,
                                           const unsigned long middle_layer,
                                           const int middle_layer_type,
                                           const double dropout_rate,
                                           const string &sda_params) {
  setupSdA(sda_params, dropout_rate);
  this->sda_layer_size = sda_neurons.size();

  // SdAの末尾レイヤの出力数がMLPの入力数となる
  this->input_neuron_num = sda_neurons[this->sda_layer_size - 1].size();
  this->middle_neuron_num = middle;
  this->output_neuron_num = output;
  this->middle_layer_number = middle_layer;
  this->middle_layer_type = middle_layer_type;

  this->middle_neurons.resize(middle_layer_number);
  for (vector<Neuron> n : this->middle_neurons) n.resize(middle_neuron_num);

  vector<Neuron> neuronPerLayer(middle_neuron_num);

  vector<double> emptyVector;

  for (int layer = 0; layer < middle_layer_number; ++layer) {
    if (layer == 0)
      for (int neuron = 0; neuron < middle_neuron_num; ++neuron)
        // 中間層の最初の層については，SdA末尾レイヤのニューロン数がニューロンへの入力数となる
        neuronPerLayer[neuron] = Neuron(input_neuron_num, emptyVector, emptyVector, emptyVector,
                                        0, 0.0, middle_layer_type, dropout_rate);
    else
      for (int neuron = 0; neuron < middle_neuron_num; ++neuron)
        // それ以降の層については，中間層の各層のニューロン数がニューロンへの入力数となる
        neuronPerLayer[neuron] = Neuron(middle_neuron_num, emptyVector, emptyVector, emptyVector,
                                        0, 0.0, middle_layer_type, dropout_rate);
    this->middle_neurons[layer] = neuronPerLayer;
  }

  this->outputNeurons.resize(output_neuron_num);
  for (int neuron = 0; neuron < output; ++neuron) {
    this->outputNeurons[neuron] = Neuron(middle_neuron_num, emptyVector, emptyVector, emptyVector,
                                         0, 0.0, 1, dropout_rate);
  }
}

void MultiLayerPerceptron::setupSdA(const string &sda_params, const double dropout_rate) {
  stringstream ss(sda_params);
  string item;
  vector<string> elems_per_sda;
  vector<string> elems_per_neuron;
  vector<string> elems_per_param;

  // $ でSdA単位で分割する（= SdA層のレイヤ数）
  while (getline(ss, item, '$')) if (!item.empty()) elems_per_sda.push_back(item);
  sda_neurons.resize(elems_per_sda.size());
  sda_out.resize(elems_per_sda.size());
  item = "";
  ss.str("");
  ss.clear(stringstream::goodbit);

  for (unsigned long sda = 0, n_l = elems_per_sda.size(); sda < n_l; ++sda) {
    // ' でニューロン単位で分割する
    ss = stringstream(elems_per_sda[sda]);
    while (getline(ss, item, '\'')) if (!item.empty()) elems_per_neuron.push_back(item);
    sda_neurons[sda].resize(elems_per_neuron.size());
    sda_out[sda].resize(elems_per_neuron.size());
    item = "";
    ss.str("");
    ss.clear(stringstream::goodbit);

    for (unsigned long neuron = 0, n_n = elems_per_neuron.size(); neuron < n_n; ++neuron) {
      // パラメータごとに分割する
      ss = stringstream(elems_per_neuron[neuron]);
      while (getline(ss, item, '|')) if (!item.empty()) elems_per_param.push_back(item);
      item = "";
      ss.str("");
      ss.clear(stringstream::goodbit);

      double bias = stod(elems_per_param.back());
      elems_per_param.pop_back();


      unsigned long iteration = stoul(elems_per_param.back());
      elems_per_param.pop_back();

      vector<double> weight = separate_by_camma(elems_per_param[0]);
      vector<double> m = separate_by_camma(elems_per_param[1]);
      vector<double> nu = separate_by_camma(elems_per_param[2]);

      sda_neurons[sda][neuron] = Neuron(weight.size(), weight, m, nu,
                                        iteration, bias, 1, dropout_rate);

      elems_per_param.clear();
    }
    elems_per_neuron.clear();
  }
  elems_per_sda.clear();
}

vector<double> MultiLayerPerceptron::separate_by_camma(const string &input) {
  vector<double> result;
  stringstream ss = stringstream(input);
  string item;
  while (getline(ss, item, ',')) if (!item.empty()) result.push_back(stod(item));
  item = "";
  ss.str("");
  ss.clear(stringstream::goodbit);

  return result;
}

/**
 * 教師入力データと教師出力データを元にニューラルネットワークを学習する
 * @param x 二次元の教師入力データ，データセット * データ
 * @param answer 教師入力データに対応した二次元の教師出力データ，データセット * データ
 */
void MultiLayerPerceptron::learn(const vector<vector<double>> &x,
                                 const vector<vector<double>> &answer) {
  h = vector<vector<double>>(middle_layer_number, vector<double>(middle_neuron_num, 0.0));
  o = vector<double>(output_neuron_num, 0.0);

  int succeed = 0; // 連続正解回数のカウンタを初期化

  random_device rnd; // 非決定的乱数生成器
  mt19937 mt; // メルセンヌ・ツイスタ
  mt.seed(rnd());
  uniform_real_distribution<double> real_rnd(0.0, 1.0); // 0.0以上1.0未満の範囲で値を生成する

  for (int trial = 0; trial < this->MAX_TRIAL; ++trial) {

    for (int layer = 0; layer < sda_layer_size; ++layer) {
      for (unsigned long neuron = 0, num_neuron = sda_neurons[layer].size();
           neuron < num_neuron; ++neuron) {
        sda_neurons[layer][neuron].dropout(real_rnd(mt));
      }
    }
    for (int layer = 0; layer < middle_layer_number; ++layer) {
      for (int neuron = 0; neuron < middle_neuron_num; ++neuron) {
        middle_neurons[layer][neuron].dropout(real_rnd(mt));
      }
    }
    for (int neuron = 0; neuron < output_neuron_num; ++neuron) {
      outputNeurons[neuron].dropout(1.0); // 出力層ニューロンはDropoutさせない
    }

    // 使用する教師データを選択
    vector<double> in = x[trial % answer.size()]; // 利用する教師入力データ
    vector<double> ans = answer[trial % answer.size()]; // 教師出力データ

    vector<thread> threads(num_thread);
    unsigned long charge;

    // Feed Forward
    // SdA First Layer
    threads.clear();
    if (sda_neurons[0].size() <= num_thread) charge = 1;
    else charge = sda_neurons[0].size() / num_thread;
    for (unsigned long i = 0, num_neuron = sda_neurons[0].size(); i < num_neuron; i += charge) {
      if (i != 0 && num_neuron / i == 1) {
        threads.push_back(thread(&MultiLayerPerceptron::sdaFirstLayerOutThread, this,
                                      ref(in), i, num_neuron));
      } else {
        threads.push_back(thread(&MultiLayerPerceptron::sdaFirstLayerOutThread, this,
                                      ref(in), i, i + charge));
      }
    }
    for (thread &th : threads) th.join();

    // SdA Other Layer
    if (sda_layer_size > 1) {
      for (unsigned long layer = 1, last_layer = sda_layer_size - 1;
           layer <= last_layer; ++layer) {
        threads.clear();
        if (sda_neurons[layer].size() <= num_thread) charge = 1;
        else charge = sda_neurons[layer].size() / num_thread;
        for (unsigned long i = 0, num_neuron = sda_neurons[layer].size();
             i < num_neuron; i += charge) {
          if (i != 0 && num_neuron / i == 1) {
            threads.push_back(thread(&MultiLayerPerceptron::sdaOtherLayerOutThread, this,
                                     layer, i, num_neuron));
          } else {
            threads.push_back(thread(&MultiLayerPerceptron::sdaOtherLayerOutThread, this,
                                     layer, i, i + charge));
          }
        }
        for (thread &th : threads) th.join();
      }
    }


    // 1層目の中間層の出力計算
    threads.clear();
    if (middle_neuron_num <= num_thread) charge = 1;
    else charge = middle_neuron_num / num_thread;
    for (int i = 0; i < middle_neuron_num; i += charge) {
      if (i != 0 && middle_neuron_num / i == 1) {
        threads.push_back(thread(&MultiLayerPerceptron::middleFirstLayerForwardThread, this,
                                 i, middle_neuron_num));
      } else {
        threads.push_back(thread(&MultiLayerPerceptron::middleFirstLayerForwardThread, this,
                                 i, i + charge));
      }
    }
    for (thread &th : threads) th.join();

    // 一つ前の中間層より得られた出力を用いて，以降の中間層を順に計算
    if (middle_layer_number > 1) {
      if (middle_neuron_num <= num_thread) charge = 1;
      else charge = middle_neuron_num / num_thread;
      for (unsigned long layer = 1, last_layer = middle_layer_number - 1;
           layer <= last_layer; ++layer) {
        threads.clear();
        for (int i = 0; i < middle_neuron_num; i += charge) {
          if (i != 0 && middle_neuron_num / i == 1) {
            threads.push_back(thread(&MultiLayerPerceptron::middleLayerForwardThread, this,
                                     layer, i, middle_neuron_num));
          } else {
            threads.push_back(thread(&MultiLayerPerceptron::middleLayerForwardThread, this,
                                     layer, i, i + charge));
          }
        }
        for (thread &th : threads) th.join();
      }
    }

    // 出力値を推定：中間層の最終層の出力を用いて，出力層の出力計算
    threads.clear();
    if (output_neuron_num <= num_thread) charge = 1;
    else charge = output_neuron_num / num_thread;
    for (int i = 0; i < output_neuron_num; i += charge) {
      if (i != 0 && output_neuron_num / i == 1) {
        threads.push_back(thread(&MultiLayerPerceptron::outForwardThread, this,
                                 i, output_neuron_num));
      } else {
        threads.push_back(thread(&MultiLayerPerceptron::outForwardThread, this,
                                 i, i + charge));
      }
    }
    for (thread &th : threads) th.join();

    successFlg = true;

    // Back Propagation (learn phase)
    //region 出力層を学習する
    threads.clear();
    if (output_neuron_num <= num_thread) charge = 1;
    else charge = output_neuron_num / num_thread;
    for (int i = 0; i < output_neuron_num; i += charge) {
      if (i != 0 && output_neuron_num / i == 1) {
        threads.push_back(thread(&MultiLayerPerceptron::outLearnThread, this,
                                 ref(in), ref(ans), i, output_neuron_num));
      } else {
        threads.push_back(thread(&MultiLayerPerceptron::outLearnThread, this,
                                 ref(in), ref(ans), i, i + charge));
      }
    }
    for (thread &th : threads) th.join();
    //endregion

    // 連続成功回数による終了判定
    if (successFlg) {
      succeed++;
      if (succeed >= x.size()) break;
      else continue;
    } else succeed = 0;

    //region 中間層の更新．末尾層から先頭層に向けて更新する

    //region 中間層の層数が2以上の場合のみ，中間層の最終層の学習をする
    if (middle_layer_number > 1) {
      threads.clear();
      if (middle_neuron_num <= num_thread) charge = 1;
      else charge = middle_neuron_num / num_thread;
      for (int i = 0; i < middle_neuron_num; i += charge) {
        if (i != 0 && middle_neuron_num / i == 1) {
          threads.push_back(thread(&MultiLayerPerceptron::middleLastLayerLearnThread, this,
                                   i, middle_neuron_num));
        } else {
          threads.push_back(thread(&MultiLayerPerceptron::middleLastLayerLearnThread, this,
                                   i, i + charge));
        }
      }
      for (thread &th : threads) th.join();
    }
    //endregion

    //region 出力層と入力層に最も近い層一つずつを除いた残りの中間層を入力層に向けて学習する
    if (middle_layer_number > 2) {
      if (middle_neuron_num <= num_thread) charge = 1;
      else charge = middle_neuron_num / num_thread;
      for (unsigned long layer = middle_layer_number - 2; layer >= 1; --layer) {
        threads.clear();
        for (int i = 0; i < middle_neuron_num; i += charge) {
          if (i != 0 && middle_neuron_num / i == 1) {
            threads.push_back(thread(&MultiLayerPerceptron::middleMiddleLayerLearnThread, this,
                                     layer, i, middle_neuron_num));
          } else {
            threads.push_back(thread(&MultiLayerPerceptron::middleMiddleLayerLearnThread, this,
                                     layer, i, i + charge));
          }
        }
        for (thread &th : threads) th.join();
      }
    }
    //endregion

    //region 中間層の最初の層を学習する
    threads.clear();
    if (middle_neuron_num <= num_thread) charge = 1;
    else charge = middle_neuron_num / num_thread;
    for (int i = 0; i < middle_neuron_num; i += charge) {
      if (i != 0 && middle_neuron_num / i == 1) {
        threads.push_back(thread(&MultiLayerPerceptron::middleFirstLayerLearnThread, this,
                                 i, middle_neuron_num));
      } else {
        threads.push_back(thread(&MultiLayerPerceptron::middleFirstLayerLearnThread, this,
                                 i, i + charge));
      }
    }
    for (thread &th : threads) th.join();
    //endregion

    if (sda_layer_size > 1) {
      threads.clear();
      if (sda_neurons[sda_layer_size - 1].size() <= num_thread) charge = 1;
      else charge = sda_neurons[sda_layer_size - 1].size() / num_thread;
      for (unsigned long i = 0, num_neuron = sda_neurons[sda_layer_size - 1].size();
           i < num_neuron; i += charge) {
        if (i != 0 && num_neuron / i == 1) {
          threads.push_back(thread(&MultiLayerPerceptron::sdaLastLayerLearnThread, this,
                                        i, num_neuron));
        } else {
          threads.push_back(thread(&MultiLayerPerceptron::sdaLastLayerLearnThread, this,
                                        i, i + charge));
        }
      }
      for (thread &th : threads) th.join();
    }

    if (sda_layer_size > 2) {
      for (unsigned long layer = sda_layer_size - 2; layer >= 1; --layer) {
        unsigned long num_neuron = sda_neurons[layer].size();
        if (num_neuron <= num_thread) charge = 1;
        else charge = num_neuron / num_thread;
        threads.clear();
        for (int i = 0; i < num_neuron; i += charge) {
          if (i != 0 && num_neuron / i == 1) {
            threads.push_back(thread(&MultiLayerPerceptron::sdaMiddleLayerLearnThread, this,
                                     layer, i, num_neuron));
          } else {
            threads.push_back(thread(&MultiLayerPerceptron::sdaMiddleLayerLearnThread, this,
                                     layer, i, i + charge));
          }
        }
        for (thread &th : threads) th.join();
      }
    }

    threads.clear();
    unsigned long num_neuron = sda_neurons[0].size();
    if (num_neuron <= num_thread) charge = 1;
    else charge = num_neuron / num_thread;
    for (int i = 0; i < num_neuron; i += charge) {
      if (i != 0 && num_neuron / i == 1) {
        threads.push_back(thread(&MultiLayerPerceptron::sdaFirstLayerLearnThread, this,
                                      ref(in), i, num_neuron));
      } else {
        threads.push_back(thread(&MultiLayerPerceptron::sdaFirstLayerLearnThread, this,
                                      ref(in), i, i + charge));
      }
    }
    for (thread &th : threads) th.join();

    //endregion
  }

  // 全ての教師データで正解を出すか，収束限度回数を超えた場合に終了
}

/**
 * ニューラルネットワークの状態をまとめた文字列を返す
 * @return  ニューラルネットワークの状態（重み付け）をまとめた文字列
 */
string MultiLayerPerceptron::toString() {
  // 戻り値変数
  string str = "";

  // 中間層ニューロン出力
  str += " middle neurons ( ";
  for (int layer = 0; layer < middle_layer_number; ++layer) {
    for (int neuron = 0; neuron < middle_neuron_num; ++neuron) {
      str += middle_neurons[layer][neuron].toString();
    }
  }
  str += ") ";

  // 出力層ニューロン出力
  str += " output neurons ( ";
  for (int neuron = 0; neuron < output_neuron_num; ++neuron) {
    str += outputNeurons[neuron].toString();
  }
  str += ") ";

  return str;
}

void MultiLayerPerceptron::sdaFirstLayerOutThread(const vector<double> &in,
                                                  const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron)
    sda_out[0][neuron] = sda_neurons[0][neuron].output(in);
}

void MultiLayerPerceptron::sdaOtherLayerOutThread(const int layer,
                                                  const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    sda_out[layer][neuron] = sda_neurons[layer][neuron].output(sda_out[layer - 1]);
  }
}

void MultiLayerPerceptron::middleFirstLayerForwardThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // SdAの最終層の出力を入れる
    h[0][neuron] = middle_neurons[0][neuron].learn_output(sda_out[sda_layer_size - 1]);
  }
}

void MultiLayerPerceptron::middleLayerForwardThread(const int layer,
                                                    const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    h[layer][neuron] = middle_neurons[layer][neuron].learn_output(h[layer - 1]);
  }
}

void MultiLayerPerceptron::outForwardThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    o[neuron] = outputNeurons[neuron].learn_output(h[middle_layer_number - 1]);
  }
}

/**
 * 出力層の学習，スレッドを用いて並列学習するため，学習するニューロンの開始点と終了点も必要
 * 誤差関数には交差エントロピーを，活性化関数にシグモイド関数を用いるため，deltaは 出力 - 教師出力 で得られる
 * @param in 入力データ
 * @param ans 教師出力データ
 * @param o 出力層の出力データ
 * @param h 中間層の出力データ
 * @param begin 学習するニューロンセットの開始点
 * @param end 学習するニューロンセットの終了点
 */
void MultiLayerPerceptron::outLearnThread(const vector<double> &in, const vector<double> &ans,
                                          const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // 出力層ニューロンのdeltaの計算
    double delta = o[neuron] - ans[neuron];

    cout << "MLP ce: " << crossEntropy(o[neuron], ans[neuron]) << endl;

    // 教師データとの誤差が十分小さい場合は学習しない．そうでなければ正解フラグをfalseに
    if (crossEntropy(o[neuron], ans[neuron]) < MAX_GAP) continue;
    else successFlg = false;

    // 出力層の学習
    outputNeurons[neuron].learn(delta, h[middle_layer_number - 1]);
  }
}

/**
 * 中間層の最終層の学習．中間層の層数が2以上の場合のみこれを使う．
 * 活性化関数に何を使うかで，deltaの計算式が変わる
 * @param h 中間層の出力データ
 * @param begin 学習するニューロンセットの開始点
 * @param end 学習するニューロンセットの終了点
 */
void MultiLayerPerceptron::middleLastLayerLearnThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // 中間層ニューロンのdeltaを計算
    double sumDelta = 0.0;
    for (int k = 0; k < output_neuron_num; ++k) {
      Neuron n = outputNeurons[k];
      sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
    }

    // どの活性化関数を用いるかで，deltaの計算方法が変わる
    double delta;
    if (middle_layer_type == 0) {
      delta = 1.0 * sumDelta;
    } else if (middle_layer_type == 1) {
      delta = (h[middle_layer_number - 1][neuron]
               * (1.0 - h[middle_layer_number - 1][neuron])) * sumDelta;
    } else if (middle_layer_type == 2) {
      delta = (1.0 - pow(h[middle_layer_number - 1][neuron], 2)) * sumDelta;
    } else {
      // ReLU
      if (h[middle_layer_number - 1][neuron] > 0) delta = 1.0 * sumDelta;
      else delta = 0 * sumDelta;
    }

    // 学習
    middle_neurons[middle_layer_number - 1][neuron].learn(delta, h[middle_layer_number - 2]);
  }
}

/**
 * 出力層と入力層に最も近い層一つずつを除いた残りの中間層を入力層に向けて学習する．中間層が3層以上の場合にこれを使う．
 * @param h 中間層の出力データ
 * @param begin 学習するニューロンセットの開始点
 * @param end 学習するニューロンセットの終了点
 */
void MultiLayerPerceptron::middleMiddleLayerLearnThread(const int layer,
                                                        const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // 中間層ニューロンのdeltaを計算
    double sumDelta = 0.0;
    for (int k = 0; k < middle_neuron_num; ++k) {
      Neuron n = middle_neurons[layer + 1][k];
      sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
    }

    double delta;
    if (middle_layer_type == 0) {
      delta = 1.0 * sumDelta;
    } else if (middle_layer_type == 1) {
      delta = (h[layer][neuron] * (1.0 - h[layer][neuron])) * sumDelta;
    } else if (middle_layer_type == 2) {
      delta = (1.0 - pow(h[layer][neuron], 2)) * sumDelta;
    } else {
      // ReLU
      if (h[layer][neuron] > 0) delta = 1.0 * sumDelta;
      else delta = 0 * sumDelta;
    }

    // 学習
    middle_neurons[layer][neuron].learn(delta, h[layer - 1]);
  }
}

/**
 * 中間層の最初の層を学習する
 * @param h 中間層の出力データ
 * @param in 教師入力データ
 * @param begin 学習するニューロンセットの開始点
 * @param end 学習するニューロンセットの終了点
 */
void MultiLayerPerceptron::middleFirstLayerLearnThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    // 中間層ニューロンのdeltaを計算
    double sumDelta = 0.0;

    if (middle_layer_number > 1) {
      for (int k = 0; k < middle_neuron_num; ++k) {
        Neuron n = middle_neurons[1][k];
        sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
      }
    } else {
      for (int k = 0; k < output_neuron_num; ++k) {
        Neuron n = outputNeurons[k];
        sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
      }
    }

    double delta;
    if (middle_layer_type == 0) {
      delta = 1.0 * sumDelta;
    } else if (middle_layer_type == 1) {
      delta = (h[0][neuron] * (1.0 - h[0][neuron])) * sumDelta;
    } else if (middle_layer_type == 2) {
      delta = (1.0 - pow(h[0][neuron], 2)) * sumDelta;
    } else {
      // ReLU
      if (h[0][neuron] > 0) delta = 1.0 * sumDelta;
      else delta = 0 * sumDelta;
    }

    // 学習
    middle_neurons[0][neuron].learn(delta, sda_out[sda_layer_size - 1]);
  }
}

void MultiLayerPerceptron::sdaLastLayerLearnThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    double sumDelta = 0.0;
    for (unsigned long k = 0, num_neuron = middle_neurons[0].size(); k < num_neuron; ++k) {
      Neuron n = middle_neurons[0][k];
      sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
    }

    double delta;

    // sigmoid
    delta = (sda_out[sda_layer_size - 1][neuron]
             * (1.0 - sda_out[sda_layer_size - 1][neuron])) * sumDelta;

    sda_neurons[sda_layer_size - 1][neuron].learn(delta, sda_out[sda_layer_size - 2]);
  }
}

void MultiLayerPerceptron::sdaMiddleLayerLearnThread(const int layer, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    double sumDelta = 0.0;
    for (unsigned long k = 0, num_neuron = sda_neurons[layer + 1].size(); k < num_neuron; ++k) {
      Neuron n = sda_neurons[layer + 1][k];
      sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
    }

    double delta;

    // sigmoid
    delta = (sda_out[layer][neuron] * (1.0 - sda_out[layer][neuron])) * sumDelta;

    sda_neurons[layer][neuron].learn(delta, sda_out[layer - 1]);
  }
}

void MultiLayerPerceptron::sdaFirstLayerLearnThread(const vector<double> &in, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    double sumDelta = 0.0;

    if (sda_layer_size > 1) {
      for (unsigned long k = 0, num_neuron = sda_neurons[1].size(); k < num_neuron; ++k) {
        Neuron n = sda_neurons[1][k];
        sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
      }
    } else {
      for (unsigned long k = 0, num_neuron = middle_neurons[0].size(); k < num_neuron; ++k) {
        Neuron n = middle_neurons[0][k];
        sumDelta += n.getInputWeightIndexOf(neuron) * n.getDelta();
      }
    }

    double delta;

    // sigmoid
    delta = (sda_out[0][neuron] * (1.0 - sda_out[0][neuron])) * sumDelta;

    sda_neurons[0][neuron].learn(delta, in);
  }
}

/**
 * 与えられたデータをニューラルネットワークに入力し，出力を返す
 * @param input ニューラルネットワークに入力するデータ
 * @param showResult 結果をコンソールに出力するかを指定する
 */
vector<double> MultiLayerPerceptron::out(const vector<double> &input, const bool showResult) {
  // Feed Forward
  // SdA First Layer
  vector<thread> threads(num_thread);
  unsigned long charge;
  threads.clear();
  if (sda_neurons[0].size() <= num_thread) charge = 1;
  else charge = sda_neurons[0].size() / num_thread;
  for (unsigned long i = 0, num_neuron = sda_neurons[0].size(); i < num_neuron; i += charge) {
    if (i != 0 && num_neuron / i == 1) {
      threads.push_back(thread(&MultiLayerPerceptron::sdaFirstLayerOutThread, this,
                                    ref(input), i, num_neuron));
    } else {
      threads.push_back(thread(&MultiLayerPerceptron::sdaFirstLayerOutThread, this,
                                    ref(input), i, i + charge));
    }
  }
  for (thread &th : threads) th.join();

  // SdA Other Layer
  if (sda_layer_size > 1) {
    for (unsigned long layer = 1, last_layer = sda_layer_size - 1;
         layer <= last_layer; ++layer) {
      threads.clear();
      if (sda_neurons[layer].size() <= num_thread) charge = 1;
      else charge = sda_neurons[layer].size() / num_thread;
      for (unsigned long i = 0, num_neuron = sda_neurons[layer].size();
           i < num_neuron; i += charge) {
        if (i != 0 && num_neuron / i == 1) {
          threads.push_back(thread(&MultiLayerPerceptron::sdaOtherLayerOutThread, this,
                                   layer, i, num_neuron));
        } else {
          threads.push_back(thread(&MultiLayerPerceptron::sdaOtherLayerOutThread, this,
                                   layer, i, i + charge));
        }
      }
      for (thread &th : threads) th.join();
    }
  }


  learnedH = vector<vector<double>>(middle_layer_number, vector<double>(middle_neuron_num, 0));
  learnedO = vector<double>(output_neuron_num, 0);

  threads.clear();
  if (middle_neuron_num <= num_thread) charge = 1;
  else charge = middle_neuron_num / num_thread;
  for (int i = 0; i < middle_neuron_num; i += charge) {
    if (i != 0 && middle_neuron_num / i == 1) {
      threads.push_back(thread(&MultiLayerPerceptron::middleFirstLayerOutThread, this,
                               i, middle_neuron_num));
    } else {
      threads.push_back(thread(&MultiLayerPerceptron::middleFirstLayerOutThread, this,
                               i, i + charge));
    }
  }
  for (thread &th : threads) th.join();

  if (middle_layer_number > 1) {
    if (middle_neuron_num <= num_thread) charge = 1;
    else charge = middle_neuron_num / num_thread;
    for (unsigned long layer = 1, last_layer = middle_layer_number - 1;
         layer <= last_layer; ++layer) {
      threads.clear();
      for (int i = 0; i < middle_neuron_num; i += charge) {
        if (i != 0 && middle_neuron_num / i == 1) {
          threads.push_back(thread(&MultiLayerPerceptron::middleLayerOutThread, this,
                                   layer, i, middle_neuron_num));
        } else {
          threads.push_back(thread(&MultiLayerPerceptron::middleLayerOutThread, this,
                                   layer, i, i + charge));
        }
      }
      for (thread &th : threads) th.join();
    }
  }

  threads.clear();
  if (output_neuron_num <= num_thread) charge = 1;
  else charge = output_neuron_num / num_thread;
  for (int i = 0; i < output_neuron_num; i += charge) {
    if (i != 0 && output_neuron_num / i == 1) {
      threads.push_back(thread(&MultiLayerPerceptron::outOutThread, this,
                               i, output_neuron_num));
    } else {
      threads.push_back(thread(&MultiLayerPerceptron::outOutThread, this,
                               i, i + charge));
    }
  }
  for (thread &th : threads) th.join();

  if (showResult) {
    for (int neuron = 0; neuron < output_neuron_num; ++neuron) {
      cout << "output[" << neuron << "]: " << learnedO[neuron] << " ";
    }
    cout << endl;
  }

  return learnedO;
}

void MultiLayerPerceptron::middleFirstLayerOutThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    learnedH[0][neuron] = middle_neurons[0][neuron].output(sda_out[sda_layer_size - 1]);
  }
}

void MultiLayerPerceptron::middleLayerOutThread(const int layer, const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    learnedH[layer][neuron] = middle_neurons[layer][neuron].output(learnedH[layer - 1]);
  }
}

void MultiLayerPerceptron::outOutThread(const int begin, const int end) {
  for (int neuron = begin; neuron < end; ++neuron) {
    learnedO[neuron] = outputNeurons[neuron].output(learnedH[middle_layer_number - 1]);
  }
}

double MultiLayerPerceptron::crossEntropy(const double output, const double answer) {
  return -answer * log(output) - (1.0 - answer) * log(1.0 - output);
}
