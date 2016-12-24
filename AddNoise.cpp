
#include "AddNoise.h"
#include <vector>
#include <random>
#include <algorithm>

using std::vector;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;

/**
 * データごとに0.0以上1.0未満の乱数を生成し，rate未満であればそのデータを0.0にする
 * @param input ノイズをのせるデータ
 * @param rate ノイズをのせる確率
 * @return ノイズをのせたデータ
 */
vector<vector<double>> add_noise(const vector<vector<double>> &input, const float rate) {
  random_device rnd;
  mt19937 mt;
  mt.seed(rnd());
  uniform_real_distribution<double> real_rnd(0.0, 1.0);

  vector<vector<double>> result(input);

  for (unsigned long i = 0, i_s = result.size(); i < i_s; ++i) {
    for (unsigned long j = 0, j_s = result[i].size(); j < j_s; ++j) {
      if (real_rnd(mt) < rate) result[i][j] = 0.0;
    }
  }

  return result;
}

/**
 * データごとに0.0以上1.0未満の乱数を生成し，rate未満であればそのデータをデータセットの最小値以上最大値未満の乱数にする
 * @param input ノイズをのせるデータ
 * @param rate ノイズをのせる確率
 * @return ノイズをのせたデータ
 */
vector<vector<double>> add_random_noise(const vector<vector<double>> &input, const float rate) {
  random_device rnd;
  mt19937 mt;
  mt.seed(rnd());
  uniform_real_distribution<double> rnd_zero_one(0.0, 1.0);

  vector<vector<double>> result(input);

  for (unsigned long i = 0, i_s = result.size(); i < i_s; ++i) {
    double min = *std::min_element(result[i].begin(), result[i].end());
    double max = *std::max_element(result[i].begin(), result[i].end());
    uniform_real_distribution<double> rnd_val(min, max);
    for (unsigned long j = 0, j_s = result[i].size(); j < j_s; ++j) {
      if (rnd_zero_one(mt) < rate) {
        result[i][j] = rnd_val(mt);
      }
    }
  }

  return result;
}