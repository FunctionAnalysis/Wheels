#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "../../core/time.hpp"

#include "../../../tensor"

#include "../../math/auxmath.hpp"

#include "eigen.hpp"

using namespace wheels;
using namespace Eigen;

TEST(eigen, conversion) {
  Eigen::VectorXd ev1 = Eigen::VectorXd::Random(100);
  auto v1 = map(ev1);
  for (int i = 0; i < v1.numel(); i++) {
    ASSERT_EQ(v1[i], ev1[i]);
  }
  Eigen::MatrixXd em1 = Eigen::MatrixXd::Random(20, 30);
  auto m1 = map(em1);
  ASSERT_EQ(em1.rows(), m1.rows());
  ASSERT_EQ(em1.cols(), m1.cols());
  for (int i = 0; i < m1.rows(); i++) {
    for (int j = 0; j < m1.cols(); j++) {
      ASSERT_EQ(m1(i, j), em1(i, j));
    }
  }
}

TEST(eigen, time_compare) {
  std::default_random_engine rng;
  std::vector<Eigen::MatrixXd> A1s, B1s;
  std::vector<matx> A2s, B2s;
  A1s.reserve(1000);
  B1s.reserve(1000);
  A2s.reserve(1000);
  B2s.reserve(1000);

  // prepare data
  for (size_t i : iota(10)) {
    for (size_t j : iota(10)) {
      for (size_t k : iota(10)) {
        size_t m = (i + 1) * 10, n = (i + 1 + j) * 10, p = (k + 1) * 10;
        Eigen::MatrixXd A1 = Eigen::MatrixXd::Random(m, n);
        Eigen::MatrixXd B1 = Eigen::MatrixXd::Random(m, p);
        matx A2 = map(A1);
        matx B2 = map(B1);
        A1s.push_back(std::move(A1));
        B1s.push_back(std::move(B1));
        A2s.push_back(std::move(A2));
        B2s.push_back(std::move(B2));
      }
    }
  }

  { // ewise op test
    println("### ewise op test ###");
    std::vector<Eigen::MatrixXd> results1;
    std::vector<matx> results2;
    println("Eigen: ", time_cost([&A1s, &B1s, &results1]() {
              for (const auto &a : A1s) {
                results1.push_back(a.array() / 5.0 + a.array() + 1.0 -
                                   (a * 2.0).array());
              }
              for (const auto &b : B1s) {
                results1.push_back(b.cwiseProduct(b).array() - 1.0 + b.array());
              }
            }));
    println("wheels: ", time_cost([&A2s, &B2s, &results2]() {
              for (const auto &a : A2s) {
                results2.push_back(a.ewised() / 5.0 + a + 1.0 - a * 2.0);
              }
              for (const auto &b : B2s) {
                results2.push_back(b.ewised() * b - 1.0 + b);
              }
            }));

    // check valitidy
    ASSERT_EQ(results1.size(), results2.size());
    for (int i = 0; i < results1.size(); i++) {
      ASSERT_LE((map(results1[i]) - results2[i]).norm(), 0.01);
    }
  }

  { // solve test
    println("### solve test ###");
    std::vector<Eigen::MatrixXd> results1;
    std::vector<matx> results2;
    println("Eigen: ", time_cost([&A1s, &B1s, &results1]() {
              for (int k = 0; k < A1s.size(); k++) {
                results1.push_back(A1s[k].fullPivLu().solve(B1s[k]));
              }
            }));
    println("wheels: ", time_cost([&A2s, &B2s, &results2]() {
              for (int k = 0; k < A2s.size(); k++) {
                results2.push_back(auxmath::solve(A2s[k], B2s[k]));
              }
            }));

    // check valitidy
    ASSERT_EQ(results1.size(), results2.size());
    for (int i = 0; i < results1.size(); i++) {
      ASSERT_LE((A1s[i] * results1[i] - B1s[i]).norm(), 1e-3);
      ASSERT_LE((A2s[i] * results2[i] - B2s[i]).norm(), 1e-3);
    }
  }
}