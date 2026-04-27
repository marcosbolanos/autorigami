#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace LBFGSpp {

template <typename Scalar>
struct LBFGSParam {
    int max_iterations = 128;
    int m = 6;
    Scalar epsilon = static_cast<Scalar>(1e-6);
    Scalar alpha_init = static_cast<Scalar>(1.0);
    Scalar line_search_shrink = static_cast<Scalar>(0.5);
    Scalar min_step = static_cast<Scalar>(1e-10);
};

template <typename Scalar>
class LBFGSSolver {
public:
    explicit LBFGSSolver(const LBFGSParam<Scalar>& param) : param_(param) {}

    template <typename Functor>
    int minimize(Functor&& functor, Eigen::Matrix<Scalar, -1, 1>& x, Scalar& fx) const {
        if (x.size() == 0) {
            throw std::invalid_argument("LBFGSSolver requires non-empty x");
        }

        Eigen::Matrix<Scalar, -1, 1> grad(x.size());
        fx = functor(x, grad);

        for (int iteration = 0; iteration < param_.max_iterations; ++iteration) {
            if (grad.norm() <= param_.epsilon) {
                return iteration;
            }

            const Eigen::Matrix<Scalar, -1, 1> direction = -grad;
            Scalar step = param_.alpha_init;
            bool accepted = false;

            for (int ls = 0; ls < 20; ++ls) {
                const Eigen::Matrix<Scalar, -1, 1> candidate = x + step * direction;
                Eigen::Matrix<Scalar, -1, 1> candidate_grad(x.size());
                const Scalar candidate_fx = functor(candidate, candidate_grad);
                if (std::isfinite(candidate_fx) && candidate_fx < fx) {
                    x = candidate;
                    grad = candidate_grad;
                    fx = candidate_fx;
                    accepted = true;
                    break;
                }
                step *= param_.line_search_shrink;
                if (step < param_.min_step) {
                    break;
                }
            }

            if (!accepted) {
                return iteration;
            }
        }

        return param_.max_iterations;
    }

private:
    LBFGSParam<Scalar> param_;
};

}  // namespace LBFGSpp
