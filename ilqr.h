#ifndef INC_2019_ILQR_H
#define INC_2019_ILQR_H

#include <vector>
#include <chrono>
#include <iostream>

#include "extern/eigen/Eigen/Dense"

/*
 * Convenience definitions
 */

template<int M, int N>
using mat = Eigen::Matrix<double, M, N>;

template<int N>
using vec = Eigen::Matrix<double, N, 1>;

/*
 * Straightforward templated iLQR implementation 
 *
 * Template arguments:
 * X: number of dimensions of state space
 * U: number of dimensions of action space
 * T: horizon (number of discrete time steps)
 */
template<int X, int U>
struct ILQR {

    /*
     * The following variable names may differ from the usual naming scheme to
     * better align with the symbols used in the reference paper "Synthesis and
     * Stabilization of Complex Behaviors through Online Trajectory Optimization"
     */

    // current state sequence
    std::vector<vec<X>> x = {vec<X>::Zero()};
    // current action sequence
    std::vector<vec<U>> u;

    // previous state sequence
    std::vector<vec<X>> prev_x;
    // previous action sequence
    std::vector<vec<U>> prev_u;

    // initial cost
    double trajCosts = -1.0;
    // control limits
    vec<U> uMax = vec<U>::Zero();
    vec<U> uMin = vec<U>::Zero();

    // parts of the dynamics function jacobian wrt. state
    std::vector<mat<X, X>> fx;
    // parts of the dynamics function jacobian wrt. action
    std::vector<mat<X, U>> fu;

    // cost function gradient 
    std::vector<vec<X + U>> l;
    // parts of the cost function gradient wrt. state
    std::vector<vec<X>> lx;
    // parts of the cost function gradient wrt. action
    std::vector<vec<U>> lu;

    // cost function hessian
    std::vector<mat<X + U, X + U>> L;
    // parts of the cost function hessian wrt. state, state
    std::vector<mat<X, X>> lxx;
    // parts of the cost function hessian wrt. action, action
    std::vector<mat<U, U>> luu;
    // parts of the cost function hessian wrt. action, state
    std::vector<mat<U, X>> lux;

    // gradient of value wrt. state
    vec<X> Vx;
    // hessian of value wrt. state, state
    mat<X, X> Vxx;

    // gradient of cost-to-go wrt. state
    vec<X> Qx;
    // gradient of cost-to-go wrt. action
    vec<U> Qu;
    // hessian of cost-to-go wrt. state, state
    mat<X, X> Qxx;
    // hessian of cost-to-go wrt. action, action
    mat<U, U> Quu;
    // hessian of cost-to-go wrt. action, state
    mat<U, X> Qux;

    // constant control components
    std::vector<vec<U>> k;
    // linear control components
    std::vector<mat<U, X>> K;

    // horizon
    size_t T = 20;

    // amount of iterations done in last update
    size_t iterations = 0;

    // maximum amount of iterations to improve trajectory
    size_t maxIterations = 5;

    // the wall clock time needed for one update
    double elapsedUpdateTime = 0;

    // whether to use the bfgs instead of finite diff. for costs
    bool useBfgs = false;
    
    ILQR() {
        uMax = uMax.array() + INFINITY;
        uMin = uMin.array() - INFINITY;
    }

    virtual vec<X> dynamics(vec<X>, vec<U>, int) = 0;
    virtual double costs(vec<X>, vec<U>, int) = 0;

    /*
     * Helper functions for differentiation
     */

    double diffEps = 10.0e-4;

    virtual mat<X, X + U> jacobian(vec<X> x, vec<U> u, int t) {

        mat<X, X + U> jac = mat<X, X + U>::Zero();

        for (int i = 0; i < X; i += 1) {
            vec<X> h = vec<X>::Zero();
            h[i] = diffEps;
            vec<X> x0 = x + h;
            vec<X> x1 = x - h;
            jac.col(i) = (dynamics(x0, u, t) - dynamics(x1, u, t)) / (2.0 * diffEps);
        }
        for (int i = 0; i < U; i += 1) {
            vec<U> h = vec<U>::Zero();
            h[i] = diffEps;
            vec<U> u0 = u + h;
            vec<U> u1 = u - h;
            jac.col(X + i) = (dynamics(x, u0, t) - dynamics(x, u1, t)) / (2.0 * diffEps);
        }

        return jac;
    }

    virtual vec<X + U> gradient(vec<X> x, vec<U> u, int t) {

        vec<X + U> grad = vec<X + U>::Zero();

        for (int i = 0; i < X; i += 1) {
            vec<X> h = vec<X>::Zero();
            h[i] = diffEps;
            vec<X> x0 = x + h;
            vec<X> x1 = x - h;
            grad[i] = (costs(x0, u, t) - costs(x1, u, t)) / (2.0 * diffEps);
        }
        for (int i = 0; i < U; i += 1) {
            vec<U> h = vec<U>::Zero();
            h[i] = diffEps;
            vec<U> u0 = u + h;
            vec<U> u1 = u - h;
            grad[X + i] = (costs(x, u0, t) - costs(x, u1, t)) / (2.0 * diffEps);
        }

        return grad;
    }

    virtual mat<X + U, X + U> hessian(
            vec<X> x,
            vec<U> u,
            int t,
            vec<X + U>& grad) {

        mat<X + U, X + U> hess = mat<X + U, X + U>::Zero();

        for (int i = 0; i < X; i += 1) {
            vec<X> h = vec<X>::Zero();
            h[i] = diffEps;
            vec<X> x0 = x + h;
            hess.col(i) = (gradient(x0, u, t) - grad) / diffEps;
        }
        for (int i = 0; i < U; i += 1) {
            vec<U> h = vec<U>::Zero();
            h[i] = diffEps;
            vec<U> u0 = u + h;
            hess.col(X + i) = (gradient(x, u0, t) - grad) / diffEps;
        }

        return hess;
    }

    virtual void bfgsUpdate(
            vec<X> x,
            vec<U> u,
            int t,
            vec<X> prev_x,
            vec<U> prev_u,
            vec<X + U>& grad,
            mat<X + U, X + U>& hessian) {

        vec<X + U> s;
        s << x - prev_x, u - prev_u;

        vec<X + U> newGrad = gradient(x, u, t);
        vec<X + U> y = newGrad - grad;
        grad = newGrad;

        double d = y.transpose() * s;

        if (d == 0) { 
            return;
        }

        mat<X + U, X + U> firstMat = (y * y.transpose()) / d;
        vec<X + U> v = hessian * s;
        mat<X + U, X + U> secondMat = (v * v.transpose()) / (s.transpose() * v);

        hessian += firstMat - secondMat;
    }

    bool update() {

        if (x.size() != T) {

            x.resize(T, vec<X>::Zero());
            u.resize(T, vec<U>::Zero());

            prev_x.resize(T, vec<X>::Zero());
            prev_u.resize(T, vec<U>::Zero());

            fx.resize(T, mat<X, X>::Zero());
            fu.resize(T, mat<X, U>::Zero());

            l.resize(T, vec<X + U>::Zero());
            lx.resize(T, vec<X>::Zero());
            lu.resize(T, vec<U>::Zero());

            L.resize(T, mat<X + U, X + U>::Zero());
            lxx.resize(T, mat<X, X>::Zero());
            luu.resize(T, mat<U, U>::Zero());
            lux.resize(T, mat<U, X>::Zero());

            k.resize(T, vec<U>::Zero());
            K.resize(T, mat<U, X>::Zero());
        }

        std::chrono::steady_clock::time_point begin
            = std::chrono::steady_clock::now();
        
        bool trajectoryChanged = true;
        bool improved = false;

        double mu = 1.0;
        double minMu = 10e-6;
        double muDelta = 2;
        double minMuDelta = 2;

        // do intial trajectory rollout 
        trajCosts = 0.0;
        for (size_t t = 0; t < T-1; t += 1) {
            x[t+1] = dynamics(x[t], u[t], t);
            trajCosts += costs(x[t], u[t], t);
        }
        trajCosts += costs(x[T-1], u[T-1], T-1);

        for (size_t s = 0; s < maxIterations; s += 1) { 

            if (trajectoryChanged) {
                // recalculate derivatives around new trajectory
                #pragma omp parallel for
                for (size_t t = 0; t < T; t += 1) {
                    mat<X, X + U> dynamicsJacobian = 
                        jacobian(x[t], u[t], t);
                    fx[t] = dynamicsJacobian.block(0, 0, X, X);
                    fu[t] = dynamicsJacobian.block(0, X, X, U);

                    if (s != 0 && useBfgs) {
                        bfgsUpdate(x[t], u[t], t, prev_x[t], prev_u[t], l[t], L[t]);
                    } else {
                        l[t] = gradient(x[t], u[t], t);
                        lx[t] = l[t].block(0, 0, X, 1);
                        lu[t] = l[t].block(X, 0, U, 1);

                        L[t] = hessian(x[t], u[t], t, l[t]);
                        lxx[t] = L[t].block(0, 0, X, X);
                        luu[t] = L[t].block(X, X, U, U);
                        lux[t] = L[t].block(X, 0, U, X);
                    }
                }

                trajectoryChanged = false;
            }

            // backward pass 
        
            // initialize value components with costs of final state
            Vx = lx[T-1];
            Vxx = lxx[T-1];

            for (int t = T-1; t > -1; t -= 1) {
                
                // update cost-go-go components 
                Qx = lx[t] + fx[t].transpose() * Vx; 
                Qu = lu[t] + fu[t].transpose() * Vx; 
                Qxx = lxx[t] + fx[t].transpose() * Vxx * fx[t];
                Quu = luu[t] + fu[t].transpose() * Vxx * fu[t];
                Qux = lux[t] + fu[t].transpose() * Vxx * fx[t];

                // regularized cost-to-go components
                mat<X, X> modVxx = Vxx.array() + mat<X, X>::Identity().array() * mu;
                mat<U, U> modQuu = luu[t] + fu[t].transpose() * modVxx * fu[t];
                mat<U, X> modQux = lux[t] + fu[t].transpose() * modVxx * fx[t];

                // compute control components
                mat<U, U> H = -modQuu.inverse();

                k[t] = H * Qu;
                K[t] = H * modQux;

                vec<U> c = u[t] + k[t];
                // apply control limits
                for (size_t d = 0; d < U; d += 1) {
                    if (c[d] > uMax[d]) {
                        k[t][d] = uMax[d] - u[t][d];
                        K[t].row(d) = mat<1, X>::Zero();
                    }
                    if (c[d] < uMin[d]) {
                        k[t][d] = uMin[d] - u[t][d];
                        K[t].row(d) = mat<1, X>::Zero();
                    }
                }

                // update value components, using improved value update
                Vx = Qx + K[t].transpose() * Quu * k[t] 
                    + K[t].transpose() * Qu 
                    + Qux.transpose() * k[t];
                Vxx = Qxx + K[t].transpose() * Quu * K[t] 
                    + K[t].transpose() * Qux
                    + Qux.transpose() * K[t];
                Vxx = 0.5 * (Vxx.transpose() + Vxx);
            }
            
            // simple parallel linesearch, prevents overshooting
        
            double alphaSteps[6] = {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125};

            #pragma omp parallel for
            for (int i = 0; i < 6; ++i) {

                double alpha = alphaSteps[i];

                std::vector<vec<X>> states(T);
                states[0] = x[0];
                std::vector<vec<U>> actions(T);

                double newCosts = 0.0;

                for (size_t t = 0; t < T-1; t += 1) {
                    actions[t] = u[t] + K[t] * (states[t] - x[t]) + k[t] * alpha; 
                    states[t+1] = dynamics(states[t], actions[t], t);
                    newCosts += costs(states[t], actions[t], t);
                }
                actions[T-1] = u[T-1] + K[T-1] * (states[T-1] - x[T-1]) + k[T-1] * alpha; 
                newCosts += costs(states[T-1], actions[T-1], T-1);

                #pragma omp critical
                if (newCosts < trajCosts) {
                    prev_x = x;
                    prev_u = u;
                    x = states;
                    u = actions;
                    trajCosts = newCosts;
                    trajectoryChanged = true;
                    improved = true;
                }
            }

            if (trajectoryChanged) {
                // linesearch found alpha that allows improvement
                
                // very stupid, but idk how to do it elegantly in eigen
                double gradNorm = 0.0;
                for (size_t i = 0; i < u.size(); ++i) {
                    gradNorm += (k[i].array().abs() / u[i].array().abs()).mean();
                }
                gradNorm /= u.size();
                
                // check if gradient is small enougth
                if (s > 0 && gradNorm < 10e-6 && mu < 10e-5) {
                    iterations = s + 1;
                    break;
                }

                // decrease regularization
            
                muDelta = std::min(1.0/minMuDelta, muDelta/minMuDelta);

                if (mu * muDelta > minMu) {
                    mu = mu * muDelta;
                } else if (mu * muDelta <= minMu) {
                    mu = 0.0;
                }
            } else {
                // no improvement possible for any alpha
                
                muDelta = std::max(minMuDelta, muDelta * minMuDelta);
                mu = std::max(minMu, mu * muDelta);

                if (mu > 10e7) {
                    iterations = s + 1;
                    break;
                }
            }

            iterations = s + 1;
        }
        
        // measure time for statistics

        std::chrono::steady_clock::time_point end
            = std::chrono::steady_clock::now();

        elapsedUpdateTime = std::chrono::duration_cast<
            std::chrono::milliseconds>(end - begin).count();

        return improved;
    }
};

#endif
