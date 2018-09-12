#include <math.h> 
#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Local function
static double wrap_rads(double r);
// KF class
KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {

    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
    std::cout << "x_(prior) =" << x_ << std::endl;

}

void KalmanFilter::Update(const VectorXd &z) {

    VectorXd y = z - H_ * x_;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;

    MatrixXd I = MatrixXd::Identity(4, 4);

    //new state
    x_ = x_ + (K * y);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    // map state to measurement space
    VectorXd z_pred = VectorXd(3);
    z_pred(0) = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
    z_pred(1) = atan2(x_(1),x_(0));
    if (z_pred(0) <1e-4)
    {
        return;
    }
    std::cout << "z =" << z << std::endl;
    z_pred(2) = (x_(0)*x_(2) + x_(1)*x_(3)) / z_pred(0);
    std::cout << "z_pred =" << z_pred << std::endl;
    VectorXd y = z - z_pred;
    std::cout << "y(before) =" << y << std::endl;
    y(1) = wrap_rads(y(1)); // wrap angle between [PI ,- PI)
    std::cout << "y(after) =" <<y << std::endl;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;

    MatrixXd I = MatrixXd::Identity(4, 4);

    //new state
    x_ = x_ + (K * y);
    P_ = (I - K * H_) * P_;
}

static double wrap_rads(double r)
{
    while ( r > M_PI ) {
        r -= 2 * M_PI;
        std::cout << "too big"<< std::endl;

    }

    while ( r <= -M_PI ) {
        r += 2 * M_PI;
        std::cout << "too small"<< std::endl;
    }

    return r;
}
