#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  noise_a_mpsps_ = 9;
     
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  count_ += 1;
  cout << "***count_: " <<count_ << endl;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);

    ekf_.x_ = VectorXd(4);
    //state covariance matrix P
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1000, 0,
              0, 0, 0, 1000;
    //state transition matrix P
    ekf_.F_ = MatrixXd::Identity(4, 4);          

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      ekf_.x_ << measurement_pack.raw_measurements_(0)*cos(measurement_pack.raw_measurements_(1)), 
                measurement_pack.raw_measurements_(0)*sin(measurement_pack.raw_measurements_(1)),
                0,0; // can't resolve to correct vx,vy
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_(0), 
                measurement_pack.raw_measurements_(1),
                0,
                0; // lidar does'nt give us any velocity measurement
    }
    // done initializing, no need to predict or update
    is_initialized_ = true;
    cout << "is_initialized_ using " << measurement_pack.sensor_type_ << endl;
    previous_timestamp_ = measurement_pack.timestamp_;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  //compute the time elapsed between the current and previous measurements
  float dt_s = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  if (dt_s < 0)
  {
    return;
  }
  previous_timestamp_ = measurement_pack.timestamp_;
  cout << "Processing measurement at " << measurement_pack.timestamp_ << endl;
  cout << "dt = " << dt_s<< endl;
  float dt_2 = dt_s * dt_s;
  float dt_3 = dt_2 * dt_s;
  float dt_4 = dt_3 * dt_s;

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt_s;
  ekf_.F_(1, 3) = dt_s;

  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ <<  dt_4/4*noise_a_mpsps_, 0, dt_3/2*noise_a_mpsps_, 0,
         0, dt_4/4*noise_a_mpsps_, 0, dt_3/2*noise_a_mpsps_,
         dt_3/2*noise_a_mpsps_, 0, dt_2*noise_a_mpsps_, 0,
         0, dt_3/2*noise_a_mpsps_, 0, dt_2*noise_a_mpsps_;

  ekf_.Predict();
  cout << "Prediction" << endl;
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    cout << "Update EKF, radar measurement" << endl;
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    VectorXd z = measurement_pack.raw_measurements_;
    ekf_.UpdateEKF(z);
  } 
  else
  { 
    cout << "Update EKF, laser measurement" << endl;
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    VectorXd z = measurement_pack.raw_measurements_;
    ekf_.Update(z);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
