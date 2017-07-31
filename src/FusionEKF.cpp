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
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  noise_ax_ = 9;
  noise_ay_ = 9;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
	
	// Initialization stage
  if (!is_initialized_) {
		cout << "EKF: " << endl;

		//state covariance matrix P
		ekf_.P_ = MatrixXd(4, 4);
		ekf_.P_ << 1, 0, 0, 0,
				 0, 1, 0, 0,
				 0, 0, 1000, 0,
				 0, 0, 0, 1000;

		//the initial transition matrix F_
		ekf_.F_ = MatrixXd(4, 4);
		ekf_.F_ << 1, 0, 1, 0,
				 0, 1, 0, 1,
				 0, 0, 1, 0,
				 0, 0, 0, 1;
		
		ekf_.H_ = H_laser_;

		ekf_.x_ = VectorXd(4);

		measurement_pack.sensor_type_ == MeasurementPackage::RADAR ?
			ekf_.x_ << FusionEKF::ConvertMeasurementToCartesian(measurement_pack.raw_measurements_, true)
		:
			ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
		                    
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;

    return;
  }

	// Kalman Filter Prediction stage

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = measurement_pack.timestamp_;
	
  ekf_.F_(0,2) = dt;
	ekf_.F_(1,3) = dt;

	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.Q_ << noise_ax_ * pow(dt, 4)/4, 0, noise_ax_ * pow(dt, 3)/2, 0,
						 0, noise_ay_ * pow(dt, 4)/4, 0, noise_ay_ * pow(dt, 3)/2,
						 noise_ax_ * pow(dt,3)/2, 0, noise_ax_ * pow(dt, 2), 0,
						 0, noise_ay_ * pow(dt, 3)/2, 0, noise_ay_ * pow(dt, 2);


  ekf_.Predict();

	// Kalman Filter update stage

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
		ekf_.R_ = R_radar_;
		ekf_.UpdateEKF(measurement_pack.raw_measurements_);
	}  
	else {
		ekf_.H_ = H_laser_;
		ekf_.R_ = R_laser_;
		ekf_.Update(measurement_pack.raw_measurements_);
	}

  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

// Sets initial state for radar measreurements. Intial velocity is assumed to be zero.
VectorXd FusionEKF::ConvertMeasurementToCartesian(const VectorXd &measurements, bool is_inital) {
  float rho = measurements(0);
  float phi = measurements(1);
	float rho_dot = measurements(2);

	float px = rho*cos(phi);
	float py = rho*sin(phi);
	float vx = is_inital ? 0 : rho_dot*cos(phi);
	float vy = is_inital ? 0 : rho_dot*sin(phi);
  
	VectorXd cartesian_measurements = VectorXd(4);
	cartesian_measurements << px, py, vx, vy;

  return cartesian_measurements;
}



