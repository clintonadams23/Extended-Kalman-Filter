#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

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
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	VectorXd z_pred = KalmanFilter::ConvertToPolar(x_);
	VectorXd y = z - z_pred;

	if (y(1) < -M_PI || y(1) > M_PI) 
		y(1) = KalmanFilter::NormalizeAngle(y(1));
	
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

VectorXd KalmanFilter::ConvertToPolar(const VectorXd &cartesian_vector) {	
	float px = cartesian_vector(0);
	float py = cartesian_vector(1);
	float vx = cartesian_vector(2);
	float vy = cartesian_vector(3);

	float rho = sqrt(pow(px, 2) + pow(py, 2));

	if (fabs(rho) < 0.001)
		rho = 0.001;

	if (fabs(px) < 0.001)
		px = 0.001;

	VectorXd h_function(3);
	h_function << rho,
								atan2(py, px),
								(px*vx + py*vy) / rho;

	return h_function;
}

float KalmanFilter::NormalizeAngle(float phi) {
	while (phi < -M_PI)
		phi += 2 * M_PI;
	while (phi > M_PI)
		phi -= 2 * M_PI;

	return phi;
}