#include <iostream>
#include "tools.h"
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	if (estimations.size() == 0) {
	    cout << "Empty estimation vector" << endl;
	    return rmse;
	}

	if (estimations.size() != ground_truth.size()) {
	    cout << "Error: expected the dimensions of the estimation and ground truth vectors to be the same" << endl;
	    return rmse;
	}
    VectorXd squaredResiduals(4);
    squaredResiduals << 0,0,0,0;

	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
	    VectorXd currentSquaredResiduals = (estimations[i] - ground_truth[i]).array().pow(2);
      squaredResiduals += currentSquaredResiduals;
	}

    VectorXd mean = squaredResiduals/estimations.size();
    rmse = mean.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3,4);

	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);
    
	float squared_sum = pow(px, 2) + pow(py, 2);
	float hypotenuse = sqrt(squared_sum);

	if (fabs(squared_sum) < 0.0001) {
	    cout << "Divide by zero error" << endl;
	    return Hj;
	}
	
	//compute the Jacobian matrix
	Hj << px/hypotenuse												           , py/hypotenuse                                    , 0            , 0              ,
	      -py/squared_sum													       , px/squared_sum                                   , 0            , 0              ,
	      py * (vx*py - vy*px)/ (squared_sum * hypotenuse), px * (vy*px - vx*py)/ (squared_sum * hypotenuse), px/hypotenuse, py/ hypotenuse ;

	return Hj;
}


