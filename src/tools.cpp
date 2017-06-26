#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
    auto rmse = VectorXd(4);
    rmse << 0,0,0,0;
    if(estimations.size() != ground_truth.size() or estimations.size() == 0){
        std::cout << "size error in RMSE calc " << std::endl;
        return rmse;
    }

    for(unsigned int i=0; i < estimations.size() ; i++){
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    rmse = rmse/estimations.size();
    rmse = rmse.array().sqrt();

    return rmse;
}

Eigen::VectorXd Tools::CalculatePosFromRadar(const Eigen::VectorXd &radar_mes)
{
    auto ro = radar_mes(RO);
    auto theta = radar_mes(THETA);
    auto state = VectorXd(STATE_SIZE);
    // x,y, V, Yaw, YawRate
    state << ro*std::cos(theta) , ro*std::sin(theta) ,
            0.2 , 0.1, 0;

    return state;
}

Eigen::VectorXd Tools::TransformToRadarFromState(const Eigen::VectorXd &state)
{
    // state = x,y, Vx, Vy
    auto ro = sqrt(state(X)*state(X) + state(Y)*state(Y));
    auto theta = atan2(state(Y),state(X));
    assert(theta <= M_PI and theta >= -M_PI);

    auto c1 = state(X)*state(Vx) + state(Y)*state(Vy);
    assert(fabs(c1) > 0.0001);
    assert(fabs(ro) > 0.0001);
    auto ro_dot = c1 / ro;

    auto rad_mes = VectorXd(RAD_MES_SIZE);
    rad_mes << ro , theta, ro_dot;

    return rad_mes;
}


double Tools::NormalizeAngle(const double& angle){
    auto ret_angle = angle;

    while (ret_angle> M_PI) ret_angle-=2.*M_PI;
    while (ret_angle<-M_PI) ret_angle+=2.*M_PI;

    return ret_angle;

}
