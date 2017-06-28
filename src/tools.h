#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  enum state_E{
      X = 0,
      Y,
      V,
      YAW,
      YAWRATE,

      STATE_SIZE
  };
  enum rad_mes_E{
    RO = 0,
    THETA,
    RO_DOT,

    RAD_MES_SIZE
  };

  /**
  * A helper method to calculate RMSE.
  */
  static VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  static Eigen::VectorXd CalculatePosFromRadar(const Eigen::VectorXd &radar_mes);
  static Eigen::VectorXd TransformToRadarFromState(const Eigen::VectorXd &state);

  static double NormalizeAngle(const double &angle);
};

#endif /* TOOLS_H_ */
