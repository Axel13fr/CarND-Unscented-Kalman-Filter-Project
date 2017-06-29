#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

    ///* initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    ///* if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    ///* if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    VectorXd x_;

    ///* state covariance matrix
    MatrixXd P_;

    ///* predicted sigma points matrix
    MatrixXd Xsig_pred_;

    ///* time when the state is true, in us
    long long time_us_;

    ///* previous timetamp
    long long previous_timestamp_;

    ///* Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    ///* Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    ///* Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    ///* Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    ///* Radar measurement noise standard deviation radius in m
    double std_radr_;

    ///* Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    ///* Radar measurement noise standard deviation radius change in m/s
    double std_radrd_ ;

    ///* Weights of sigma points
    VectorXd weights_;

    ///* State dimension
    static constexpr int n_x_ = 5;

    ///* Augmented state dimension
    static constexpr int n_aug_ = 7;

    ///* Sigma point spreading parameter
    static constexpr int lambda_ = 3 - n_x_;


    /**
   * Constructor
   */
    UKF();

    /**
   * Destructor
   */
    virtual ~UKF();

    /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
    void Prediction(double delta_t);

    /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
    void UpdateLidar(MeasurementPackage meas_package);

    /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
    void UpdateRadar(MeasurementPackage meas_package);

private:

    /**
     * @brief GetDeltaT
     * @param meas_package
     * @return updated delta time between measurement and last update
     */
    double GetDeltaT(MeasurementPackage meas_package);

    /**
     * @brief GenerateSigmaPoints
     * @return sigma point matrix
     */
    MatrixXd GenerateSigmaPoints();
    /**
     * @brief AugmentedSigmaPoints
     * @return augmented sigma point matrix
     */
    MatrixXd AugmentedSigmaPoints();

     // Predict sigma points for prediction step (from augmented sig pts to predicted sig pts)
    void SigmaPointPrediction(const MatrixXd &Xsig_aug, const double delta_t);

    // Predict state & cov matrix for prediction step
    void PredictMeanAndCovariance();

    // Predict sigma points from state space to measurement space for update
    void PredictRadarMeasurement(MatrixXd &ZSig_out, VectorXd &z_out, MatrixXd &S_out);
    void UpdateStateRadar(const Eigen::MatrixXd &Zsig, const VectorXd &z_pred, const MatrixXd &S_pred,const VectorXd &z_meas);

    /**
     * @brief IsMeasurementUsed
     * @param meas_package
     * @return true if the measurement type shall be processed
     */
    bool IsMeasurementUsed(MeasurementPackage meas_package);

    // Predict sigma points from state space to measurement space for update
    void PredictLidarMeasurement(MatrixXd& ZSig_out,VectorXd& z_out, MatrixXd& S_out);
    void UpdateStateLidar(const MatrixXd &Zsig, const VectorXd &z_pred, const MatrixXd &S_pred, const VectorXd &z_meas);

};

#endif /* UKF_H */
