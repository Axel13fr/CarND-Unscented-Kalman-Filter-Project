#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

constexpr int UKF::n_x_;
constexpr int UKF::n_aug_;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() :
    is_initialized_(false),
    use_laser_(true),    // if this is false, laser measurements will be ignored (except during init)
    use_radar_(true),      // if this is false, radar measurements will be ignored (except during init)
    x_(VectorXd::Zero(5)),// initial state vector
    P_(MatrixXd::Zero(5, 5)),// initial covariance matrix
    Xsig_pred_(MatrixXd::Zero(n_x_, 2 * n_aug_ + 1)),
    previous_timestamp_(0),
    weights_(VectorXd(2*n_aug_+1))
{
    // Pre init weights once for all compuation
    double weight_0 = lambda_/(lambda_+n_aug_);
    weights_(0) = weight_0;
    for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
        double weight = 0.5/(n_aug_+lambda_);
        weights_(i) = weight;
    }

    // CRTV model
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3; // ~0 to 100kph in 10s

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.4; // pi/8


    // Sensor model
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

}

UKF::~UKF() {}


double UKF::GetDeltaT(MeasurementPackage meas_package)
{
    constexpr double MICROSECS_TO_SECS = 1/1000000.0;
    return (meas_package.timestamp_us_ - previous_timestamp_)*MICROSECS_TO_SECS;
}

bool UKF::IsMeasurementUsed(MeasurementPackage meas_package)
{
    return ((meas_package.sensor_type_ == MeasurementPackage::RADAR and use_radar_)
            or
            (meas_package.sensor_type_ == MeasurementPackage::LASER and use_laser_));
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

    if (!is_initialized_) {
        /**
          * Initialize the state x_ with the first measurement.
          * Create the covariance matrix.
          * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */
        // Initial cov set to 3m for position and 2m/s for speed
        P_ << 1,0,0,0,0,
                0,1,0,0,0,
                0,0,1,0,0,
                0,0,0,0.5,0, // orientation grad^2
                0,0,0,0,0.5;   // yaw rate
        cout << P_ << endl;

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            /**
          Convert radar from polar to cartesian coordinates and initialize state.
          */
            x_ = Tools::CalculatePosFromRadar(meas_package.raw_measurements_);
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            /**
      Initialize state.
      */
            x_ <<  meas_package.raw_measurements_(0),
                    meas_package.raw_measurements_(1),
                    0.2,0.1,0.0;
        }

        previous_timestamp_ = meas_package.timestamp_us_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
   *  Prediction
   ****************************************************************************/

    /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
    if(IsMeasurementUsed(meas_package)){
        auto delta_t = GetDeltaT(meas_package);
        Prediction(delta_t);
    }else{
        // measurement type ignored
    }


    /*****************************************************************************
   *  Update
   ****************************************************************************/

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR and use_radar_) {
        // Radar updates
        UpdateRadar(meas_package);
        previous_timestamp_ = meas_package.timestamp_us_;

    } else if(meas_package.sensor_type_ == MeasurementPackage::LASER and use_laser_){
        // Laser updates
        UpdateLidar(meas_package);
        previous_timestamp_ = meas_package.timestamp_us_;
    } else{
        // wrong measurement !
        if(meas_package.sensor_type_ != MeasurementPackage::RADAR
                and
           meas_package.sensor_type_ != MeasurementPackage::LASER){

            cout << "No measurement type !";
        }

    }

    // print the output
    cout << "Fused pos x_ = " << x_(0) << " y_ = " << x_(1)  << " V = " << x_(2) << " Yaw = " << x_(3) << endl;
    cout << "sig_x = " << P_(0,0) << " sig_y = " << P_(1,1) << "sig_V = " << P_(2,2) <<  endl;
}

MatrixXd UKF::GenerateSigmaPoints() {

    //create sigma point matrix
    MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

    //calculate square root of P
    MatrixXd A = P_.llt().matrixL();

    //set first column of sigma point matrix
    Xsig.col(0)  = x_;

    //set remaining sigma points
    for (int i = 0; i < n_x_; i++)
    {
        Xsig.col(i+1)     = x_ + sqrt(lambda_+n_x_) * A.col(i);
        Xsig.col(i+1+n_x_) = x_ - sqrt(lambda_+n_x_) * A.col(i);
    }

    //print result
    //std::cout << "Xsig = " << std::endl << Xsig << std::endl;

    //write result
    return Xsig;

}

MatrixXd UKF::AugmentedSigmaPoints() {

    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_*std_a_;
    P_aug(6,6) = std_yawdd_*std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i< n_aug_; i++)
    {
        Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
        // Normalize the angle even for sigma points !
        //Xsig_aug.col(i+1)(3)= Tools::NormalizeAngle(Xsig_aug.col(i+1)(3));
        //Xsig_aug.col(i+1)(4)= Tools::NormalizeAngle(Xsig_aug.col(i+1)(4));
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
        //Xsig_aug.col(i+1+n_aug)(3)= Tools::NormalizeAngle(Xsig_aug.col(i+1+n_aug)(3));
        //Xsig_aug.col(i+1+n_aug)(4)= Tools::NormalizeAngle(Xsig_aug.col(i+1+n_aug)(4));
    }

    //print result
    //std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

    //write result
    return Xsig_aug;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

    // Generate Sigma points
    MatrixXd Xsig_aug = AugmentedSigmaPoints();

    // Predict Sigma points
    SigmaPointPrediction(Xsig_aug,delta_t);

    // Predict mu and P
    PredictMeanAndCovariance();

    //MatrixXd Xsig_aug = augmentSigmaPoints();
    //predictSigmaPoints(delta_t, Xsig_aug);
    //predictMeanAndCovariance();

}

void UKF::PredictMeanAndCovariance() {

    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ = x_+ weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3) > M_PI)
          x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI)
          x_diff(3) += 2. * M_PI;

        //x_(3)= Tools::NormalizeAngle(x_(3));

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }

    //print result
    //std::cout << "Predicted state" << std::endl;
    //std::cout << x << std::endl;
    //std::cout << "Predicted covariance matrix" << std::endl;
    //std::cout << P << std::endl;

}

void UKF::SigmaPointPrediction(const MatrixXd& Xsig_aug, const double delta_t) {

    //predict sigma points
    for (int i = 0; i< 2*n_aug_+1; i++)
    {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }

    //print result
    //std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::PredictLidarMeasurement(MatrixXd& ZSig_out,VectorXd& z_out, MatrixXd& S_out)
{
    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 2;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);

        // measurement model
        Zsig(Tools::X,i) = p_x;
        Zsig(Tools::Y,i) = p_y;

    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_laspx_*std_laspx_,0,
            0, std_laspy_*std_laspy_;
    S = S + R;

    //print result
    //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
    //std::cout << "S: " << std::endl << S << std::endl;

    //write result
    ZSig_out = Zsig;
    z_out = z_pred;
    S_out = S;


}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**

      Complete this function! Use lidar data to update the belief about the object's
      position. Modify the state vector, x_, and covariance, P_.

      You'll also need to calculate the lidar NIS.
      */
    // Predict measurement
    MatrixXd Zsig;
    VectorXd z_pred;
    MatrixXd S_pred;
    PredictLidarMeasurement(Zsig,z_pred,S_pred);

    // Update
    VectorXd Z_meas = VectorXd(2); // X and Y measurements
    Z_meas <<  meas_package.raw_measurements_(Tools::X),
            meas_package.raw_measurements_(Tools::Y);

    UpdateStateLidar(Zsig,z_pred,S_pred,Z_meas);

}

void UKF::PredictRadarMeasurement(MatrixXd& ZSig_out,VectorXd& z_out, MatrixXd& S_out) {

    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        Zsig(Tools::RO,i) = sqrt(p_x*p_x + p_y*p_y);                           //r
        Zsig(Tools::THETA,i) = atan2(p_y,p_x);                                 //phi
        Zsig(Tools::RO_DOT,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);  //r_dot

    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        z_diff(1) = Tools::NormalizeAngle(z_diff(1));

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0,std_radrd_*std_radrd_;
    S = S + R;

    //print result
    //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
    //std::cout << "S: " << std::endl << S << std::endl;

    //write result
    ZSig_out = Zsig;
    z_out = z_pred;
    S_out = S;
}
/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**

  Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
    // Predict measurement
    MatrixXd Zsig;
    VectorXd z_pred;
    MatrixXd S_pred;
    PredictRadarMeasurement(Zsig,z_pred,S_pred);

    // Update
    VectorXd Z_meas = meas_package.raw_measurements_;
    UpdateStateRadar(Zsig,z_pred,S_pred,Z_meas);

    // Compute Radar NIS

}

void UKF::UpdateStateRadar(const MatrixXd& Zsig,const VectorXd& z_pred, const MatrixXd &S_pred,const VectorXd& z_meas) {

    //set measurement dimension, radar can measure r, phi, and r_dot
    int n_z = 3;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        z_diff(1) = Tools::NormalizeAngle(z_diff(1));

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        x_diff(3) = Tools::NormalizeAngle(x_diff(3));

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S_pred.inverse();

    //residual
    VectorXd z_diff = z_meas - z_pred;

    //angle normalization
    z_diff(1) = Tools::NormalizeAngle(z_diff(1));

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S_pred*K.transpose();

    //print result
    //std::cout << "Updated state x: " << std::endl << x << std::endl;
    //std::cout << "Updated state covariance P: " << std::endl << P << std::endl;


}

void UKF::UpdateStateLidar(const MatrixXd& Zsig,const VectorXd& z_pred, const MatrixXd &S_pred,const VectorXd& z_meas) {

    //set measurement dimension, lidar is x and y position measurement
    int n_z = 2;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S_pred.inverse();

    //residual
    VectorXd z_diff = z_meas - z_pred;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S_pred*K.transpose();

    //print result
    //std::cout << "Updated state x: " << std::endl << x << std::endl;
    //std::cout << "Updated state covariance P: " << std::endl << P << std::endl;

}

