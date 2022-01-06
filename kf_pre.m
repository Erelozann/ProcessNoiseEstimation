function [StatePrediction,...
          StatePredictionCov,...
          OutputPrediction,...
          OutputPredictionCov,...
          KalmanGain] = kf_pre(StateEstimate,...
                               StateEstimateCov,...
                               FilterParameters)

T = FilterParameters.T;
H = FilterParameters.H;
R = FilterParameters.R;
Q = FilterParameters.Q;

x = StateEstimate;
P = StateEstimateCov;

% Dynamic Model
% Constant velocity model

F = [eye(2) T*eye(2);
    zeros(2) eye(2)];

G = [T^2/2*eye(2);
    T*eye(2)];

    
    % ------ Prediction -- Time Update--------------------------------------
    % State Prediction
    x_1 = F*x;
    P_1 = F*P*F' + G*Q*G';  %StatePredictionsCov

    % Ouput Prediction
    z_1 = H*x_1;

    % Output Prediction Covariance
    S = H*P_1*H' + R;

    % Kalman Gain
    K = P_1*H'/S;

    % Output
    StatePrediction = x_1;
    StatePredictionCov = P_1;
    OutputPrediction = z_1;
    OutputPredictionCov = S;
    KalmanGain = K;
end
