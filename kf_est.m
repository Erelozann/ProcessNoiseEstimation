
function [...
    StateEstimate, ...
    StateEstimateCov,...
    L] = kf_est(StatePrediction,...
                               StatePredictionCov,...
                               OutputPrediction, ...
                               OutputPredictionCov, ...
                               KalmanGain,...
                               Measurement)
                           
    Innovation = Measurement-OutputPrediction;
    K = KalmanGain;
    S = OutputPredictionCov;
    olcumBoyutu = size(Innovation,1);
    
    % State Update
    x = StatePrediction + K*Innovation;
    
    % Covariance Update
    P = StatePredictionCov - K*S*K';

    % Simetrikligi Koru:
    P = (P + P')/2;
    
    %Likelihood Update:
    
    mahalanobisUzaklik = Innovation'/S*Innovation;
    L = exp(-mahalanobisUzaklik/2)/sqrt(((2*pi)^olcumBoyutu)*det(S));
    
    % Output
    StateEstimate = x;
    StateEstimateCov = P;

end