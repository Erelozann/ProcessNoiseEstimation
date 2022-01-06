function proc_noise = kf_proc_noise_est(OutputPrediction, ...
                                    OutputPredictionCov, ...
                                    Measurement,...
                                    FilterParameters)
                           
H = FilterParameters.H;
Q = FilterParameters.Q;
T = FilterParameters.T;

G = [T^2/2*eye(2);
    T*eye(2)];

Q = G*Q*G';
proc_noise = Q*H'*inv(OutputPredictionCov)*(Measurement-OutputPrediction);