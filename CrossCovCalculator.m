function CrossCov = CrossCovCalculator(Local_Trackers,PrevCrossCov)


T = Local_Trackers(1).Parameters.T;
H_i = Local_Trackers(1).Parameters.H;
Q = Local_Trackers(1).Parameters.Q;
K_i = Local_Trackers(1).KalmanGain;

H_j = Local_Trackers(2).Parameters.H;
K_j = Local_Trackers(2).KalmanGain;


% Dynamic Model
% Constant velocity model

F = [eye(2) T*eye(2);
    zeros(2) eye(2)];

G = [T^2/2*eye(2);
    T*eye(2)];


CrossCov = (eye(4)-K_i*H_i)*(F*PrevCrossCov*F'+G*Q*G')*(eye(4)-K_j*H_j)';
