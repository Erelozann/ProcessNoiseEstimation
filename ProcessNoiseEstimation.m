clear 
close all
%% Initialization
% True Target Model

T = 1; %Sample Interval

% State Transition Model

F = [eye(2)  T*eye(2); %State Transition Matrix
     zeros(2) eye(2)];

G = [T^2/2*eye(2); %Process Noise Gain Matrix
     T*eye(2)];

sigma_a = 2; %Acceleration Noise Standard Deviation
Q = (sigma_a)^2*eye(2); %Process Noise Covariance

% Measurement Model 

H = [eye(2) zeros(2)]; %Measurement Matrix

sigma_v = 20; %Measurement Noise Standard Deviation
R = (sigma_v)^2*eye(2); %Measurement Noise Covariance
% Simulation Parameters

Step_Num = 100;

x0_bar = [5e3 5e3 25 25]';
P0_bar = diag((x0_bar/10).^2);

x_k = zeros(4,Step_Num);
y_k = zeros(2,Step_Num);
w_k = zeros(4,Step_Num);
chol_R = chol(R);
chol_P0_bar = chol(P0_bar);

 x_k(:,1) = x0_bar + chol_P0_bar*randn(4,1); % True Target Data Generation
 y_k(:,1) = H*x_k(:,1)+ chol_R*randn(2,1);
 
    for i=2:Step_Num
        w_k(:,i) = G*(sigma_a*randn(2,1));
        x_k(:,i) = F*x_k(:,i-1) + w_k(:,i); % True Target Data Generation
        y_k(:,i) = H*x_k(:,i)+ chol_R*randn(2,1);
    end
              

% Filter Parameters
Filter_Parameters.H=  H;                      
Filter_Parameters.Q = Q;
Filter_Parameters.R = R;
Filter_Parameters.T = T;


% Estimate Initialization

Tracker.Parameters = Filter_Parameters;
Tracker.StateEstimate(:,1) = x0_bar;
Tracker.StateEstimateCov(:,:,1) = P0_bar;
Tracker.ProcessNoise = zeros(4,Step_Num);


est_proc_noise_err = zeros(4,Step_Num);
true_proc_noise_err = zeros(4,Step_Num);

true_err = zeros(4,Step_Num);
true_err(:,1) = x_k(:,1)-Tracker.StateEstimate(:,1);

    
    for i=2:Step_Num
        
            
            % Prediction:
            
            [StatePrediction, ...
                StatePredictionCov, ...
                OutputPrediction, ...
                OutputPredictionCov, ...
                KalmanGain] = kf_pre(Tracker.StateEstimate(:,i-1),...
                Tracker.StateEstimateCov(:,:,i-1),...
                Tracker.Parameters);
            
            % Estimation:
            [...
                Tracker.StateEstimate(:,i), ...
                Tracker.StateEstimateCov(:,:,i),~] = kf_est(...
                StatePrediction,...
                StatePredictionCov,...
                OutputPrediction, ...
                OutputPredictionCov, ...
                KalmanGain,...
                y_k(:,i));
                
             Tracker.ProcessNoise(:,i) = kf_proc_noise_est(OutputPrediction, ...
                                        OutputPredictionCov, ...
                                        y_k(:,i),...
                                        Tracker.Parameters);
            mx_a = (eye(4)-KalmanGain*Tracker.Parameters.H);
            est_proc_noise_err(:,i) = mx_a*F*est_proc_noise_err(:,i-1)+mx_a*Tracker.ProcessNoise(:,i);
            true_proc_noise_err(:,i) = mx_a*F*true_proc_noise_err(:,i-1)+mx_a*w_k(:,i);
            true_err(:,i) = x_k(:,i)-Tracker.StateEstimate(:,i);     

    end
    
    
    plot(Tracker.ProcessNoise(3,:))
    hold on
    plot(w_k(3,:))
    legend({'Estimated ','True'}, 'fontsize', 10);
    ylabel(' Process Noise X (m/s^2)', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;
    
    
    hold off
    figure;
    plot(Tracker.ProcessNoise(4,:))
    hold on
    plot(w_k(4,:))
    hold off
    legend({'Estimated','True'}, 'fontsize', 10);
    ylabel('Process Noise Y (m/s^2)', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;
    
    
    figure;
    plot(est_proc_noise_err(3,:))
    hold on
    plot(true_proc_noise_err(3,:))
    legend({'Estimated','True'}, 'fontsize', 10);
    ylabel('Error in Vel X (m/s)', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;
    
    
    figure;
    plot(est_proc_noise_err(4,:))
    hold on
    plot(true_proc_noise_err(4,:))
    legend({'Estimated State Error','True State Error'}, 'fontsize', 10);
    ylabel('Error in Vel Y(m/s)', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;

