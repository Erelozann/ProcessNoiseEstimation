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

sigma_a = 20; %Acceleration Noise Standard Deviation
Q = (sigma_a)^2*eye(2); %Process Noise Covariance

% Measurement Model 

H = [eye(2) zeros(2)]; %Measurement Matrix

sigma_v = 20; %Measurement Noise Standard Deviation
R = (sigma_v)^2*eye(2); %Measurement Noise Covariance
% Simulation Parameters

Step_Num = 100;
Smooth_Param = 5;

x0_bar = [5e3 5e3 25 25]';
P0_bar = diag((x0_bar/10).^2);

x_k = zeros(4,Step_Num);
y_k = zeros(2,Step_Num);
w_k = zeros(4,Step_Num);
chol_R = chol(R)';
chol_P0_bar = chol(P0_bar)';

 x_k(:,1) = x0_bar + chol_P0_bar*randn(4,1); % True Target Data Generation
 y_k(:,1) = H*x_k(:,1)+ chol_R*randn(2,1);
 
    for k=2:Step_Num
        w_k(:,k) = G*(sigma_a*randn(2,1));
        x_k(:,k) = F*x_k(:,k-1) + w_k(:,k); % True Target Data Generation
        y_k(:,k) = H*x_k(:,k)+ chol_R*randn(2,1);
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

Smoother.StateEstimate = zeros(4,Smooth_Param);
Smoother.StateEstimateCov = zeros(4,4,Smooth_Param);

est_proc_noise_err = zeros(4,Step_Num);
true_proc_noise_err = zeros(4,Step_Num);

true_err = zeros(4,Step_Num);
true_err(:,1) = x_k(:,1)-Tracker.StateEstimate(:,1);
    
StateToProcessTransitionMatrix = [eye(4) -F];
    for k=2:Step_Num
        
            
            % Prediction:
            
            [Tracker.StatePrediction(:,k-1), ...
                Tracker.StatePredictionCov(:,:,k-1), ...
                OutputPrediction, ...
                OutputPredictionCov, ...
                KalmanGain] = kf_pre(Tracker.StateEstimate(:,k-1),...
                Tracker.StateEstimateCov(:,:,k-1),...
                Tracker.Parameters);
            
            % Estimation:
            [...
                Tracker.StateEstimate(:,k), ...
                Tracker.StateEstimateCov(:,:,k),~] = kf_est(...
                Tracker.StatePrediction(:,k-1),...
                Tracker.StatePredictionCov(:,:,k-1),...
                OutputPrediction, ...
                OutputPredictionCov, ...
                KalmanGain,...
                y_k(:,k));
                
            
            if k>=Smooth_Param
                
                C = Tracker.StateEstimateCov(:,:,k-1)*F'*inv(Tracker.StatePredictionCov(:,:,k-1));
                
                Smoother.StateEstimate(:,Smooth_Param) = Tracker.StateEstimate(:,k);
                Smoother.StateEstimateCov(:,:,Smooth_Param) = Tracker.StateEstimateCov(:,:,k);
                
                for i=Smooth_Param:-1:2
                    
                    Smoother.StateEstimate(:,i-1) = Tracker.StateEstimate(:,k-1) + C*(Smoother.StateEstimate(:,i)-Tracker.StatePrediction(:,k-1));
                    Smoother.StateEstimateCov(:,:,i-1) = Tracker.StateEstimateCov(:,:,k-1) + C*(Smoother.StateEstimateCov(:,:,i)-Tracker.StatePredictionCov(:,:,k-1))*C';
                    
                end
                
                SmoothedStatesCovarianceMatrix = [Smoother.StateEstimateCov(:,:,2) Smoother.StateEstimateCov(:,:,2)*C';
                                                  C*Smoother.StateEstimateCov(:,:,2) Smoother.StateEstimateCov(:,:,1)];
                
                Tracker.ProcessNoise(:,k-Smooth_Param+2) = Smoother.StateEstimate(:,2)-F*Smoother.StateEstimate(:,1);
                Tracker.ProcessNoiseCov(:,:,k-Smooth_Param+2) = StateToProcessTransitionMatrix*SmoothedStatesCovarianceMatrix*StateToProcessTransitionMatrix';
            end
            

            
%             mx_a = (eye(4)-KalmanGain*Tracker.Parameters.H);
%             est_proc_noise_err(:,k) = mx_a*F*est_proc_noise_err(:,k-1)+mx_a*Tracker.ProcessNoise(:,k);
%             true_proc_noise_err(:,k) = mx_a*F*true_proc_noise_err(:,k-1)+mx_a*w_k(:,k);
%             true_err(:,k) = x_k(:,k)-Tracker.StateEstimate(:,k);     

    end
    
    for i=3:Smooth_Param
        
        SmoothedStatesCovarianceMatrix = [Smoother.StateEstimateCov(:,:,i) Smoother.StateEstimateCov(:,:,i)*C';
                                          C*Smoother.StateEstimateCov(:,:,i) Smoother.StateEstimateCov(:,:,i-1)];
        
        Tracker.ProcessNoise(:,Step_Num-Smooth_Param+i) = Smoother.StateEstimate(:,i)-F*Smoother.StateEstimate(:,i-1);
        Tracker.ProcessNoiseCov(:,:,Step_Num-Smooth_Param+i) = StateToProcessTransitionMatrix*SmoothedStatesCovarianceMatrix*StateToProcessTransitionMatrix';
    end
    
    ProcessNoiseStd = sqrt(diag(G*Q*G'));
    
    plot(Tracker.ProcessNoise(3,2:end))
    hold on
    plot(w_k(3,2:end))
    legend({'Estimated ','True'}, 'fontsize', 10);
    ylabel(' Process Noise X (m/s^2)', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;
         
    hold off
    figure;
    plot(sqrt(squeeze(Tracker.ProcessNoiseCov(3,3,2:end)))')
    hold on
    plot(repmat(ProcessNoiseStd(3),1,Step_Num-1))
    legend({'Estimated ','True'}, 'fontsize', 10);
    ylabel(' Process Noise Std X (m/s^2)', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;
    
    hold off
    figure;
    plot(Tracker.ProcessNoise(4,2:end))
    hold on
    plot(w_k(4,2:end))
    legend({'Estimated','True'}, 'fontsize', 10);
    ylabel('Process Noise Y (m/s^2)', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;
    
    hold off
    figure;
    plot(sqrt(squeeze(Tracker.ProcessNoiseCov(4,4,2:end)))')
    hold on
    plot(repmat(ProcessNoiseStd(4),Step_Num-1))
    legend({'Estimated ','True'}, 'fontsize', 10);
    ylabel(' Process Noise Std Y (m/s^2)', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;
     

    
    
%     figure;
%     plot(est_proc_noise_err(3,:))
%     hold on
%     plot(true_proc_noise_err(3,:))
%     legend({'Estimated','True'}, 'fontsize', 10);
%     ylabel('Error in Vel X (m/s)', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;
%     
%     
%     figure;
%     plot(est_proc_noise_err(4,:))
%     hold on
%     plot(true_proc_noise_err(4,:))
%     legend({'Estimated State Error','True State Error'}, 'fontsize', 10);
%     ylabel('Error in Vel Y(m/s)', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;
% 
