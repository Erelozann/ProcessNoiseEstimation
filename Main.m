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
MC_Num = 100; %Monte Carlo Number

x0_bar = [5e3 5e3 25 25]';
P0_bar = diag((x0_bar/10).^2);


x_k = zeros(4,Step_Num,MC_Num);
y_k_1 = zeros(2,Step_Num,MC_Num);
y_k_2 = zeros(2,Step_Num,MC_Num);

chol_R = chol(R);
chol_P0_bar = chol(P0_bar);

for MC = 1:MC_Num
 x_k(:,1,MC) = x0_bar + chol_P0_bar*randn(4,1); % True Target Data Generation
 y_k_1(:,1) = H*x_k(:,1,MC)+ chol_R*randn(2,1);
 y_k_2(:,1) = H*x_k(:,1,MC)+ chol_R*randn(2,1);
    for i=2:Step_Num
        x_k(:,i,MC) = F*x_k(:,i-1,MC) + G*(sigma_a*randn(2,1)); % True Target Data Generation
        y_k_1(:,i,MC) = H*x_k(:,i,MC)+ chol_R*randn(2,1);
        y_k_2(:,i,MC) = H*x_k(:,i,MC)+ chol_R*randn(2,1);
    end
end

yk_centralized = [y_k_1;
                  y_k_2];
              


%% Naive Solution

% Filter Parameters
Filter_Parameters(1).H=  H;                      
Filter_Parameters(1).Q = Q;
Filter_Parameters(1).R = R;
Filter_Parameters(1).T = T;

Filter_Parameters(2).H=  H;                     
Filter_Parameters(2).Q = Q;
Filter_Parameters(2).R = R;
Filter_Parameters(2).T = T;

% Estimate Initialization

Local_Trackers(1).Parameters = Filter_Parameters(1);
Local_Trackers(1).StateEstimate(:,1) = x0_bar;
Local_Trackers(1).StateEstimateCov(:,:,1) = P0_bar; 
Local_Trackers(1).KalmanGain = zeros(4,2);

Local_Trackers(2).Parameters = Filter_Parameters(2);
Local_Trackers(2).StateEstimate(:,1) = x0_bar;
Local_Trackers(2).StateEstimateCov(:,:,1) = P0_bar;
Local_Trackers(1).KalmanGain = zeros(4,2);
CrossCov = zeros(4,4,Step_Num);

NaiveFusionCenter.CombinedStateEstimate(:,1) = x0_bar;
NaiveFusionCenter.CombinedStateEstimateCov(:,:,1) = P0_bar;

CovIntFusionCenter.CombinedStateEstimate(:,1) = x0_bar;
CovIntFusionCenter.CombinedStateEstimateCov(:,:,1) = P0_bar;

CrossCovFusionCenter.CombinedStateEstimate(:,1) = x0_bar;
CrossCovFusionCenter.CombinedStateEstimateCov(:,:,1) = P0_bar;

eps_naive = zeros(MC_Num,round(Step_Num/2)); % Normalized Estimation Error Squares Matrix
err_naive = zeros(4,round(Step_Num/2),MC_Num);

eps_covint = zeros(MC_Num,round(Step_Num/2)); % Normalized Estimation Error Squares Matrix
err_covint = zeros(4,round(Step_Num/2),MC_Num);

eps_crosscov = zeros(MC_Num,round(Step_Num/2)); % Normalized Estimation Error Squares Matrix
err_crosscov = zeros(4,round(Step_Num/2),MC_Num);

eps_opt_crosscov = zeros(MC_Num,round(Step_Num/2)); % Normalized Estimation Error Squares Matrix
err_opt_crosscov = zeros(4,round(Step_Num/2),MC_Num);


for MC = 1:MC_Num
%     
    err_naive(:,1,MC) = x_k(:,1,MC)-NaiveFusionCenter.CombinedStateEstimate(:,1);
    eps_naive(MC,1) = err_naive(:,1,MC)'*inv(NaiveFusionCenter.CombinedStateEstimateCov(:,:,1))*err_naive(:,1,MC);
     
    err_covint(:,1,MC) = x_k(:,1,MC)-CovIntFusionCenter.CombinedStateEstimate(:,1);
    eps_covint(MC,1) = err_covint(:,1,MC)'*inv(CovIntFusionCenter.CombinedStateEstimateCov(:,:,1))*err_covint(:,1,MC);
    
    err_crosscov(:,1,MC) = x_k(:,1,MC)-CrossCovFusionCenter.CombinedStateEstimate(:,1);
    eps_crosscov(MC,1) = err_crosscov(:,1,MC)'*inv(CrossCovFusionCenter.CombinedStateEstimateCov(:,:,1))*err_crosscov(:,1,MC);
      
    err_opt_crosscov(:,1,MC) = x_k(:,1,MC)-CrossCovFusionCenter.CombinedStateEstimate(:,1);
    eps_opt_crosscov(MC,1) = err_opt_crosscov(:,1,MC)'*inv(CrossCovFusionCenter.CombinedStateEstimateCov(:,:,1))*err_opt_crosscov(:,1,MC);
    
    for i=2:Step_Num
        
        for k=1:2
            
            % Prediction:
            
            [StatePrediction, ...
                StatePredictionCov, ...
                OutputPrediction, ...
                OutputPredictionsCov, ...
                Local_Trackers(k).KalmanGain] = kf_pre(Local_Trackers(k).StateEstimate(:,i-1),...
                Local_Trackers(k).StateEstimateCov(:,:,i-1),...
                Local_Trackers(k).Parameters);
            
            % Estimation:
            [...
                Local_Trackers(k).StateEstimate(:,i), ...
                Local_Trackers(k).StateEstimateCov(:,:,i),~] = kf_est(...
                StatePrediction,...
                StatePredictionCov,...
                OutputPrediction, ...
                OutputPredictionsCov, ...
                Local_Trackers(k).KalmanGain,...
                yk_centralized(2*k-1:2*k,i,MC));
            

        end
        % Optimal Fusion CrossCov
        CrossCov(:,:,i) = CrossCovCalculator(Local_Trackers,CrossCov(:,:,i-1));
        
            
        if  mod(i,2) == 1
            
            i_ = (i+1)/2;
            
            % Naive Fusion
            
            InvLocalAgent1_Cov = inv(Local_Trackers(1).StateEstimateCov(:,:,i));
            InvLocalAgent2_Cov = inv(Local_Trackers(2).StateEstimateCov(:,:,i));
            
            NaiveFusionCenter.CombinedStateEstimateCov(:,:,i_) = inv(InvLocalAgent1_Cov+InvLocalAgent2_Cov);
            NaiveFusionCenter.CombinedStateEstimate(:,i_) = NaiveFusionCenter.CombinedStateEstimateCov(:,:,i_)*(InvLocalAgent1_Cov*Local_Trackers(1).StateEstimate(:,i)+...
                                                                                                      InvLocalAgent2_Cov*Local_Trackers(2).StateEstimate(:,i));
            
                                                                                                  
                                                                                             
            
            err_naive(:,i_,MC) = x_k(:,i,MC)-NaiveFusionCenter.CombinedStateEstimate(:,i_);
            eps_naive(MC,i_) = err_naive(:,i_,MC)'*inv(NaiveFusionCenter.CombinedStateEstimateCov(:,:,i_))*err_naive(:,i_,MC);     
            
            % Covariance Intersection 
            w = Covariance_Intersection(InvLocalAgent1_Cov,InvLocalAgent2_Cov);
            CovIntFusionCenter.CombinedStateEstimateCov(:,:,i_) = inv(w*InvLocalAgent1_Cov+(1-w)*InvLocalAgent2_Cov);
            CovIntFusionCenter.CombinedStateEstimate(:,i_) = CovIntFusionCenter.CombinedStateEstimateCov(:,:,i_)*(w*InvLocalAgent1_Cov*Local_Trackers(1).StateEstimate(:,i)+...
                                                                                                      (1-w)*InvLocalAgent2_Cov*Local_Trackers(2).StateEstimate(:,i)); 
            
            err_covint(:,i_,MC) = x_k(:,i,MC)-CovIntFusionCenter.CombinedStateEstimate(:,i_);
            eps_covint(MC,i_) = err_covint(:,i_,MC)'*inv(CovIntFusionCenter.CombinedStateEstimateCov(:,:,i_))*err_covint(:,i_,MC);
            

            % Heuristic Cross Cov
            
            HeuristicCrossCov = .4*(Local_Trackers(1).StateEstimateCov(:,:,i).*Local_Trackers(2).StateEstimateCov(:,:,i)).^(1/2);
     
           InvLocalAgent1_Cov =  inv(Local_Trackers(1).StateEstimateCov(:,:,i)-HeuristicCrossCov );
           InvLocalAgent2_Cov = inv(Local_Trackers(2).StateEstimateCov(:,:,i)-HeuristicCrossCov );
           CrossCovFusionCenter.CombinedStateEstimateCov(:,:,i_) = inv(InvLocalAgent1_Cov+InvLocalAgent2_Cov);  
           CrossCovFusionCenter.CombinedStateEstimate(:,i_) = CrossCovFusionCenter.CombinedStateEstimateCov(:,:,i_)*(InvLocalAgent1_Cov*Local_Trackers(1).StateEstimate(:,i)+...
                                                                                                      InvLocalAgent2_Cov*Local_Trackers(2).StateEstimate(:,i));
            
            
            err_crosscov(:,i_,MC) = x_k(:,i,MC)-CrossCovFusionCenter.CombinedStateEstimate(:,i_);
            eps_crosscov(MC,i_) = err_crosscov(:,i_,MC)'*inv(CrossCovFusionCenter.CombinedStateEstimateCov(:,:,i_))*err_crosscov(:,i_,MC);
            
             % Optimum Cross Cov
           InvLocalAgent1_Cov =  inv(Local_Trackers(1).StateEstimateCov(:,:,i)-CrossCov(:,:,i));
           InvLocalAgent2_Cov = inv(Local_Trackers(2).StateEstimateCov(:,:,i)-CrossCov(:,:,i));
           OptCrossCovFusionCenter.CombinedStateEstimateCov(:,:,i_) = inv(InvLocalAgent1_Cov+InvLocalAgent2_Cov);  
           OptCrossCovFusionCenter.CombinedStateEstimate(:,i_) = OptCrossCovFusionCenter.CombinedStateEstimateCov(:,:,i_)*(InvLocalAgent1_Cov*Local_Trackers(1).StateEstimate(:,i)+...
                                                                                                      InvLocalAgent2_Cov*Local_Trackers(2).StateEstimate(:,i));
            
            
            err_opt_crosscov(:,i_,MC) = x_k(:,i,MC)-OptCrossCovFusionCenter.CombinedStateEstimate(:,i_);
            eps_opt_crosscov(MC,i_) = err_opt_crosscov(:,i_,MC)'*inv(OptCrossCovFusionCenter.CombinedStateEstimateCov(:,:,i_))*err_opt_crosscov(:,i_,MC);
            
        end 
    end 
end
 

time = 0:2:Step_Num-1;
err_naive_mean = sqrt(mean(err_naive.^2,3));
err_covint_mean = sqrt(mean(err_covint.^2,3));
err_crosscov_mean = sqrt(mean(err_crosscov.^2,3));
err_opt_crosscov_mean = sqrt(mean(err_opt_crosscov.^2,3));

eps_naive_mean = mean(eps_naive,1);
eps_covint_mean = mean(eps_covint,1);
eps_crosscov_mean = mean(eps_crosscov,1);
eps_crosscov_opt_mean = mean(eps_opt_crosscov,1);

figure;
thresh_min = chi2inv(0.005,MC_Num*4)/MC_Num;
thresh_max = chi2inv(0.995,MC_Num*4)/MC_Num;
hold on
plot(time,eps_naive_mean)
plot(time,eps_covint_mean)
plot(time,eps_crosscov_mean)
plot(time,eps_crosscov_opt_mean)

plot(time,repmat(thresh_min,1,50))
plot(time,repmat(thresh_max,1,50))
legend({'Naive','Covariance Intersection','Cross Covariance','Optimum Cross Covariance','Threshold Min','Threshold Max'}, 'fontsize', 10);
ylabel('NEES', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;


figure;
hold on;
plot(time(2:end),err_naive_mean(1,2:end));
plot(time(2:end),err_covint_mean(1,2:end));
plot(time(2:end),err_crosscov_mean(1,2:end));
plot(time(2:end),err_opt_crosscov_mean(1,2:end));
legend({'Naive','Covariance Intersection','Cross Covariance','Optimum Cross Covariance'}, 'fontsize', 10);
ylabel('RMS of Position X', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;

figure;
hold on;
plot(time(2:end),err_naive_mean(2,2:end));
plot(time(2:end),err_covint_mean(2,2:end));
plot(time(2:end),err_crosscov_mean(2,2:end));
plot(time(2:end),err_opt_crosscov_mean(2,2:end));
legend({'Naive','Covariance Intersection','Cross Covariance','Optimum Cross Covariance'}, 'fontsize', 10);
ylabel('RMS of Position Y', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;


figure;
hold on;
plot(time(2:end),err_naive_mean(3,2:end));
plot(time(2:end),err_covint_mean(3,2:end));
plot(time(2:end),err_crosscov_mean(3,2:end));
plot(time(2:end),err_opt_crosscov_mean(3,2:end));
legend({'Naive','Covariance Intersection','Cross Covariance','Optimum Cross Covariance'}, 'fontsize', 10);
ylabel('RMS of Velocity X', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;


figure;
hold on;
plot(time(2:end),err_naive_mean(4,2:end));
plot(time(2:end),err_covint_mean(4,2:end));
plot(time(2:end),err_crosscov_mean(4,2:end));
plot(time(2:end),err_opt_crosscov_mean(4,2:end));
legend({'Naive','Covariance Intersection','Cross Covariance','Optimum Cross Covariance'}, 'fontsize', 10);
ylabel('RMS of Velocity Y', 'fontsize', 12); grid on; xlabel('Time', 'fontsize', 12); grid on;

