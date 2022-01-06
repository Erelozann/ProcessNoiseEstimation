function [w_star] = Covariance_Intersection(inv_Cov_B,inv_Cov_C)

discretization_step = 20;
w = linspace(0,1,discretization_step);
y = zeros(1,discretization_step);

for i=1:discretization_step
    Cov_A_temp = inv(w(i)*inv_Cov_B + (1-w(i))*inv_Cov_C);
    y(i) = det(Cov_A_temp);
end
[~,star_index] = min(y);
 w_star = w(star_index);
   
    