function karmanfilter

%Set biases for wheel encoder, steering position, GPS velocity and GPS position
bias_wheel = +1;
bias_str = -1;
bias_gps_v = 2;
bias_gps_p = 0;

% Measurement noise covariance
R = 0.64;

% Process noise covariance
Q = .005;

% State transition model
A = 1;

% Observation model
C1 = 1;
C2 = 1;
C3 = 1;
C4 = 1;

% Duration
N = 1000;

% Run ====================================================================

% Generate random signals for simulation
x_wheel = 30 + sin(5*linspace(0,1,N)*pi);
x_str = 32 + cos(5*linspace(0,1,N)*pi);
x_gps_v = 27 + sin(5*linspace(0,1,N)*pi);
x_gps_p = sin(5*linspace(0,1,N)*pi) + 20 + cos(5*linspace(0,1,N)*pi);

%average signal value
x_ave = (x_wheel + x_str + x_gps_v + x_gps_p)/4;

% Add some process noise with covariance Q
w_wheel = sqrt(Q) * randn(size(x_wheel));
w_str = sqrt(Q) * randn(size(x_str));
w_gps_v = sqrt(Q) * randn(size(x_gps_v));
w_gps_p = sqrt(Q) * randn(size(x_gps_p));
x_wheel = x_wheel + w_wheel;
x_str = x_str + w_str;
x_gps_v = x_gps_v + w_gps_v;
x_gps_p = x_gps_p + w_gps_p;

% simulate noisy sensor values
z_wheel = bias_wheel + x_wheel + sqrt(R) * randn(size(x_wheel));
z_str = bias_str + x_str + sqrt(R) * randn(size(x_str));
z_gps_v = bias_gps_p + x_gps_v + sqrt(R) * randn(size(x_gps_v));
z_gps_p = bias_gps_v + x_gps_p + sqrt(R) * randn(size(x_gps_p));

% Run the Kalman filter on fused sensors
xhat = kalman([z_wheel; z_str; z_gps_v; z_gps_p], A, [C1; C2; C3;C4], [R 0 0 0; 0 R 0 0; 0 0 R 0; 0 0 0 R;], Q);

% Plot 
subplot(1,1,1)
hold on
plot(x_ave, 'k')
plot(xhat, 'r')
legend({'Actual', 'Estimated'})
hold off
title(sprintf('%s: RMS error = %f', 'Plot', sqrt(sum((x_ave-xhat).^2)/length(x_ave))))



function xhat = kalman(z, A, C, R, Q)
% Simple Kalman Filter (linear) with optimal gain, no control signal
%
% z Measurement signal              m observations X # of observations
% A State transition model          n X n, n = # of state values
% C Observation model               m X n
% R Covariance of observation noise m X m
% Q Covariance of process noise     n X n
%
% Based on http://en.wikipedia.org/wiki/Kalman_filter, but matrices named
% A, C, G instead of F, H, K.


% Initializtion =====================================================

% Number of sensors
m = size(C, 1);

% Number of state values
n = size(C, 2);

% Number of observations
numobs = size(z, 2);

% Use linear least squares to estimate initial state from initial
% observation
xhat = zeros(n, numobs);
xhat(:,1) = C \ z(:,1);

% Initialize P, I
P = ones(size(A));
I = eye(size(A));

% Kalman Filter =====================================================

for k = 2:numobs
    
    % Predict
    xhat(:,k) = A * xhat(:,k-1);
    P         = A * P * A' + Q;
    
    % Update
    G         = P  * C' / (C * P * C' + R);
    P         = (I - G * C) * P;
    xhat(:,k) = xhat(:,k) + G * (z(:,k) - C * xhat(:,k));
    
end