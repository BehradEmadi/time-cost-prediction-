% Clear workspace and load data
clc;
clear;

% Load the dataset
load('project2.mat'); % Ensure the file extension is correct

% Transpose inputs and targets for the neural network
inputs = train_in';
targets = train_targets';

% Initialize variables
numHiddenLayers = 10; % Number of hidden layers to test
perf = Inf(numHiddenLayers, 1); % Performance of each network
outputs = cell(numHiddenLayers, 1); % Store outputs for each network
tr_mem = cell(numHiddenLayers, 1); % Training record for each network

% Train neural networks with different numbers of hidden neurons
for i = 1:numHiddenLayers
    % Define and train the neural network
    net = newff(inputs, targets, i);
    [net, tr] = train(net, inputs, targets);
    
    % Store training record and network performance
    tr_mem{i} = tr;
    outputs{i} = net(inputs);
    perf(i) = perform(net, outputs{i}, targets);
    
    % Track the best performing network
    if perf(i) < min(perf)
        net_best = net;
        best_neuron = i;
        best_performance = perf(i);
    end
end

% Sort performances and find the best network
[sorted_performance, idx] = sort(perf);
best_neuron = idx(1);
best_performance = sorted_performance(1);

% Plot performance of the best network
figure;
plotperform(tr_mem{best_neuron});
title('Training Performance of the Best Network');

% Save the best network
save('filename.mat', 'net_best');

% Test the best network
train_outputs = net_best(train_in');
test_outputs = net_best(test_in');

% Plot regression results
figure;
plotregression(targets, train_outputs, 'Train', test_targets', test_outputs, 'Test');
title('Regression of Best Network Outputs');
