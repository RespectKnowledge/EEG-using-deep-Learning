%%  %%%%%%%%%%%% The Classification of EEG using CNN network %%%%%%%%%%%%%%%%%%%%%%%%


%%   Second approach Design Network with random generation dataset
% You should use your own dataset according to design ceriteria
Training_ReductWindows_G=rand(662,800);
% The data specification
height = 1;
width = 800;
channels = 1;
sampleSize = 662;
CNN_TrainingData = reshape(Training_ReductWindows_G,[height, width, channels, sampleSize]);
% Training_Labels_Bin_G=[ones(1,662);2*ones(1,662);3*ones(1,662)]
% CNN_TrainingLabels =categorical(Training_Labels_Bin_G)'; 
%% The Training Labels for each class
label(1:220,:) = {'W'} % 1st Label %
label(220:440,:) = {'X'}; % 2nd Label %
label(440:662,:)={'Z'};
CNN_TrainingLabels = categorical(label); % Label vector is ready %
%%  The Network Design in the paper
InputLayer = imageInputLayer([height,width,channels]); %'DataAugmentation', 'none'); %'Normalization', 'none');
%inputLayer=imageInputLayer([1 6000]);
c1=convolution2dLayer([1 200],20,'stride',1);
p1=maxPooling2dLayer([1 20],'stride',10);
c2=convolution2dLayer([1 30],400,'numChannels',20);
p2=maxPooling2dLayer([1 10],'stride',[1 2]);
f1=fullyConnectedLayer(500);
f2=fullyConnectedLayer(3);
s1=softmaxLayer;
outputLayer=classificationLayer;
convnet=[InputLayer; c1; p1; c2; p2; f1; f2; s1;outputLayer]
%% How to build hyperparameters of the network
%opts = trainingOptions('sgdm');
% Define the Training options
                opts=trainingOptions('sgdm',...
                'InitialLearnRate',0.001,...
                'LearnRateSchedule','none',...
                'LearnRateDropPeriod',8,...
                'L2Regularization',0.005,...
                'MaxEpochs',100,...
                'MiniBatchSize',32,...
                'Verbose',true)

%% Train the Model using dataset and number of training labels, model and hyper-parameters
convnetModel = trainNetwork(CNN_TrainingData, CNN_TrainingLabels, convnet, opts);

% Design Feature matricis based on CCN features at particular layer
trainingFeatures_conf1 = activations(convnetModel, CNN_TrainingLabels,'c2','MiniBatchSize', 32, 'OutputAs', 'columns');
%% Compute the accurcay of that network
[labels,err_test] = classify(convnetModel, CNN_TrainingLabels, 'MiniBatchSize', 64);
confMat = confusionmat(CNN_TrainingLabels.Labels, labels);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
mean(diag(confMat));