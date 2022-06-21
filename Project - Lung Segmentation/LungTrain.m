%%
clc

imds = imageDatastore("C:\Users\tahre\Documents\NormalOrCovid", ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

numTrainFiles = 1500;
[imdsTrain2,imdsValidation2] = splitEachLabel(imds,numTrainFiles,'randomize');

inputSize = [48 48];
imdsTrain=augmentedImageDatastore(inputSize, imdsTrain2,'ColorPreprocessing','rgb2gray');
imdsValidation=augmentedImageDatastore(inputSize, imdsValidation2,'ColorPreprocessing','rgb2gray');

numClasses = 2;

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(4,8,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(4,16,'Padding','same')
    batchNormalizationLayer
    reluLayer     
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',8, ...
    'Shuffle', 'every-epoch',...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation2.Labels;
accuracy = mean(YPred == YValidation);
disp(accuracy)

%%

I = imread("C:\Users\tahre\Documents\NormalOrCovid\Normal\normal-12.png");
I = imresize(I, [48 48]);


label = classify(net,I);% Classify the picture
I = imresize(I, [256 256]);
imshow(I)
title(char(label));          % Show the class label
drawnow


