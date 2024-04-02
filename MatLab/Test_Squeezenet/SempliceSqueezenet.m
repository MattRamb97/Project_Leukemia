%% creo il training set
clear all
warning off

%carica dataset
load('Train_Dataset.mat','DATA');
yE=DATA{2};%label dei patterns
NX=DATA{1};%immagini

siz=[227 227];
clear nome trainingImages
for pattern=1:size(NX,2)
    IM=NX(pattern).im;%singola data immagine
    %si deve fare resize immagini per rendere compatibili con CNN
    IM=imresize(IM,[siz(1) siz(2)]);
    if size(IM,3)==1
        IM(:,:,2)=IM;
        IM(:,:,3)=IM(:,:,1);
    end
    trainingImages(:,:,:,pattern)=IM;
end

%% carica rete pre-trained
net = squeezenet;  %load Squeezenet
siz=[227 227];

numClasses = 2;%number of classes

lgraph = layerGraph(net); 

newConvLayer =  convolution2dLayer([1, 1], numClasses, 'WeightLearnRateFactor',10,'BiasLearnRateFactor',10, 'Name','new_conv');
lgraph = replaceLayer(lgraph,'conv10',newConvLayer);

newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions', newClassificatonLayer);

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageSource(imageSize,trainingImages, ...
        categorical(yE'),'DataAugmentation',imageAugmenter);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',11, ...
    'MaxEpochs',7, ...
    'InitialLearnRate',2e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,lgraph,options);


%% creo test set
root = '../mio_code/Test_Squeezenet/pTest_Dataset.mat';
load(root,'DATA');

NX = DATA{1};

clear nome test testImages
for pattern=1:size(NX,2)
    IM=NX(pattern).im;%singola data immagine

    IM=imresize(IM,[siz(1) siz(2)]);
    if size(IM,3)==1
        IM(:,:,2)=IM;
        IM(:,:,3)=IM(:,:,1);
    end
    testImages(:,:,:,pattern)=IM;
end
    
    
%% classifico test patterns
[YPred,scores] = classify(netTransfer,testImages);    

yy = cell2mat(DATA{4});

%calcolo accuracy
[a,b]=max(scores');

% modifica classificazione per 0 e 1 e non classi 1 e 2
for i=1:1867
    b(i)=b(i)-1;
end

ACC=sum(b==yy)./length(yy);








