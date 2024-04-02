%% creo il training set
clear all
warning off

%scegli valore di datas in base a quale dataset vi serve
%datas=29;

%carica dataset
load('Train_Dataset.mat','DATA');
%NF=size(DATA{3},1); %number of folds
%DIV=DATA{3};%divisione fra training e test set
%DIM1=DATA{1};%numero di training pattern
%DIM2=DATA{5};%numero di pattern
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
imageSize=size(IM);
   
    
%% carica rete pre-trained
net = alexnet;  %load AlexNet
siz=[227 227];
%se riesci con i tempi computazionali prova:
%net = vgg16;
%siz=[224 224];

%parametri rete neurale
miniBatchSize = 30;
learningRate = 1e-4;
metodoOptim='sgdm';
options = trainingOptions(metodoOptim,...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',30,...
    'InitialLearnRate',learningRate,...
    'Verbose',false,...
    'Plots','training-progress');
numIterationsPerEpoch = floor(10661/miniBatchSize);

%for fold=1:NF
    %close all force
    %try 
    %DIM1=DATA{4}(fold);
    %end
    
    %trainPattern=(DIV(fold,1:DIM1));
    %testPattern=(DIV(fold,DIM1+1:DIM2));
    %y=yE(DIV(fold,1:DIM1));%training label
    %yy=yE(DIV(fold,DIM1+1:DIM2));%test label
    numClasses = 2;%number of classes
    
    %inserire qui funzione per creare pose aggiuntive, in input si prende
    %(trainingImages,y) e in output restituisci una nuova versione di
    %(trainingImages,y) aggiornata con nuove immagini
    %%%%%%%%%%%
    
    %creazione pattern aggiuntivi mediante tecnica standard
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXScale',[1 2], ...
        'RandYReflection',true, ...
        'RandYScale',[1 2],...
        'RandRotation',[-10 10],...
        'RandXTranslation',[0 5],...
        'RandYTranslation', [0 5]);
    augimdsTrain = augmentedImageSource(imageSize,trainingImages, ...
        categorical(yE'),'DataAugmentation',imageAugmenter);
    
    % tuning della rete
    % The last three layers of the pretrained network net are configured for 1000 classes.
    % These three layers must be fine-tuned for the new classification problem. Extract all layers, except the last three, from the pretrained network.
    layersTransfer = net.Layers(1:end-3);
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];
    netTransfer = trainNetwork(augimdsTrain,layers,options);
  
    
    %% creo test set
    load('pTest_Dataset.mat','DATA');
    
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
    [outclass, score] =  classify(netTransfer,testImages);
    
    yy = DATA{4};
    %calcolo accuracy
    [a,b]=max(score');
    ACC=sum(b==yy)./length(yy);
    
    %salvate quello che vi serve
    %%%%%
    
%end


