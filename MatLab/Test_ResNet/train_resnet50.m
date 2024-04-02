function [ tResNet50 , acc_ptest] = train_resnet50()

    % ---------------------------------------------------------------------
    % DATASET E CREAZIONE DEL TRAINING SET %
    % ---------------------------------------------------------------------
    
    %carico set PHASE-I
    root = '../mio_code/Test_Squeezenet/Train_Dataset.mat';
    load(root,'DATA');
    
    % all -> class 1
    % hem -> class 0
    NX=DATA{1}; %immagini
    yE=num2cell(DATA{2}); %labels
    

    clear nome trainingImages
    for pattern=1:size(NX,2)

        IM=double(NX(pattern).im); %singola data immagine
        
        avgclr = mean(mean(IM(178:273,178:273,:,:),1),2); %normalizzazione
        IM = bsxfun(@minus, IM(114:337,114:337,:,:), avgclr);
        
        trainingImages(:,:,:,pattern)=IM;
    end
    
    pattern = size(NX,2) + 1;
    
    % carico set PHASE-II
    root = '../mio_code/Test_Squeezenet/pTest_Dataset.mat';
    load(root,'DATA');
    NX=DATA{1}; %immagini
    yE2=DATA{4}; %labels

    for i=1:size(NX,2)
        
        IM=double(NX(i).im); %singola data immagine
        
        avgclr = mean(mean(IM(178:273,178:273,:,:),1),2); %normalizzazione
        IM = bsxfun(@minus, IM(114:337,114:337,:,:), avgclr);
        
        trainingImages(:,:,:,pattern)=IM;
        pattern = pattern + 1;
    end
    
    yEf=[yE,yE2]; % labels totali finali
    
    imageSize=size(IM); %size imput ResNet50
    
    % ---------------------------------------------------------------------
    % CARICAMENTO E ADDESTRAMENTO DELLA RESNET50 %
    % ---------------------------------------------------------------------
    
    net = resnet50; % load della rete ResNet50

    % parametri rete neurale
    miniBatchSize = 32; % come vincitore del contest
    learningRate = 0.001; % per le prime 10 epoche
    metodoOptim='sgdm'; % stochastic gradient descent
    options = trainingOptions(metodoOptim,...
        'MiniBatchSize',miniBatchSize,...
        'MaxEpochs',20,...
        'InitialLearnRate',learningRate,...
        'LearnRateDropFactor',0.1, ... % riduco il learning rate dopo 10 epoche da 0.001 a 0.0001
        'LearnRateDropPeriod',10, ...
        'Verbose',false,...
        'Plots','training-progress');

    % creazione pattern aggiuntivi mediante tecnica standard
    % come descritto nel paper del vincitore come data augmentation vado a
    % utilizzare la rotazione random delle immagini da 0 a 350°
    imageAugmenter = imageDataAugmenter( ...
            'RandXReflection',true, ...
            'RandXScale',[1 2], ...
            'RandYReflection',true, ...
            'RandYScale',[1 2],...
            'RandRotation',[0 350],...
            'RandXTranslation',[0 5],...
            'RandYTranslation', [0 5]);

    augimdsTrain = augmentedImageDatastore(imageSize,trainingImages, ...
            categorical(yEf'),'DataAugmentation',imageAugmenter);

    
    lgraph = layerGraph(net); % lgraph dalla rete ResNet50

    % Creazione del nuovo layer fullyconnected e sostituzione
    numClasses = 2;
    layer_1 = fullyConnectedLayer(numClasses, ...
            'WeightsInitializer','glorot', ...
            'BiasLearnRateFactor', 0, ...
            'Name','fc2');
    lgraph = replaceLayer(lgraph,'fc1000',layer_1);

    % Creazione del nuovo layer softmax e sostituzione
    layer_2 = softmaxLayer('Name','fc2_softmax');
    lgraph = replaceLayer(lgraph,'fc1000_softmax',layer_2);

    % Creazione del nuovo layer classification e sostituzione
    layer_3 = classificationLayer( ...
        'Classes', ["0" "1"], ... 
        'Name','ClassificationLayer_fc2');

    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',layer_3);

    % addestramento rete neurale
    tResNet50 = trainNetwork(augimdsTrain,lgraph,options);

    % ---------------------------------------------------------------------
    % TEST DELLA RETE TRAMITE IL PRELIMINARY TEST DATASET %
    % ---------------------------------------------------------------------

    % creo TEST set
    root = '../mio_code/Test_Squeezenet/pTest_Dataset.mat';
    load(root,'DATA');

    NX = DATA{1};

    clear nome test testImages
    for pattern=1:size(NX,2)
        IM=double(NX(pattern).im);%singola data immagine

        avgclr = mean(mean(IM(178:273,178:273,:,:),1),2);
        IM = bsxfun(@minus, IM(114:337,114:337,:,:), avgclr);
        
        testImages(:,:,:,pattern)=IM;
    end
    
    % classifico test patterns
    outclass = classify(tResNet50,testImages);
    
    yy = cell2mat(DATA{4});
    
    %calcolo accuracy
    acc_ptest = sum(str2num(char(outclass))==yy')./length(yy);
    
end