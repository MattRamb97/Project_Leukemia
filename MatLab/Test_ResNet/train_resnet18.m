function [ tResNet18 , acc_ptest] = train_resnet18()

    % ---------------------------------------------------------------------
    % CREAZIONE DEL TRAINING SET %
    % ---------------------------------------------------------------------
    
    %carico set PHASE-I
    root = '../mio_code/Test_Squeezenet/Train_Dataset.mat';
    load(root,'DATA');
    % labels
    % all -> class 1
    % hem -> class 0
    yE=num2cell(DATA{2});
    NX=DATA{1}; %immagini

    clear nome trainingImages
    for pattern=1:(size(NX,2)-1)/4

        %singola data immagine
        IM=double(NX(pattern).im); 
        
        avgclr = mean(mean(IM(178:273,178:273,:,:),1),2);
        IM = bsxfun(@minus, IM(114:337,114:337,:,:), avgclr);
        
        trainingImages(:,:,:,pattern)=IM;
    end
    
    pattern = (size(NX,2)-1)/4 + 1;
    
    % carico set PHASE-II
    root = '../mio_code/Test_Squeezenet/pTest_Dataset.mat';
    load(root,'DATA');
    NX=DATA{1};
    yE2=DATA{4};
    
    for i=1:size(NX,2)

        %singola data immagine
        IM=double(NX(i).im); 
        
        avgclr = mean(mean(IM(178:273,178:273,:,:),1),2);
        IM = bsxfun(@minus, IM(114:337,114:337,:,:), avgclr);
        
        trainingImages(:,:,:,pattern)=IM;
        pattern = pattern + 1;
    end
    
    % labels totali finali
    yEf=[yE(1:2665),yE2];
    
    imageSize=size(IM);
    
    % ---------------------------------------------------------------------
    % CARICAMENTO E ADDESTRAMENTO DELLA RESNET18 %
    % ---------------------------------------------------------------------
    
    % load della rete ResNet50
    net = resnet18;

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
            categorical(cell2mat(yEf')),'DataAugmentation',imageAugmenter);

    % lgraph dalla rete ResNet50
    lgraph = layerGraph(net);

    % Creazione del nuovo layer fullyconnected e sostituzione
    numClasses = 2;
    layer_1 = fullyConnectedLayer(numClasses, ...
            'WeightsInitializer','glorot', ...
            'BiasLearnRateFactor', 0, ...
            'Name','fc2');
    lgraph = replaceLayer(lgraph,'fc1000',layer_1);

    % Creazione del nuovo layer softmax e sostituzione
    layer_2 = softmaxLayer('Name','fc2_softmax');
    lgraph = replaceLayer(lgraph,'prob',layer_2);

    % Creazione del nuovo layer classification e sostituzione
    layer_3 = classificationLayer( ...
        'Classes', ["0" "1"], ... 
        'Name','ClassificationLayer_fc2');

    lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',layer_3);

    % assemblo la rete neurale dopo aver modificato i layer per 2 classi di
    % output come da problema 

    tResNet18 = trainNetwork(augimdsTrain,lgraph,options);

    % ---------------------------------------------------------------------
    % TEST DELLA RETE TRAMITE IL PRELIMINARY TEST DATASET %
    % ---------------------------------------------------------------------

    % creo test set
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
    outclass =  classify(tResNet18,testImages);
    
    yy = cell2mat(DATA{4});
    
    %calcolo accuracy
    acc_ptest=sum(str2num(char(outclass))==yy')./length(yy);
    
end