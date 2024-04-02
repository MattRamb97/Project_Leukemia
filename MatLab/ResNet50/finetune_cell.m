% MAIN del codice della funzione finetune_cell richiamata dal codice del
% file Phrase_III_of_ISBI_NMC.m

% varargin -> 'ininallabel', [], 'trainset', [0,1,2,3], 'testset', -1, 'whichresnet', 50

function [net, info] = finetune_cell(varargin)

    % MODIFICATO flag per utilizzo della gpu
    useGPU = 0;

    opts.rootPath = '../data/';
    opts.trainset = [1,2,3];
    opts.testset = [0];
    opts.whichresnet = 50;
    opts.ininallabel = [];

    % aggiorna la struttura opts in base alle coppie parametro-valore specificate
    % da varargin
    % opts.trainset diventa [0,1,2,3]
    % opts.testset -> -1
    % opts.wichresnet -> 50, 101 o 152
    [opts, varargin] = vl_argparse(opts, varargin) ;

    % expDir - > '../data/models/cell50res-0 1 2 3'
    % serve per recuperare il file imdb.mat 
    opts.expDir = fullfile(opts.rootPath, 'models/', ...
        sprintf('cell%dres-%s', opts.whichresnet, num2str(opts.trainset)));

    % MODIFICATO
    % sostituisco per modelpath sprintf con fullfile, commentando la riga del
    % primo codice citato
    % modelpath -> '../data/models/imagenet-resnet-50-dag.mat'
    opts.modelpath =  fullfile(opts.rootPath, 'models/', ...
        sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet));
    %opts.modelpath =  sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet);

    % MODIFICATO
    % tolgo il simbolo '/' da '/cells' per correttezza del root della directory
    % dataDir -> '../data/cells
    opts.dataDir = fullfile(opts.rootPath, 'cells');
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
    opts.whitenData = true ;
    opts.contrastNormalization = true ;
    opts.train = struct('gpus', useGPU) ;

    % ripetizione inutile perchè già effettuato nella 22esima riga
    opts = vl_argparse(opts, varargin) ;

    if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end

    % imdbPath -> '../data/models/cell50res-0 1 2 3/imdb.mat'
    if exist(opts.imdbPath, 'file')
        imdb = load(opts.imdbPath) ;
    else
        imdb = cell_get_database(opts);
        mkdir(opts.expDir) ;
        save(opts.imdbPath, '-struct', 'imdb');
    end

    % ininallabel prese dal file imdb.mat per le immagini 
    if ~isempty(opts.ininallabel)
        imdb.images.label(ismember(imdb.images.set, opts.testset))=opts.ininallabel;
    end

    % richiamo alla funzione secondaria cnn_init_res
    net = cnn_init_res('modelPath', opts.modelpath, 'numcls', numel(imdb.meta.classes)) ;
    
    net.meta.classes.name = imdb.meta.classes(:)' ;
    
    trainfn = @cnn_train_dag ;
    
    % [net,stats] = cnn_train_dag(net, imdb, getBatch, varargin)
    [net, info] = trainfn(net, imdb, getBatch(opts), ...
        'expDir', opts.expDir, ...
        net.meta.trainOpts, ...
        opts.train, ...
        'train', find(ismember(imdb.images.set, opts.trainset)),...
        'val', find(ismember(imdb.images.set, opts.testset))) ;
end

% -------------------------------------------------------------------------

% FUNZIONI SECONDARIE RICHIAMATE 

% funzione getBatch richiamata per trainfn -> cnn_train_dag 
function fn = getBatch(opts)
    bopts = struct('numGpus', numel(opts.train.gpus), 'trainset', opts.trainset, 'modelType', 'res') ;
    
    % richiamo ad altra funzione secondaria
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end


% funzione cnn_init_res richiamata per creare la rete net
function net = cnn_init_res(varargin)
opts.numcls = 1000;
opts.modelPath = 'imagenet-resnet-101-dag.mat';

% aggiorna la struttura opts in base alle coppie parametro-valore specificate
% da varargin
% opts.numcls -> numel(imdb.meta.classes)
% opts.modelPath -> '../data/models/imagenet-resnet-50-dag.mat'
opts = vl_argparse(opts, varargin) ;

% load della rete ResNet50
net = load(opts.modelPath);

% definizione della dimensione di inpute delle immagini per la rete
net.meta.inputSize = [224 224 3];

% definizione del learning rate per ognuna delle 30 epoche
% [ 0.001 (10 epoche), 0.0001 (10 epoche), 0.0001 (10 epoche) ]
net.meta.trainOpts.learningRate = [0.001 * ones(1,10), 0.0001 * ones(1,10), 0.0001 * ones(1,10)];
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 32;

% numero di elementi di net.meta.trainOpts.learningRate -> 30 elementi
% quindi 30 epoche
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% load della rete preaddestrata ResNet50
net = dagnn.DagNN.loadobj(net) ;


net.layers(end-1).block.size = [1 1 2048 opts.numcls];
net.params(end-1).value = net.params(end-1).value(:,:,:,1+mod(0:opts.numcls-1,1000));
net.params(end).value = net.params(end).value(1+mod(0:opts.numcls-1,1000),:);
net.addLayer('loss', dagnn.Loss('loss', 'log'), ...
    {'prob','label'}, 'objective') ;
net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
    {'prob','label'}, 'error') ;
end

% -------------------------------------------------------------------------

function inputs = getDagNNBatch(opts, imdb, batch)

% MODIFICATO flag per utilizzo della gpu
useGPU = 1;

imfiles = fullfile(imdb.imageDir, imdb.images.name(batch));

% MODIFICATO
% Aggiunto per modificare i link alle directory delle immagini
for ii=1:length(imfiles)
    imfiles{ii}= replace(imfiles{ii},'\','/');
end

% MODIFICATO a 0 per mio test
opts.numGpus = useGPU;

if ismember(imdb.images.set(batch(1)),opts.trainset)
    if opts.numGpus > 0
        images = vl_imreadjpeg(imfiles, 'Resize', [450, 450], 'Pack', 'CropLocation', 'center', 'CropSize',[1, 1], 'GPU', 'Flip', 'NumThreads', 8);
    else
        images = vl_imreadjpeg(imfiles, 'Resize', [450, 450], 'Pack', 'CropLocation', 'center', 'CropSize',[1, 1], 'Flip', 'NumThreads', 1);
    end
    
    images = images{1};
    avgclr = mean(mean(images(178:273,178:273,:,:),1),2);
    for idx = 1:length(imfiles)
        if rand > 0.5, images(:,:,:,idx)=flipud(images(:,:,:,idx)) ; end
        if rand > 0.5, images(:,:,:,idx)= rot90(images(:,:,:,idx)) ; end
        images(:,:,:,idx)=imrotate(images(:,:,:,idx), 11.5*(randi(8)-1), 'bilinear', 'crop'); 
    end
    labels = imdb.images.label(1,batch);
    images = bsxfun(@minus, images(114:337,114:337,:,:), avgclr);
else
    if opts.numGpus > 0
        images = vl_imreadjpeg(imfiles, 'Resize', [450, 450], 'Pack', 'CropLocation', 'center', 'CropSize',[1, 1], 'GPU', 'NumThreads', 12);
    else
        images = vl_imreadjpeg(imfiles, 'Resize', [450, 450], 'Pack', 'CropLocation', 'center', 'CropSize',[1, 1], 'NumThreads', 1);
    end
    
    images = images{1};
    labels = imdb.images.label(1,batch) ;
    avgclr = mean(mean(images(178:273,178:273,:,:),1),2);
    images = bsxfun(@minus, images(114:337,114:337,:,:), avgclr);
end
inputs = {'data', images, 'label', labels} ;
end


