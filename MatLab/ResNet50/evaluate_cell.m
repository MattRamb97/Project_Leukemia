function [labels, scores, predlabel] = evaluate_cell(varargin)

% MODIFICATO flag per utilizzo della gpu
useGPU = 0;

opts.rootPath = '../data/';
opts.trainset = [1,2,3];
opts.testset = 0;
opts.whichresnet = 50;

% aggiorna la struttura opts in base alle coppie parametro-valore specificate
% da varargin
[opts, varargin] = vl_argparse(opts, varargin) ;

% expDir - > '../data/models/cell50res-0 1 2 3'
% serve per recuperare il file imdb.mat 
opts.expDir = fullfile(opts.rootPath, 'models/', ...
    sprintf('cell%dres-%s', opts.whichresnet, num2str(opts.trainset))) ;

% MODIFICATO
% sostituisco per modelpath la funzione sprintf con fullfile, commentando 
% la riga della prima funzione citata
% modelpath -> '../data/models/imagenet-resnet-50-dag.mat'
opts.modelpath =  fullfile(opts.rootPath, 'models/', ...
    sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet));
%opts.modelpath =  sprintf('imagenet-resnet-%d-dag.mat', opts.whichresnet);

% MODIFICATO
% tolgo il simbolo '/' da '/cells' per correttezza della root della directory

% '.../data/cells'
opts.dataDir = fullfile(opts.rootPath, 'cells');

% '.../data/models/cell50res-0  1  2  3/imdb.mat'
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.train = struct('gpus', useGPU) ;
opts = vl_argparse(opts, varargin) ;

if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end

% load della struttura imdb con classes e images
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
else
    imdb = cell_get_database(opts);
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb');
end

% Richiamo della ResNet alla 30esima epoca, dopo l'addestramento
net = load(fullfile(opts.expDir,'net-epoch-30.mat'));
net = dagnn.DagNN.loadobj(net.net);

% MODIFICATO
% Inizialmente move era impostato a 'gpu', inserisco 'cpu' perchè non
% supporto l'utilizzo della gpu: per mio test
if useGPU == 0
    net.move('cpu');
else 
    net.move('gpu');
end

net.mode = 'test' ;

% ID solo delle immagini di test (totale: 2586 valori)
testID = find(ismember(imdb.images.set, opts.testset));

% array di 2586 valori iniziali posti a 0
predlabel = zeros(1, numel(testID));

% array di 2586 valori iniziali posti a 0
scores = zeros(1, numel(testID));

% sarebbero 15114 labels, ma prendo solo le 2586 indicate da testID
labels = imdb.images.label(testID);

for idx = 1:numel(testID)
    
    % '.../data/cells/phase_3/*.bmp'
    flnm = fullfile(imdb.imageDir, imdb.images.name{testID(idx)});
    
    % MODIFICATO
    % Aggiunto per modificare i link alle directory
    flnm = replace(flnm,'\','/');

    images = single(imread(flnm));
    if size(images,1) == 600
        images = images(76:525,76:525,:);
    end
    avgclr = mean(mean(images(178:273,178:273,:,:),1),2);
    
    % crea immagini a 4 dimensioni
    images = cat(4, images, fliplr(images));
    images = cat(4, images, flipud(images));
    images = cat(4, images, rot90(images));
    images = cat(4, images, imrotate(images, 30, 'crop'), imrotate(images, 60, 'crop'));
    images = cat(4, images, imrotate(images, 10, 'crop'), imrotate(images, 20, 'crop'));
%     images =  cat(4, images, imrotate(images, 45, 'crop'));
%     images =  cat(4, images, imrotate(images, 22.5, 'crop'));
%     images =  cat(4, images, imrotate(images, 11.5, 'crop'));
    im_crop = images(114:337, 114:337,:,:);
    
    % applica l'operatore binario alle due matrici, in questo caso minus
    % per la differenza
    im_crop = bsxfun(@minus, im_crop, avgclr);
    
    
    % MODIFICATO
    % All'inizio era 'inputs = {'data', gpuArray(im_crop)}', tolgo tale
    % funzione perchè non posso utilizzare la gpu, così da averlo
    % direttamente in memoria locale: per mio test
    if useGPU == 0
        inputs = {'data', im_crop} ;
    else 
        inputs = {'data', gpuArray(im_crop)} ;
    end
    
    net.eval(inputs) ;
    scr = net.vars(net.getVarIndex('prob')).value ;
    scr = squeeze(gather(scr));
    pID = mean(scr,2);
    pID(1) = single(pID(1) > 40/101);
    predlabel(idx) = 2-pID(1);
    scores(idx) = pID(2);
    if imdb.images.label(testID(idx)) ~= 2-pID(1)
        fprintf('%s %d %d.\n', imdb.images.name{testID(idx)}, 2-imdb.images.label(testID(idx)), pID(1));
    end
end

confmat = full(sparse(labels', predlabel, 1, 2, 2));
precision = diag(confmat)./sum(confmat,2);
recall = diag(confmat)./sum(confmat,1)';
f1Scores =  2*(precision.*recall)./(precision+recall);
meanF1 = mean(f1Scores);
confmat = bsxfun(@times, confmat, 1./max(sum(confmat,2),eps));
bacc = mean(diag(confmat));
fprintf('meanF1=%f, bacc=%f.\n', [meanF1, bacc]);
end

