function predID = neighborhood_correction(estID, net)

    run('../vlfeat/toolbox/vl_setup');
    
    root = '../mio_code/Test_Squeezenet/fTest_Dataset_v2.mat';
    load(root,'DATA');
    
    NX = DATA{1}; % immagini

    opts.trainset = [0,1,2,3];
    opts.testset = -1;
    opts.whichresnet = 50;
    opts.numWords = 8 ;
    opts.numDescrsPerWord = 3000 ;
    
    testID = 2586; % numero di immagini totali
    layer = 'res5c_branch2a'; % layer per output
    feats = cell(1, testID); % cell vuoto di 2586 valori
    norml2 = @(x) bsxfun(@times, x, 1./(sqrt(sum(x.^2,1))+eps));

    if exist(sprintf('feats-res%s-cv%s_%s.mat', num2str(opts.whichresnet), num2str(opts.trainset), num2str(opts.testset)),'file')
        feats = load(sprintf('feats-res%s-cv%s_%s.mat', num2str(opts.whichresnet), num2str(opts.trainset), num2str(opts.testset)));
        feats = feats.feats;
    else
        for idx = 1:testID

            IM=double(NX(idx).im); %singola data immagine
            
            avgclr = mean(mean(IM(178:273,178:273,:,:),1),2);
            im_resized = bsxfun(@minus, images, avgclr);
            im_resized = cat(4, im_resized, fliplr(im_resized)); im_resized = cat(4, im_resized, flipud(im_resized));
            im_resized = cat(4, im_resized, rot90(im_resized));
            im_resized = cat(4, im_resized, imrotate(im_resized, 30, 'crop'), imrotate(im_resized, 60, 'crop'));
            im_resized = cat(4, im_resized, imrotate(im_resized, 10, 'crop'), imrotate(im_resized, 20, 'crop'));

            % single output 4D-single -> net.vars(i).value
            res = activations(net,im_resized,layer);
            feat = permute(gather(res), [3,1,2,4]);
            feats{idx} = norml2(reshape(feat,size(feat,1),[]));
        end
        save(sprintf('feats-res%s-cv%s_%s', num2str(opts.whichresnet), num2str(opts.trainset), num2str(opts.testset)), '-v7.3', 'feats');
    end

    FVENC = cell(1, testID);
    predID = estID;
    
    for epoch = 1:10
        
        % [MEANS, COVARIANCES, PRIORS] = vl_gmm(X, NumClusters);
        [MEANS, COVARIANCES, PRIORS] = vl_gmm(vl_colsubset(cat(2,feats{:}), opts.numWords*opts.numDescrsPerWord), opts.numWords,'Initialization', 'kmeans', 'CovarianceBound', 0.0001);
        
        for idx = 1:testID
            % FVENC{idx} = vl_fisher(X, MEANS, COVARIANCES, PRIORS);
            FVENC{idx} = vl_fisher(feats{idx}, MEANS, COVARIANCES, PRIORS, 'Improved');
        end
        
        % concatenazione a dimensione 2
        ENC = cat(2, FVENC{:});
        kernel = ENC' * ENC;
        
        for itr = 1:5
            fakeID = predID;
            
            % da 1 a 2586
            for subi = 1:numel(fakeID)
                
                % [simlar, subj] = maxk(A,k)
                % ritorna simlar come i 7 più grandi valori dal più grande 
                % al più piccolo trovati e subj i loro indici
                [simlar, subj] = maxk(kernel(subi,:), 7);
                
                % classi dei soggetti indicati dagli indici del kernel
                cls = fakeID(subj)';
                
                % se la somma delle classi è <=3 
                % 0 0 0 0 1 1 1 -> sum(cls) = 3, numero massimo per cui gli
                % zeri siano in maggioranza
                % e la classe della cellula del soggeto è 1 -> all, allora 
                % la classe della cellula è incorretta, e la imposto a 
                % 0 -> hem
                if sum(cls)<=3 && predID(subi) == 1
                    predID(subi) = 0;
                end
                
                % se la somma delle classi è >= 4 
                % 1 1 1 1 0 0 0 -> sum(cls) = 4, numero minimo per cui gli
                % uni siano in maggioranza
                % e la classe della cellula del soggeto è 0 -> hem, allora 
                % la classe è incorretta, e la imposto a 1
                if sum(cls)>=4 && predID(subi) == 0
                    predID(subi) = 1;
                end
                
                % altrimenti lascio la label com'è inizialmente
            end
        end
    end
end