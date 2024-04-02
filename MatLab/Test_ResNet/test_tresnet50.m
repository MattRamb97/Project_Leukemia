function [ labels, scores ] = test_tresnet50(net)

    % carico set PHASE-III

    root = '../mio_code/Test_Squeezenet/fTest_Dataset_v2.mat';
    load(root,'DATA');

    % immagini
    NX = DATA{1};

    % labels tutte a 0, come vettore iniziale 'vuoto'
    labels = zeros(1,size(NX,2));
    
    % scores tutti a 0, come vettore iniziale 'vuoto'
    scores = zeros(2,size(NX,2));

    % indice della riga dello score
    line = 1;

    clear nome test testImages
    for i=1:size(NX,2)

            IM=double(NX(i).im); %singola data immagine
            
            avgclr = mean(mean(IM(178:273,178:273,:,:),1),2);
            
            % concatenazione a 4 dimensioni
            IM = cat(4, IM, fliplr(IM));
            IM = cat(4, IM, flipud(IM));
            IM = cat(4, IM, rot90(IM));
            IM = cat(4, IM, imrotate(IM, 30, 'crop'),...
                imrotate(IM, 60, 'crop'));
            IM = cat(4, IM, imrotate(IM, 10, 'crop'), ...
                imrotate(IM, 20, 'crop'));
            
            % [224 224 : :]
            IM = IM(114:337,114:337,:,:);
            
            IM = bsxfun(@minus, IM, avgclr);

            % Creo le 36 copie differenti della stessa immagine come descritto
            % nel paper per avere 36 labels differenti o non della stessa
            % immagine e la label iniziale è poi il majority voting tra le 36

            tempLabels = zeros(1,36);
            tempScr = zeros(2,36);
            angle = 0;
            for pattern=1:36
                images = imrotate(IM,angle,'crop');
                % Faccio classificare dalla rete le 36 copie
                [outclass,scr] = classify(net,images);
                tempLabels (pattern) = double(string(mode(outclass)));
                tempScr(line,pattern) = mean(scr(:,1));
                line = line + 1;
                tempScr(line,pattern) = mean(scr(:,2));
                
                % reset line score
                line = 1;
                angle = angle + 10;
            end

            % La label finale dell'immagine è la classe più votata tra le 36
            % copie
            labels(i) = double(string(mode(tempLabels)));
            scores(line,i) = mean(tempScr(1,:));
            line = line + 1;
            scores(line,i) = mean(tempScr(2,:));
            
            % reset line score
            line = 1;
    end  

end