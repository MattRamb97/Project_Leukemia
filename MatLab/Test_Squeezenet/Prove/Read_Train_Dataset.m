clear all;
clc;

% Numero di blocchi delle immagini realmente inseriti
totalImage = 1;

for i = 1 : 3
    % Lettura Fold(i) - All
    imgFiles = dir(strcat('/Users/matteorambaldi/MEGA/Tesi/Leukemia/C-NMC_training_data/fold_',int2str(i),'/all/*.bmp'));
    for ind = 1 : length(imgFiles)
        filename = convertCharsToStrings(imgFiles(ind).folder)+'/'+convertCharsToStrings(imgFiles(ind).name);
        tempImg = imread(filename);
        % Morphological open
        SE = strel('disk', 25);
        IM = imopen (tempImg, SE);
        % Suddivisione dell'immagine in blocchi 60x60x3 sovrapposti
        % Sovrapposizione di 30 per le immagini positive
        y=1;
        while (y~=421)
            x=1;
            while (x~=421)
                block = IM(x:x+59,y:y+59,:);
                % allblack = Funzione che ritorna se tutti gli elementi 
                % dell'immagine sono neri oppure no. 
                % Se = 0 -> tutto il blocco è nero
                % Se = 1 -> il blocco si può aggiungere
                if (allblack(block)==1)
                    % Resize di un fattore 2 per portare le immagini a 
                    % dimensione 30x30x3 per questione di memoria
                    DATA{1}(totalImage).im = imresize(block,0.5); 
                    % ALL -> class 1
                    DATA{2}(totalImage).labels = 1;
                    totalImage = totalImage + 1;
                end 
                x = x + 30;
            end
            y = y + 30;
        end
    end
    
    % Lettura Fold(i) - hem
    imgFiles = dir(strcat('/Users/matteorambaldi/MEGA/Tesi/Leukemia/C-NMC_training_data/fold_',int2str(i),'/hem/*.bmp'));
    for ind = 1 : length(imgFiles)
        filename = convertCharsToStrings(imgFiles(ind).folder)+'/'+convertCharsToStrings(imgFiles(ind).name);
        tempImg = imread(filename);
        % Morphological open
        SE = strel('disk', 25);
        IM = imopen (tempImg, SE);
        % Suddivisione dell'immagine in blocchi 60x60x3 sovrapposti
        % Sovrapposizione di 16 per le immagini positive
        y=1;
        while (y~=441)
            x=1;
            while (x~=441)
                % Colonna sx dell'immagine con x fissa a 397
                % e y che invece varia
                if (x==397) && (y~=397)
                    block = IM(x:450,y:y+59,:);
                    for p = 1:3
                        for k = 54:60
                            for j=1:60
                                block(k,j,p)=0;
                            end
                        end 
                    end
                else
                    if (y==397) && (x~=397)
                        % Riga down dell'immagine con y fissa a 397
                        % e x che invece varia
                        block = IM(x:x+59,y:450,:);
                        for p = 1:3
                            for k = 1:60
                                for j=54:60
                                    block(k,j,p)=0;
                                end
                            end 
                        end
                    else
                        if (y==397) && (x==397)
                            % Sia x che y a 397
                            % Ultimo blocco in basso a dx
                            block = IM(x:450,y:450,:);
                            for p = 1:3
                                for k = 54:60
                                    for j=54:60
                                        block(k,j,p)=0;
                                    end
                                end 
                            end
                        else
                            block = IM(x:x+59,y:y+59,:);
                        end
                    end
                end
                % allblack = Funzione che ritorna se tutti gli elementi 
                % dell'immagine sono neri oppure no. 
                % Se = 0 -> tutto il blocco è nero
                % Se = 1 -> il blocco si può aggiungere
                if (allblack(block)==1)
                    % Resize di un fattore 2 per portare le immagini a 
                    % dimensione 30x30x3 per questione di memoria
                    DATA{1}(totalImage).im = imresize(block,0.5);
                    % HEM -> class 0
                    DATA1{2}(totalImage) = 0;
                    totalImage = totalImage + 1;
                end
                x = x + 44;
            end
            y = y + 44;
        end
    end
end

save('BALL_Train.mat','DATA1','-v7.3');

