%% Lettura e creazione Train_Dataset dalla cartella C-NMC_training_data

clear;
clc;
totalImage = 1;

for i = 0 : 2
    % Lettura Fold(i) - All
    imgFiles = dir(strcat('/Users/matteorambaldi/MEGA/Tesi/Leukemia/C-NMC_training_data/fold_',int2str(i),'/all/*.bmp'));
    for ind = 1 : length(imgFiles)
        filename = convertCharsToStrings(imgFiles(ind).folder)+'/'+convertCharsToStrings(imgFiles(ind).name);
        tempImg = imread(filename);
        DATA{1}(totalImage).im = tempImg; %(totalImage).im
        % ALL -> class 1
        DATA{2}(totalImage) = 1; %.labels
        totalImage = totalImage + 1;
    end
    
    % Lettura Fold(i) - hem
    imgFiles = dir(strcat('/Users/matteorambaldi/MEGA/Tesi/Leukemia/C-NMC_training_data/fold_',int2str(i),'/hem/*.bmp'));
    for ind = 1 : length(imgFiles)
        filename = convertCharsToStrings(imgFiles(ind).folder)+'/'+convertCharsToStrings(imgFiles(ind).name);
        tempImg = imread(filename);
        DATA{1}(totalImage).im = tempImg;
        % HEM -> class 0
        DATA{2}(totalImage) = 0;
        totalImage = totalImage + 1;
    end
end

save('Train_Dataset.mat','DATA','-v7.3');


%% Lettura e creazione pTest_Dataset dalla cartella C-NMC_test_prelim_phase_data

clear;
clc;
totalImage = 1;

% Lettura del file .csv tramite funzione csvimport che crea un'unica
% matrice colonna con all'interno stringhe composte da:
% 'Patient_ID;new_names;labels'
% Options 'delimiter' -> ';' permette di dividere ciascuna componente
% creando quindi 3 colonne contententi ciascuna i valori indicati
% precedentemente

[X] = csvimport('/Users/matteorambaldi/MEGA/Tesi/Leukemia/C-NMC_test_prelim_phase_data/C-NMC_test_prelim_phase_data_labels_v2.csv', 'delimiter',';');

% Faccio partire l'indice da 2 perchè la prima riga contiene i nomi delle
% colonne (spiegazione anche per il -1)
for ind = 2 : size(X,1)
    filename = convertCharsToStrings(strcat('/Users/matteorambaldi/MEGA/Tesi/Leukemia/C-NMC_test_prelim_phase_data/C-NMC_test_prelim_phase_data/',X(ind,2)));
    tempImg = imread(filename);
    % 1° colonna -> im
    DATA{1}(totalImage).im = tempImg;
    % 2° colonna -> patient_ID
    DATA{2}(totalImage) = X(ind,1);
    % 3° colonna -> new_names
    DATA{3}(totalImage) = X(ind,2);
    % 4° colonna -> labels
    DATA{4}(totalImage) = convertCharsToStrings(X(ind,3));
    totalImage = totalImage + 1;
end

save('pTest_Dataset.mat','DATA','-v7.3');


%% Lettura e creazione fTest_Dataset dalla cartella C-NMC_test_final_phase_data

clear;
clc;
totalImage = 1;

% prova 2
imgFiles = dir(strcat('/Users/matteorambaldi/Desktop/C_NMC/C-NMC_test_final_phase_data/C-NMC_test_final_phase_data/*.bmp'));
imgFiles = natsortfiles(imgFiles);
% prova I
%imgFiles = dir(strcat('/Users/matteorambaldi/MEGA/Tesi/Leukemia/C-NMC_test_final_phase_data/C-NMC_test_final_phase_data/*.bmp'));

for ind = 1 : length(imgFiles)
    filename = convertCharsToStrings(imgFiles(ind).folder)+'/'+convertCharsToStrings(imgFiles(ind).name);
    tempImg = imread(filename);
    tempImg = imresize(tempImg, [450 450]);
    DATA{1}(totalImage).im = tempImg; 
    % Le labels non sono già presenti, le dovrà calcolare poi la rete
    % neurale 
    DATA{2}(totalImage).name = imgFiles(ind).name;
    totalImage = totalImage + 1;
end

% provaI
%save('fTest_Dataset.mat','DATA','-v7.3');

% prova II
save('fTest_Dataset_v2.mat','DATA','-v7.3');

