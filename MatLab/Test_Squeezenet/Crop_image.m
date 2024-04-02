% Funzione con cui ho ridimensionato le immagini della fase finale di test

imgFiles = dir(strcat('/Users/matteorambaldi/MEGA/Tesi/Leukemia/C-NMC_test_final_phase_data/C-NMC_test_final_phase_data/*.bmp'));
for ind = 1 : size(imgFiles, 1)
    filename = convertCharsToStrings(imgFiles(ind).folder)+'/'+convertCharsToStrings(imgFiles(ind).name);
    tempImg = imread(filename);
    newImg = imcrop(tempImg, [74 74 449 449]); 
    f = fullfile('/Users/matteorambaldi/MEGA/Tesi/Leukemia/data/cells/phase_3/',convertCharsToStrings(imgFiles(ind).name));
    imwrite(newImg, f);
end