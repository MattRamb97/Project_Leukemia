% fine-tune della ResNet50
[ tResNet50, acc_ptest ] = train_resnet50();

save('tResNet50.mat','tResNet50','-v7.3');
save('acc_ptest.mat','acc_ptest','-v7.3');

% prediction labels delle immagini di finally test
[ pLabels, scores ] = test_tresnet50(tResNet50);

save('pLabels.mat','pLabels','-v7.3');
save('scores.mat','scores','-v7.3');

[ fLabels ] = neighborhood_correction(pLabels, tResNet50);

save('fLabels.mat','fLabels','-v7.3');

% salvataggio del file per il contest
rsfl = fopen('isbi_valid.predict', 'w');
fprintf(rsfl,'%d \n', fLabels);
fclose(rsfl);
zip(sprintf('BM_%dres-cv%s', 50, '0123_4'), 'isbi_valid.predict');


