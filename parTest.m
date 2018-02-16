poolobj = parpool(15);

parfor ii = 1:10
    
    
    d = clock;
    fprintf('Test Loop %u: %u%.2u%.2u%.2u%.2u%.2f\n',ii,d(1:6));
    
    
end
delete(poolobj);
d = clock;
datastr = sprintf('./ResultsNClevel%u_%u%.2u%.2u%.2u%.2u.mat',10,d(1:5));
save(datastr,'d');