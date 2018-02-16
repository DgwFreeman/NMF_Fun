parpool(5)

parfor ii = 1:10
    
    
    d = clock;
    fprintf('Test Loop %u: %u%.2u%.2u%.2u%.2u%.2f\n',ii,d(1:6));
    
    
end
d = clock;
datastr = sprintf('./ResultsNClevel%u_%u%.2u%.2u%.2u%.2u.mat',int16(fCoding(iC)*100),d(1:5));
save(datastr,'ii');