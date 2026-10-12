function root = setup_reproduction(stage)
% Path/dependency checks for the official Widar3.0 sample.
root = fileparts(mfilename('fullpath'));
addpath(root);
addpath(fullfile(root,'csi_tool_box'));
addpath(fullfile(root,'tftb','mfiles'));
addpath(fullfile(root,'Doppler2VelocityMapping'));
if stage == 1
    required = {'butter','pca','tfrsp','tftb_window','read_bfee','csi_get_all'};
else
    required = {'fmincon','optimoptions','imrotate'};
end
for k = 1:numel(required)
    assert(~isempty(which(required{k})), ...
        'Missing function: %s. See README_CN.md for required toolboxes.',required{k});
end
end
