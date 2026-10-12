% Save this file in Widar3.0Release-Matlab, then run compare_bvp.
% Compare to the CANDIDATE reference supplied in the author's code archive.
% IMPORTANT: reference prefix is 'user', raw example prefix is 'userA'.
% Equal dimensions alone do not establish that they are the same recording.
clearvars; close all; clc;
root = fileparts(mfilename('fullpath'));
outdir = fullfile(fileparts(root),'results','userA-1-1-1-1');
ours_file = fullfile(outdir,'BVP.mat');
reference_file = fullfile(root,'BVP', ...
    'user-user-1-1-1-1-1-1e-07-100-20-100000-L0.mat');
assert(exist(ours_file,'file')==2, ...
    'Cannot find your BVP.mat. Put compare_bvp.m in the project root and run step2 first.');
assert(exist(reference_file,'file')==2,'Cannot find the bundled candidate reference BVP.');
O = load(ours_file);
R = load(reference_file);
assert(isfield(O,'velocity_spectrum_ro') && isfield(R,'velocity_spectrum_ro'), ...
    'Both files must contain velocity_spectrum_ro.');
ours = double(O.velocity_spectrum_ro);
reference = double(R.velocity_spectrum_ro);
assert(isequal(size(ours),size(reference)), ...
    'BVP dimensions differ. Do not crop or interpolate before checking sample/segment alignment.');
assert(size(ours,1)==20 && size(ours,2)==20);
assert(all(isfinite(ours(:))) && all(isfinite(reference(:))), ...
    'BVP contains NaN/Inf.');
assert(all(ours(:)>=0) && all(reference(:)>=0),'BVP should be nonnegative.');
velocity_bin = (-9:10)*0.2;
assert(isfield(O,'velocity_bin') && ...
    max(abs(O.velocity_bin(:)-velocity_bin(:)))<1e-10, ...
    'Your velocity grid does not match the official example settings.');
assert(isfield(O,'seg_length') && O.seg_length==100, ...
    'Your segment length differs from the candidate reference filename.');
T = size(ours,3);
nrmse = nan(T,1);
cosine_similarity = nan(T,1);
normalized_overlap = nan(T,1);
ours_total_power = zeros(T,1);
reference_total_power = zeros(T,1);
for k = 1:T
    a = reshape(ours(:,:,k),[],1);
    b = reshape(reference(:,:,k),[],1);
    ours_total_power(k) = sum(a);
    reference_total_power(k) = sum(b);
    if norm(b)>0
        nrmse(k) = norm(a-b)/norm(b);
    end
    if norm(a)>0 && norm(b)>0
        cosine_similarity(k) = max(0,min(1,(a'*b)/(norm(a)*norm(b))));
    end
    if sum(a)>0 && sum(b)>0
        normalized_overlap(k) = sum(min(a/sum(a),b/sum(b)));
    end
end
metrics = table((1:T)',nrmse,cosine_similarity,normalized_overlap, ...
    ours_total_power,reference_total_power, ...
    'VariableNames',{'segment','relative_L2_error','cosine_similarity', ...
    'normalized_overlap','ours_total_power','reference_total_power'});
disp(metrics);
fprintf('\nRaw relative L2 error (all frames): %.6g\n', ...
    norm(ours(:)-reference(:))/max(norm(reference(:)),eps));
valid = isfinite(cosine_similarity);
fprintf('Mean cosine similarity: %.6g\n',mean(cosine_similarity(valid)));
valid = isfinite(normalized_overlap);
fprintf('Mean normalized overlap: %.6g\n',mean(normalized_overlap(valid)));
fprintf('Relative L2 error: closer to 0 = closer raw values.\n');
fprintf('Cosine/normalized overlap: closer to 1 = closer spatial distribution.\n');
fprintf('NaN means the metric was undefined (for example, a zero-power frame).\n');
fprintf('These are descriptive comparisons, with no automatic pass/fail threshold.\n');
fprintf('\nCAUTION: sample identity and time alignment are not confirmed.\n');
fprintf('Bundled reference prefix: user; CSI example prefix: userA.\n');
fprintf('Check sample identity, geometry and orientation before claiming reproduction accuracy.\n');
writetable(metrics,fullfile(outdir,'BVP_comparison_metrics.csv'));

% All raw BVP panels share one scale; difference panels share another.
selected = unique(round(linspace(1,T,min(4,T))));
common_max = max([ours(:);reference(:)]);
if common_max==0, common_max=1; end
difference = abs(ours-reference);
diff_max = max(difference(:));
if diff_max==0, diff_max=1; end
fig = figure('Color','w','Position',[80 40 1200 1000]);
for row = 1:numel(selected)
    k = selected(row);
    panels = {reference(:,:,k),ours(:,:,k),difference(:,:,k)};
    labels = {'Bundled candidate reference','Your reproduction','Absolute difference'};
    for col = 1:3
        subplot(numel(selected),3,(row-1)*3+col);
        imagesc(velocity_bin,velocity_bin,panels{col}.');
        axis xy equal tight;
        xlabel('Body vx (m/s)'); ylabel('Body vy (m/s)');
        title(sprintf('%s | segment %d',labels{col},k));
        colorbar;
        if col<=2, caxis([0 common_max]); else, caxis([0 diff_max]); end
        ax = gca;
        if isprop(ax,'Toolbar'), ax.Toolbar.Visible='off'; end
    end
end
colormap(fig,parula);
saveas(fig,fullfile(outdir,'BVP_reference_comparison.png'));

fig = figure('Color','w','Position',[100 100 900 650]);
subplot(2,1,1);
plot(1:T,nrmse,'o-'); grid on;
xlabel('Segment'); ylabel('Relative L2 error');
title('Raw values: lower error means closer values');
subplot(2,1,2);
plot(1:T,cosine_similarity,'o-',1:T,normalized_overlap,'s-');
grid on; ylim([0 1.05]); xlabel('Segment'); ylabel('Similarity');
legend('Cosine similarity','Normalized overlap','Location','best');
title('Spatial distributions: values closer to 1 mean greater similarity');
axes_list = findall(fig,'Type','axes');
for k = 1:numel(axes_list)
    if isprop(axes_list(k),'Toolbar'), axes_list(k).Toolbar.Visible='off'; end
end
saveas(fig,fullfile(outdir,'BVP_comparison_metrics.png'));
fprintf('\nComparison images and CSV saved to: %s\n',outdir);
