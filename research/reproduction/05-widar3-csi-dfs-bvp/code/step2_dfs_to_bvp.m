% Uses the official mapping, EMD + L0 objective, and sample geometry.
% Sequential loop avoids requiring Parallel Computing Toolbox.
clearvars; close all; clc;
root = setup_reproduction(2);
sample = 'userA-1-1-1-1';
outdir = fullfile(fileparts(root),'results',sample);
dfs_file = fullfile(outdir,'DFS.mat');
assert(exist(dfs_file,'file')==2,'Run step1_csi_to_dfs.m first.');
S = load(dfs_file);
doppler_spectrum = S.doppler_spectrum;
freq_bin = S.freq_bin;
assert(isequal(freq_bin,-60:60),'Expected sorted frequency bins -60:60.');

% These are from official DVM_main.m, for the bundled position 1,
% orientation 1 sample. They MUST change for a different deployment.
torso_pos = [1.365 0.455];
torso_orientation_deg = -90;
Tx_pos = [0 0];
Rx_pos = [0.455 -0.455; 1.365 -0.455; 2.0 0; ...
          -0.455 0.455; -0.455 1.365; 0 2.0];
rx_cnt = 6;
wave_length = 299792458/5.825e9;
seg_length = 100;
velocity_bin = (-9:10)*0.2; % Official grid: -1.8 to 2.0 m/s.
M = numel(velocity_bin);
lambda = 1e-7;
sparsity_norm = 0;
max_function_evaluations = 100000;
seg_number = floor(size(doppler_spectrum,3)/seg_length);
assert(seg_number>=1,'Not enough DFS snapshots for a 100-snapshot segment.');
A = get_A_matrix(torso_pos,Tx_pos,Rx_pos,rx_cnt);
VDM = permute(get_velocity2doppler_mapping_matrix(A,wave_length, ...
    velocity_bin,freq_bin,rx_cnt),[2 3 1 4]);
U_bound = repmat(max(doppler_spectrum(:)),M,M);
velocity_spectrum = zeros(M,M,seg_number);
exit_flags = zeros(seg_number,1);
objectives = zeros(seg_number,1);
segment_seconds = zeros(seg_number,1);
function_counts = zeros(seg_number,1);
options = optimoptions('fmincon','Algorithm','sqp', ...
    'MaxFunctionEvaluations',max_function_evaluations,'Display','off');
fprintf('Computing %d BVP segments. Each segment is nominally 100 ms.\n',seg_number);
fprintf('The official fmincon objective can take substantial CPU time.\n');
tic;
for ii = 1:seg_number
    target = mean(doppler_spectrum(:,:,(ii-1)*seg_length+1:ii*seg_length),3);
    assert(all(isfinite(target(:))) && all(sum(target,2)>0), ...
        'Segment %d has invalid or missing-link DFS.',ii);
    for r = 2:rx_cnt
        target(r,:) = target(r,:)*sum(target(1,:))/sum(target(r,:));
    end
    segment_timer = tic;
    [P,fval,flag,info] = fmincon(@(P)DVM_target_func(P,VDM,lambda, ...
        target,rx_cnt,sparsity_norm),zeros(M,M),[],[],[],[], ...
        zeros(M,M),U_bound,[],options);
    assert(all(isfinite(P(:))),'Invalid solution for segment %d.',ii);
    velocity_spectrum(:,:,ii) = P;
    exit_flags(ii) = flag;
    objectives(ii) = fval;
    segment_seconds(ii) = toc(segment_timer);
    function_counts(ii) = info.funcCount;
    fprintf('Segment %d/%d: exit flag %d, objective %.6g, %.1f s\n', ...
        ii,seg_number,flag,fval,segment_seconds(ii));
    if flag<=0
        warning('Segment %d: solver did not report successful convergence.',ii);
    end
end
bvp_seconds = toc;
velocity_spectrum_ro = get_rotated_spectrum(velocity_spectrum,torso_orientation_deg);
segment_time_s = ((0:seg_number-1)*seg_length+(seg_length-1)/2)/S.sample_rate_hz;
save(fullfile(outdir,'BVP.mat'),'velocity_spectrum_ro','velocity_spectrum', ...
    'velocity_bin','segment_time_s','torso_pos','torso_orientation_deg', ...
    'Tx_pos','Rx_pos','A','wave_length','seg_length','lambda','sparsity_norm', ...
    'exit_flags','objectives','segment_seconds','function_counts','bvp_seconds','sample');
solver_log = table((1:seg_number)',exit_flags,objectives,segment_seconds,function_counts, ...
    'VariableNames',{'segment','exit_flag','objective','seconds','function_count'});
writetable(solver_log,fullfile(outdir,'solver_log.csv'));
% Dimension 1 is vx; dimension 2 is vy. Transpose for imagesc.
selected = unique(round(linspace(1,seg_number,min(6,seg_number))));
fig = figure('Color','w','Position',[100 100 1100 700]);
for k = 1:numel(selected)
    ii = selected(k);
    subplot(2,3,k);
    imagesc(velocity_bin,velocity_bin,velocity_spectrum_ro(:,:,ii).');
    axis xy equal tight; xlabel('Body vx (m/s)'); ylabel('Body vy (m/s)');
    title(sprintf('Segment %d, t = %.2f s',ii,segment_time_s(ii))); colorbar;
end
saveas(fig,fullfile(outdir,'BVP_snapshots.png'));
fprintf('BVP complete in %.1f s. Results: %s\n',bvp_seconds,outdir);
fprintf('Check solver_log.csv: saved arrays alone do not prove convergence.\n');
