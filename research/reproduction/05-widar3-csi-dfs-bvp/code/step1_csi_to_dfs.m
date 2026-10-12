% Run this file first. Uses the six CSI files bundled by the authors.
clearvars; close all; clc;
root = setup_reproduction(1);
sample = 'userA-1-1-1-1';
prefix = fullfile(root,'Data',sample);
outdir = fullfile(fileparts(root),'results',sample);
if ~exist(outdir,'dir'), mkdir(outdir); end
for r = 1:6
    assert(exist(sprintf('%s-r%d.dat',prefix,r),'file') == 2, ...
        'Missing receiver %d CSI file.',r);
    [x,~] = csi_get_all(sprintf('%s-r%d.dat',prefix,r));
    assert(~isempty(x) && size(x,2)==90 && all(isfinite(x(:))), ...
        'Receiver %d CSI is empty or invalid.',r);
    fprintf('Rx%d: %d packets, %d CSI channels\n',r,size(x,1),size(x,2));
    if r == 1, csi_rx1 = x; end
end
fprintf('Generating DFS with the official get_doppler_spectrum...\n');
tic;
[dfs_fft,freq_fft] = get_doppler_spectrum(prefix,6,3,'stft');
dfs_seconds = toc;
% The official function returns FFT order: [0:60,-60:-1].
% Sort BOTH bins and spectrum so their correspondence is preserved.
[freq_bin,order] = sort(freq_fft);
doppler_spectrum = dfs_fft(:,order,:);
assert(size(doppler_spectrum,1)==6 && size(doppler_spectrum,2)==121);
assert(all(isfinite(doppler_spectrum(:))), ...
    'DFS has NaN/Inf. Inspect the CSI and spectrum normalization.');
for r = 1:6
    y = squeeze(doppler_spectrum(r,:,:));
    assert(any(y(:)>0),'Receiver %d DFS is entirely zero.',r);
end
sample_rate_hz = 1000; % Official code assumes a nominal 1000 packets/s.
time_s = (0:size(doppler_spectrum,3)-1)/sample_rate_hz;
save(fullfile(outdir,'DFS.mat'),'doppler_spectrum','freq_bin', ...
    'time_s','sample_rate_hz','sample','dfs_seconds');
fig = figure('Color','w','Position',[100 100 1100 700]);
for r = 1:6
    subplot(2,3,r);
    imagesc(time_s,freq_bin,squeeze(doppler_spectrum(r,:,:)));
    axis xy; xlabel('Nominal time (s)'); ylabel('Doppler frequency (Hz)');
    title(sprintf('Receiver %d',r)); colorbar;
end
saveas(fig,fullfile(outdir,'DFS_6_receivers.png'));
fig = figure('Color','w');
plot((0:size(csi_rx1,1)-1)/sample_rate_hz,abs(csi_rx1(:,1)));
xlabel('Nominal time (s)'); ylabel('CSI amplitude');
title('Rx1: first antenna, first subcarrier'); grid on;
saveas(fig,fullfile(outdir,'CSI_amplitude_rx1.png'));
fprintf('DFS complete in %.1f s. Results: %s\n',dfs_seconds,outdir);
fprintf('Next: run step2_dfs_to_bvp.m\n');
