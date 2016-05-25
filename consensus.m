%function Z = consensus(pred,net)
%
% Global scene map estimation by consensus.
%   pred is from doForward
%   net is from loadModel
%
%-- Ayan Chakrabarti <ayanc@ttic.edu>
function Z = consensus(pred,net)

% Size and pad arithmetic
filt_x = (size(net.k,1)-1)/2; % Filter size
tot_x = 2*filt_x; % Increase FFT size by this much

Zht = size(pred,1)+2*tot_x;
Zwd = size(pred,2)+2*tot_x;

% Copy pre-computed FFTs from net
Kf = net.Kf; Kfc = net.Kfc; Kfm = net.Kfm; Rf = net.Rf;
bins = net.bins;

% Scale output probabilities
pred = bsxfun(@times,pred,net.ksc);

% Edge taper stuff
taper=15; % No of iterations to taper
tap1 = tot_x-filt_x; tap2 = filt_x; % to 0 and const
tcrop = tap1+tap2;
px = [linspace(0,1,tap1) ones(1,tap2)]; 
pnx = px(end:-1:1); py = px'; pny = pnx';

px = gpuArray(single(px)); py = gpuArray(single(py));
pnx = gpuArray(single(pnx)); pny = gpuArray(single(pny));

% Init X to argmin
X = zeros([Zht Zwd net.numk],'single','gpuArray');

postMAP(0,tot_x,size(pred,1),size(pred,2),net.numk,net.nbins);
X = fft2(X);

% Alternating minimization
for blog = [-10:0.125:7]
  beta = 2^blog;
  fprintf('\r beta = 2^(%.4f)   ',blog);
  Z = sum(X.*Kfc,3) ./ (Kfm+Rf/beta);

  % Taper edges in initial iterations to avoid ringing artifacts
  % from Fourier computation.
  if taper > 0
    Z = real(ifft2(Z));
    
    Z(1:tcrop,:) = bsxfun(@times,Z(tcrop+1,:),py);
    Z(end-tcrop+1:end,:) = bsxfun(@times,Z(end-tcrop,:),pny);
    Z(:,1:tcrop) = bsxfun(@times,Z(:,tcrop+1),px);
    Z(:,end-tcrop+1:end) = bsxfun(@times,Z(:,end-tcrop),pnx);

    Z = fft2(Z);
    taper = taper-1;
  end;
  
  X = real(ifft2(bsxfun(@times,Kf,Z)));
  
  postMAP(beta,tot_x,size(pred,1),size(pred,2),net.numk,net.nbins);
  X = fft2(X);

end;
Z = sum(X.*Kfc,3) ./ (Kfm+Rf/beta);

%%% Gather and Crop out center Z
Z = gather(real(ifft2(Z)));
Z = Z(1+tot_x:end-tot_x,1+tot_x:end-tot_x);

fprintf('\n');
