% function net = loadModel(path_to_model_file)
%
%   Load kernels, bins, and network weights from trained model.
%
% path = Full path of trained caffemodel.h5 file
% 
%-- Ayan Chakrabarti <ayanc@ttic.edu>
function net = loadModel(mh5)

% Network Definition
layers = {{ 'conv1', 1, 1}, { 'conv2', 1, 1}, { 'conv3', 1, 1}, { 'conv4', 2, 1}, { 'conv5', 4, 1}, { 'conv6', 8, 1}, { 'conv7', 16, 1}, { 'conv8', 16, 1}, { 'conv9', 16, 1}, { 'pred0', 16, 0}, };

% Build struct with all details
net = struct;

k = squeeze(h5read(mh5,'/data/derFilt/0'));
net.numk = size(k,3);
k = k(end:-1:1,end:-1:1,:);
k = permute(k,[2 1 3]);

net.bins = -squeeze(h5read(mh5,'/data/fBins/1'));
net.nbins = length(net.bins) / net.numk;
net.bins = reshape(net.bins,[net.nbins net.numk]);

scales = squeeze(h5read(mh5,'/data/fBins/0'));
scales = scales(1:net.nbins:end);
scales = reshape(scales,[1 1 net.numk]);
net.k = bsxfun(@times,k,scales);

net.layers = {}; rsize = 1;
for i = 1:length(layers)
  l = layers{i};
  dil = l{2}; relu = l{3};
  wts = h5read(mh5,['/data/' l{1} '/0']);
  wts = permute(wts,[2 1 3 4]);
  bias = h5read(mh5,['/data/' l{1} '/1']);

  net.layers{end+1} = {wts, bias, dil, relu};
  rsize = rsize + (size(wts,1)-1)*dil;
end;

net.rsize = rsize;

%Global tensor
tmp=h5read(mh5,'/data/gusamp/0');
fac = size(tmp,1); fac = (fac+1) * 4;

b_w = ceil(560/fac)+1;
b_h = ceil(426/fac)+1;

gfip = h5read(mh5,'/data/gfip0/0');
nUnits = prod(size(gfip))/b_w/b_h;

gfip = reshape(gfip,[b_w b_h nUnits]);
gfip = permute(gfip,[2 1 3]);

cx = (b_w-1)*fac+1; cx = (cx-561)/2;
cy = (b_h-1)*fac+1; cy = (cy-427)/2;

net.glob = zeros([427,561,nUnits],'single');
for i = 1:nUnits
  us = interp2(gfip(:,:,i),log2(fac));
  net.glob(:,:,i) = us(1+cy:end-cy,1+cx:end-cx);
end;

% Move everything to gpu
for i = 1:length(net.layers)
  net.layers{i}{1} = gpuArray(single(net.layers{i}{1}));
  net.layers{i}{2} = gpuArray(single(net.layers{i}{2}));
end;
net.glob = gpuArray(single(net.glob));

%%%% Precompute things for consensus

%%% Choose regularizer
rfilt = [-1 2 -1];
rf1 = [0 0 0; rfilt; 0 0 0]; rf2 = diag(rfilt);
regf = cat(3,rf1,rf1',rf2,fliplr(rf2));

% set up sizes
filt_x = (size(net.k,1)-1)/2; % Filter size
tot_x = 2*filt_x; % Increase FFT size by this much

Zht = 427+2*tot_x;
Zwd = 561+2*tot_x;

% Pad, shift and DFT of filters
Kf = padarray(net.k,[Zht Zwd] - size(net.k,1),'post');
Kf = circshift(Kf,-filt_x*[1 1]);
Kf = gpuArray(single(Kf));

% Scale everything
ksc = sqrt(1./sum(sum(net.k.^2,1),2));

% Downweight 0th derivs
zdidx = min(min(net.k,[],1),[],2);
zdidx = find(zdidx >= 0);
ksc(zdidx) = ksc(zdidx)/4;

ksc = gpuArray(single(ksc));

Kf = bsxfun(@times,Kf,ksc);
Kf = fft2(Kf); Kfc = conj(Kf); Kfm = sum(Kf.*Kfc,3);

net.Kf = Kf; net.Kfc = Kfc; net.Kfm = Kfm;

% Add regularizer
Rf = padarray(regf,[Zht Zwd]-size(regf,1),'post');
Rf = circshift(Rf,-(size(regf,1)-1)/2*[1 1]);
Rf = gpuArray(single(Rf));
Rf = fft2(Rf); Rf = sum(Rf.*conj(Rf),3);

net.Rf = Rf;

% Move bin-centers to gpu
bins = net.bins'; bins = reshape(bins,[1 1 size(bins)]);
bins = gpuArray(single(bins));
net.bins = bsxfun(@times,bins,ksc);

% Fix last layer to give scaled log-likelihoods, permuted to be
% HxWxKxB
wts = net.layers{end}{1}; bias = net.layers{end}{2};
osz = size(wts);

wts = reshape(wts, [prod(osz)/net.numk/net.nbins ...
		    net.nbins net.numk]);
wts = permute(wts,[1 3 2]);
wts = reshape(wts,osz); net.layers{end}{1} = wts;

osz = size(bias);
bias = reshape(bias, [1 net.nbins net.numk]);
bias = permute(bias,[1 3 2]);
bias = reshape(bias,osz); 
net.layers{end}{2} = bias;

% Store multiplier for doForward output
net.ksc = -2*ksc.^2;
