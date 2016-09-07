% function pred = doForward(img,net)
%
% Forward pass through image. img should be an RGB image normalized
% to [0,1]. net is the network loaded using loadModel
%
% Define a global variable MAX_SPACE to adjust size of temporary
% workspace used in convolutions. Value should be number of 32-bit
% floats (i.e., total size in bytes divided by 8). Default is 2GB,
% i.e., MAX_SPACE=1024^3 / 4;
%
%-- Ayan Chakrabarti <ayanc@ttic.edu>
function act = doForward(img,net)

glob = doVGG(img,net);

img = gpuArray(single(img));
act = img*2-1;

for i = 1:length(net.layers)
  fprintf('\r--- Layer %02d of %02d     ',i,length(net.layers));
  l = net.layers{i};
  
  pad = (size(l{1},1)-1)*l{3}/2;
  if pad > 0
    if i == 0
      act = padarray(act,[pad pad],'replicate','both');
    else
      act = padarray(act,[pad pad],0,'both');
    end;
  end;

  if i > 1
    if size(act,3) < size(l{1},3)
      act = cat(3,act,glob); clear glob;
    end;
  end;
  
  act = vConv(act,l{1},l{2},l{3},l{4});

end;
fprintf('\n');
act = reshape(act,[size(act,1) size(act,2) net.numk net.nbins]);

%%%%%%%%%%%%%%%%%%%%
% Do VGG forward pass
function glob = doVGG(img,net)

img = double(img);

img = img(22:end-22,25:end-25,:);

img = permute(img,[2 1 3]); img = img(:,:,end:-1:1);
img = img*255;
img = bsxfun(@minus,img, ...
	     reshape([103.939 116.779 123.68],[1 1 3]));

act = gpuArray(single(img));


% Do all the conv layers
idx = 1;
for i = 1:length(net.vconvs)
  for j = 1:net.vconvs(i)
    fprintf('\r--- Layer %d,%d         ',i,j);
    l = net.vlayers{idx}; idx = idx+1;
  
    pad = (size(l{1},1)-1)/2;
    if pad > 0
      act = padarray(act,[pad pad],0,'both');
    end;
    act = vConv(act,l{1},l{2},1,1);
  end;
  act0 = max(act(1:2:end,:,:),act(2:2:end,:,:));
  act = max(act0(:,1:2:end,:),act0(:,2:2:end,:));
end;
fprintf('\n');

act0 = act(1:2:end,1:2:end,:)+act(1:2:end,2:2:end,:)+...
       act(2:2:end,1:2:end,:)+act(2:2:end,2:2:end,:);
act = act0(:)/4;
act = max(0,net.vgg_fc1{1}*act + net.vgg_fc1{2});

act = net.vgg_gfp{1}*act + net.vgg_gfp{2};

bw = net.gsz(1); bh = net.gsz(2); 
fac = net.gsz(4); nUnits = net.gsz(3);

act = reshape(act,[bw bh nUnits]);
act = permute(act,[2 1 3]);

cx = (bw-1)*fac+1; cx = (cx-561)/2;
cy = (bh-1)*fac+1; cy = (cy-427)/2;

glob = zeros([427,561,nUnits],'single','gpuArray');
for i = 1:nUnits
  us = interp2(act(:,:,i),log2(fac));
  glob(:,:,i) = us(1+cy:end-cy,1+cx:end-cx);
end;


%%%%%%%%%%%%%%%%%%%%
% Conv layer forward
function out = vConv(in,wts,bias,dil,relu)

% Define a global variable MAX_SPACE to adjust memory usage.
global MAX_SPACE;
if length(MAX_SPACE) == 0
  clear MAX_SPACE
  MAX_SPACE=2 * 1024^3 / 8; % Default is 2GB
end;

[H,W,C] = size(in);
[K1,K2,~,C2] = size(wts);

wts = gpuArray(single(wts)); bias = gpuArray(single(bias));

% Check if its simply a 1x1 conv
if K1 == 1 && K2 == 1
  in = reshape(in,[H*W C]); 
  bias = reshape(bias,[1 C2]);
  wts = reshape(wts,[C C2]);
  out = bsxfun(@plus,in*wts,bias);
  out = reshape(out,[H W C2]);
  if relu == 1
    out = max(0,out);
  end;
  clear wts bias
  return
end;

% build offsets
[dy,dx,dc] = ndgrid([0:(K1-1)]*dil,[0:(K2-1)]*dil,[0:(C-1)]);
d_idx = dy(:)' + dx(:)'*H + dc(:)'*H*W;

% build patch top-lefts
K1eq = (K1-1)*dil+1; K2eq = (K2-1)*dil+1;
[y,x] = ndgrid([1:(H-K1eq+1)],[0:(W-K2eq)]);
p_tl = y(:) + x(:)*H;

out = zeros(length(p_tl),C2,'single','gpuArray');
d_idx = gpuArray(single(d_idx));
p_tl = gpuArray(single(p_tl));

wts = reshape(wts,[prod(size(wts))/C2 C2]);
bias = reshape(bias,[1 C2]);

skip = max(1,round(MAX_SPACE / length(d_idx(:))));

for i = 1:skip:length(p_tl)
  idx = [i:min(i+skip-1,length(p_tl))]';
  im2c = in(bsxfun(@plus,p_tl(idx),d_idx));
  out(idx,:) = bsxfun(@plus,im2c*wts,bias);
end;

out = reshape(out,[(H-K1eq+1) (W-K2eq+1) C2]);
if relu == 1
    out = max(0,out);
end;
clear wts bias
