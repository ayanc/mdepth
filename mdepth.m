%function Z = mdepth(im,net)
%
% Global scene map estimation by network prediction and consensus.
%   im is RGB image (normalized b/w 0-1, e.g., by im2single)
%   net is from loadModel
%   Output Z is depth map.
% 
%-- Ayan Chakrabarti <ayanc@ttic.edu>
function Z = mdepth(im,net)

fprintf('Running network on all patches ....\n');
pred = doForward(im,net);
fprintf('Globalization ....\n');
Z = consensus(pred,net);
Z = 1./max(0.1,Z);
fprintf('Done! \n');
