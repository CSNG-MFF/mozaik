function [im] = textureBasedStimulus(impath, stats, seed, sizex, sizey, libpath)

% Outputs a frame of texture generated from original image
% (application of Portilla-Simoncelli algorithm)
%
% [im] = textureBasedStimulus(impath, stats);
% 	impath: 	string containing path to initial texture
% 	stats: integer 0 - 2 specifying the statistics matched in resulting stimulus
%       0 - original image
%       1 - naturalistic texture image (matched higher order statistics)
%       2 - spectrally matched noise (matched marginal statistics only)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(strcat(libpath, "/textureSynth"));
addpath(strcat(libpath, "/textureSynth/matlabPyrTools"));
addpath(strcat(libpath, "/textureSynth/matlabPyrTools/MEX"));
warning('off', 'Octave:possible-matlab-short-circuit-operator');

im0 = pgmRead(impath);	% im0 is a double float matrix!
Nsc = 4; % Number of scales (Freeman: 4)
Nor = 4; % Number of orientations (Freeman: 4)
Na = 9;  % Spatial neighborhood is Na x Na coefficients
	 % It must be an odd number! (Freeman: ??)

if stats == 0
        im = im0;
else
    params = textureAnalysis(im0, Nsc, Nor, Na);
    if stats == 1
            cmask = [1, 1, 1, 0];
    else %stats == 2
            cmask = [1, 0, 0, 0];
    end
    Niter = 25;	% Number of iterations of synthesis loop (Freeman: 50)
    Nsx = 256;	% Size of synthetic image is Nsy x Nsx (Freeman: 1280)
    Nsy = 256;	% WARNING: Both dimensions must be multiple of 2^(Nsc+2) (Freeman: 960)
    if (exist('seed') && ~isempty(seed) )
        im = textureSynthesis(params, [Nsy Nsx, seed], Niter, cmask);
    else
        im = textureSynthesis(params, [Nsy Nsx], Niter, cmask);
    end
end
im = im(1:sizex, 1:sizey);


