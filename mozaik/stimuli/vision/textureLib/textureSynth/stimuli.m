
close all

addpath("matlabPyrTools");
addpath("matlabPyrTools/MEX");
warning('off', 'Octave:possible-matlab-short-circuit-operator');

im0 = pgmRead('reptil_skin.pgm');	% im0 is a double float matrix!

Nsc = 4; % Number of scales     F: 4
Nor = 4; % Number of orientations       F: 4
Na = 9;  % Spatial neighborhood is Na x Na coefficients
	 % It must be an odd number!        ??

params = textureAnalysis(im0, Nsc, Nor, Na);

Niter = 25;	% Number of iterations of synthesis loop       F: 50
Nsx = 512;	% Size of synthetic image is Nsy x Nsx      F: 1280
Nsy = 384;	% WARNING: Both dimensions must be multiple of 2^(Nsc+2)    F: 960

res = textureSynthesis(params, [Nsy Nsx], Niter, [1, 1, 1, 0]);
resa = textureSynthesis(params, [Nsy Nsx], Niter, [1, 0, 0, 0]);

close all

figure(1)
showIm(im0, 'auto', 1, 'Original texture');
figure(2)
showIm(res, 'auto', 1, 'Naturalistic texture');
figure(3)
showIm(resa, 'auto', 1, 'Spectrally matched noise');


