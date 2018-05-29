%  SVDRANKONEUPDATE   Compute singular value decomposition of a rank-1
%  perturbation of a matrix with known singular value decomposition. Given
%  a matrix with known singular value decomposition, A = U*E*V',
%  SVDRANKONEUPDATE update efficiently calculates the SVD of the perturbed
%  matrix, Ap = (A + rho*a*b'), where rho is some scalar and a and b are
%  vectors.
%
%  [Unew, Snew, Vnew, stats] = svdRankOneUpdate(U, S, V, a, b, rho, acc) 
%  returns the SVD of Ap = Unew*diag(Snew)*Vnew'. 
%
%  For details on computing eigenvectors, see
%  "On the Efficient Update of the Singular Value Decomposition Subject to
%   Rank-One Modifications", P. Stange, Proc. Appl. Math. Mech.,
%   8: 10827–10828. doi:10.1002/pamm.200810827
%
%  version 1.8
%  Gautam Kunapuli (gkunapuli@gmail.com)
%  May 05, 2018
%
%  This program comes with ABSOLUTELY NO WARRANTY; See the GNU General 
%  Public License for more details. This is free software, and you are 
%  welcome to modify or redistribute it.

function [Unew, Snew, Vnew, stats] = svdRankOneUpdate(U, S, V, a, b, rho, acc, debug)
if nargin < 8
    debug = false;
end

% Set an internal accuracy
if nargin < 7
    acc = 1e-12;
end

% Start timer
svdTimer = tic;

% Get the dimensions of the matrix A or S. If N > M, then solve the
% transposed problem, and then transpose the solution at the end.
[M, N] = size(S);
if N > M
    warning(['Solving transposed problem! If you''re planning on calling ',...
             'this function multiple times, consider transposing the '...
             'problem setting for efficiency.']);
    
    swapped = true;
    S = S';
    [U, V] = swap(U, V);
    [a, b] = swap(a, b);
    [M, N] = size(S);
else
    swapped = false;
end

% Absorb rho into b
% b = rho * b;
a = rho * a;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% RANK-TWO update of V (left singular vectors) and E = S'S
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

E = diag(S'*S);                      % Eigenvalues S'*S
aTilde = V*S'*U'*a;                  % Compute ai and bi
[Qi, Ri] = schur([a'*a, 1; 1, 0]);
u = [b, aTilde]*Qi;
ai = u(:, 1);
bi = u(:, 2);

[V1, E1] = eigRankOneUpdate(V, E, V'*ai, Ri(1, 1), acc);
[V2, E2] = eigRankOneUpdate(V1, E1, V1'*bi, Ri(2, 2), acc);

if debug
    % Check the computations with MATLAB outputs
    A = V*diag(E)*V' + Ri(1, 1)*(ai*ai');
    A = (A + A') / 2;
    [V1m, E1m] = eig(A);
    err = norm(V1m*E1m*V1m' - V1*diag(E1)*V1');
    fprintf('V, E RANK-2 Step 1: || V1m E1m V1m'' - V1 E1 V1'' || = %g.\n', err);
    if err > sqrt(acc)
        fprintf('Check error!\n');
    end
    
    A = V*diag(E)*V' + Ri(1, 1)*(ai*ai') + Ri(2, 2)*(bi*bi');
    A = (A + A') / 2;
    [V2m, E2m] = eig(A);
    err = norm(V2m*E2m*V2m' - V2*diag(E2)*V2');
    fprintf('V, E RANK-2 Step 2: || V2m E2m V2m'' - V2 E2 V2'' || = %g.\n', err);
    if err > sqrt(acc)
        fprintf('Check error!\n');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% RANK-TWO update of U (right singular vectors) and D = SS'
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = S*S';
bTilde = U*S*V'*b;                      % Compute ao and bo
[Qo, Ro] = schur([b'*b, 1; 1, 0]);
u = [a, bTilde] * Qo;
ao = u(:, 1);
bo = u(:, 2);

[U1, D1] = eigRankOneUpdate(U, D, U'*ao, Ro(1, 1), acc);
[U2, D2] = eigRankOneUpdate(U1, D1, U1'*bo, Ro(2, 2), acc);

if debug
    % Check the computations with MATLAB outputs
    A = U*(S*S')*U' + Ro(1, 1)*(ao*ao');
    A = (A + A') / 2;
    [U1m, D1m] = eig(A);
    err = norm(U1m*D1m*U1m' - U1*diag(D1)*U1');
    fprintf('U, D RANK-2 Step 1: || U1m D1m U1m'' - U1 D1 U1'' || = %g.\n', err);
    if err > sqrt(acc)
        fprintf('Check error!\n');
    end  
    
    A = U*(S*S')*U' + Ro(1, 1)*(ao*ao') + Ro(2, 2)*(bo*bo');
    A = (A + A') / 2;
    [U2m, D2m] = eig(A);
    err = norm(U2m*D2m*U2m' - U2*diag(D2)*U2');
    fprintf('U, D RANK-2 Step 2: || U2m D2m U2m'' - U2 D2 U2'' || = %g.\n', err);
    if err > sqrt(acc)
        fprintf('Check error!\n');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Reconstruct the singular value decomposition from the various RANK-2
% updates above
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Unew = fliplr(U2);
if M > N
    Snew = [diag(sqrt(flipud(E2))); zeros(M-N, N)];
else
    Snew = [diag(sqrt(flipud(E2))), zeros(M, N-M)];
end
Vnew = fliplr(V2);

% Compute the sign and rotate V appropriately
SIGN = Unew'*U*(S + U'*a*b'*V)*V'*Vnew;
tol = N * acc * max(diag(Snew));
SIGN(abs(SIGN) < tol) = 0;
SIGN = sign(diag(SIGN));
SIGN(SIGN == 0) = 1;
Vnew = Vnew * diag(SIGN);

% Undo the transpose that we might have done at the beginning if N > M
if swapped
    Snew = Snew';
    [Unew, Vnew] = swap(Unew, Vnew);
end

% Collect some stats
stats.totalTime = toc(svdTimer);

function [b, a] = swap(a, b)
% A hacky but fast way to swap two variables