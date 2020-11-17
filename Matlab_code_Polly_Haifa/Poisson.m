% -------------------------------------------------------------
  function X = Poisson(lambda,n) % n represents the number of iterations
% -------------------------------------------------------------
% Generate a random value from the (discrete) Poisson 
% distribution with parameter lambda.
% Derek O'Connor, 6 Feb 2012.  derekroconnor@eircom.net
%
X = zeros( n,1);
for i=1:n
    k=1; usave=1;
    usave = usave*rand;
    while usave >= exp(-lambda)
        usave = usave*rand;
        k = k+1;
    end
    X(i) = k-1;
end
