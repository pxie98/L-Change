% Local 2-D Value Change 
f = @(x, y) sin(x).*cos(5*y);
f1 = @(x, y) sqrt((cos(x).*cos(5*y)).^2 + (5*sin(x).*sin(5*y)).^2);

Lx = 0.1;
Ly = Lx;

dx = 0.005; X_max = 2; X_min = -2;
dy = 0.005; Y_max = 2; Y_min = -2;
% L_max = 0.1; L_min = 0; L = [L_min:dx:L_max]
x = [X_min:dx:X_max]; Nx = (X_max-X_min)/dx;
y = [Y_min:dy:Y_max]; Ny = (Y_max-Y_min)/dy;
[X,Y] = meshgrid(x,y);
NLx = round(Lx/dx);
NLy = round(Ly/dy);
z = f(X,Y);
z1 = f1(X,Y);

% Extension on the boundary
z_new = zeros(2*NLy+Ny+1,2*NLx+Nx+1);
h = waitbar(0,'extend');
for ii = 1:(2*NLy+Ny+1), waitbar(ii/(2*NLy+Ny+1))
    for jj = 1:(2*NLx+Nx+1)
        if ii<=NLy+1 && jj<=NLx+1
            z_new(ii,jj) = z(1,1);
        elseif ii<=NLy+1 && jj>NLx+1 && jj<NLx+Nx
            z_new(ii,jj) = z(1,jj-NLx);
        elseif ii<=NLy+1 && jj>=NLx+Nx
            z_new(ii,jj) = z(1,Nx+1);
        elseif ii>NLy+1 && ii<NLy+Ny && jj<=NLx+1
            z_new(ii,jj) = z(ii-NLy,1);
        elseif ii>NLy+1 && ii<NLy+Ny && jj>NLx+1 && jj<NLx+Nx
            z_new(ii,jj) = z(ii-NLy,jj-NLx);
        elseif ii>NLy+1 && ii<NLy+Ny && jj>=NLx+Nx
            z_new(ii,jj) = z(ii-NLy,Nx+1);
        elseif ii>=NLy+Ny && jj<=NLx+1
            z_new(ii,jj) = z(Ny+1,1);
        elseif ii>=NLy+Ny && jj>NLx+1 && jj<NLx+Nx
            z_new(ii,jj) = z(Ny+1,jj-NLx);
        elseif ii>=NLy+Ny && jj>=NLx+Nx
            z_new(ii,jj) = z(Ny+1,Nx+1);
        end
    end
end,close(h)

VC = zeros(Ny+1,Nx+1);
h = waitbar(0,'VC');
for i = 1:(Ny+1), waitbar(i/(Ny+1))
    for j = 1:(Nx+1)
        max_f = max(max(z_new(i:(i+2*NLy),j:(j+2*NLx))));
        min_f = min(min(z_new(i:(i+2*NLy),j:(j+2*NLx))));
        delta = max_f - min_f;
        VC(i,j) = delta;
    end
end,close(h)
figure('Name','raw');
surf(x,y,z);
colormap(jet);
shading interp
axis tight
ylabel("y")
xlabel("x")

figure('Name','VC');
surf(x,y,VC);
colormap(jet);
shading interp
axis tight
ylabel("y")
xlabel("x")

figure('Name','gradient');
surf(x,y,z1);
colormap(jet);
shading interp
axis tight
ylabel("y")
xlabel("x")
% colormap([0 0 0]);


