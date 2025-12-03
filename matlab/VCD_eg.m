% Example of VC Density

L = 0.01; % radius

X_max = 2; X_min = -2; Nx = 40000;
dx = (X_max-X_min)/(Nx-1);
x = [X_min:dx:X_max];
Ns = 1;
NL = round(L/dx);

y = zeros(size(x)); 
y1 = zeros(size(x)); 
y2 = zeros(size(x)); 

% polygonal function
% for ii=1:Nx
%     if x(ii) < -1
%         y(ii) = x(ii)+2;
%     elseif -1 <= x(ii)&&x(ii) < 0
%         y(ii) = -2*x(ii)-1;
%     elseif 0 <= x(ii)&&x(ii) < 1
%         y(ii) = 3*x(ii)-1;
%     elseif 1 <= x(ii)&&x(ii) <=2
%         y(ii) = -4*x(ii)+6;
%     end
% end

% f1
for ii=1:Nx
    if x(ii) < 0
        y1(ii) = 2*x(ii)+2;
    elseif 0 <= x(ii)&&x(ii) <=2
        y1(ii) = 0;
    end
end
% f2
for ii=1:Nx
    if x(ii) < 0
        y2(ii) = 2*x(ii)+2;
    elseif 0 <= x(ii)&&x(ii) <=2
        y2(ii) = -x(ii)+1;
    end
end

% Extended data
data_new = zeros(size(y,1),size(y,2)+2*NL);
data_new1 = zeros(size(y,1),size(y,2)+2*NL);
data_new2 = zeros(size(y,1),size(y,2)+2*NL);

for ii = 1:size(data_new1,1)
    for jj = 1:size(data_new1,2)
        if jj <= NL
            data_new1(ii,jj)=y1(ii,1);
        elseif jj >= size(y,2) + NL
            data_new1(ii,jj)=y1(ii,size(y,2));
        else 
            data_new1(ii,jj)=y1(ii,jj-NL);
        end
    end
end
f1 = data_new1;

for ii = 1:size(data_new2,1)
    for jj = 1:size(data_new2,2)
        if jj <= NL
            data_new2(ii,jj)=y2(ii,1);
        elseif jj >= size(y,2) + NL
            data_new2(ii,jj)=y2(ii,size(y,2));
        else 
            data_new2(ii,jj)=y2(ii,jj-NL);
        end
    end
end
f2 = data_new2;

% VC Density Plotting Area
xmax = 0.08;
Fmax = 1.5;
xx = linspace(0,xmax,100);
XX = zeros(Ns,100);

% VC Density
Delta_f1 = 0;
Delta_f2 = 0;
for j = (1+NL):(Nx-NL)
    max_f1 = max(f1(1,j-NL:j+NL));
    min_f1 = min(f1(1,j-NL:j+NL));
    delta_f1 = max_f1 - min_f1;
    Delta_f1 = [Delta_f1 delta_f1];

    max_f2 = max(f2(1,j-NL:j+NL));
    min_f2 = min(f2(1,j-NL:j+NL));
    delta_f2 = max_f2 - min_f2;
    Delta_f2 = [Delta_f2 delta_f2];
end
[density1, x_values1] = ksdensity(Delta_f1,xx, 'Kernel', 'normal');
[density2, x_values2] = ksdensity(Delta_f2,xx, 'Kernel', 'normal');

XX1(Ns,:) = x_values1;
subplot(1,2,1);
plot(x_values1,density1,'k:','LineWidth',1.5);hold on;
text(0.05, 0.95, '(a)', 'Units', 'normalized', 'FontSize', 12);
set(gca, 'FontSize', 14); 
xlabel('VC');
ylabel('VC Densityc of {\it f_1}');

XX2(Ns,:) = x_values2;
subplot(1,2,2);
plot(x_values2,density2,'k:','LineWidth',1.5);hold on;
text(0.05, 0.95, '(b)', 'Units', 'normalized', 'FontSize', 12);
set(gca, 'FontSize', 14); 
xlabel('VC');
ylabel('VC Density of {\it f_2}');
