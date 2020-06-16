clc;
clear all;
load iris.dat               
setosa = iris((iris(:,5)==1),:);        % data for setosa
versicolor = iris((iris(:,5)==2),:);    % data for versicolor
virginica = iris((iris(:,5)==3),:);     % data for virginica
obsv_n = size(iris, 1);                 % total number of observations

Characteristics = {'sepal length','sepal width','petal length','petal width'};
pairs = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];
h = figure;
for j = 1:6
    x = pairs(j, 1); 
    y = pairs(j, 2);   
    subplot(2,3,j);
    plot([setosa(:,x) versicolor(:,x) virginica(:,x)],...
         [setosa(:,y) versicolor(:,y) virginica(:,y)], '.');
    xlabel(Characteristics{x},'FontSize',10); 
    ylabel(Characteristics{y},'FontSize',10);
end

cluster_n = 3;                          % Number of clusters
expo = 2.0;                             % Exponent for U
max_iter = 100;                         % Max. iteration
min_impro = 1e-6;      

% initialize fuzzy partition   
U = initfcm(cluster_n, obsv_n);     

if ishghandle(h)
    figure(h);
else
    for j = 1:6,
        x = pairs(j, 1); 
        y = pairs(j, 2);   
        subplot(2,3,j);
        plot([setosa(:,x) versicolor(:,x) virginica(:,x)],...
             [setosa(:,y) versicolor(:,y) virginica(:,y)], '.');
        xlabel(Characteristics{x},'FontSize',10); 
        ylabel(Characteristics{y},'FontSize',10);
    end
end    
% iteration
for i = 1:max_iter,
    [U, center, obj] = stepfcm(iris, U, cluster_n, expo);
    fprintf('Iteration count = %d, obj. fcn = %f\n', i, obj);
    
    if i>1 && (abs(obj - lastobj) < min_impro)
        for j = 1:6,
            subplot(2,3,j);
            for k = 1:cluster_n,
                text(center(k, pairs(j,1)), center(k,pairs(j,2)), int2str(k), 'FontWeight', 'bold');
            end
        end
        break; 
    elseif i==1
        for j = 1:6,
            subplot(2,3,j);
            for k = 1:cluster_n,
                text(center(k, pairs(j,1)), center(k,pairs(j,2)), int2str(k), 'color', [0.5 0.5 0.5]);
            end
        end
    end
    lastobj = obj;
end
center=center(:,1:4);
center=center/10;

%FCM by inbuilt method
data = load('iris.csv'); 
data=data(:,1:4);% load some sample data
n_clusters_inbuilt = 3;              % number of clusters
[center_inbuilt,U_inbuilt,obj_fcn_inbuilt] = fcm(data, n_clusters_inbuilt);