function [pred,model,prob] =KLieLapSVM(Xl,Yl,Xu,varargin)

% inout
% Xl:	labeled X
% Yl:	labels of Xl
% Xu:	unlabeled X
% varargin: Select of hyper-parameters

% output
% pred : predicted labels for Xu
% prob : the confidence of pred
% model: model

addpath(genpath('lapsvmp_lie'))

ip=inputParser;
addParameter(ip,'Kernel',0);
addParameter(ip,'KernelParam',.001)
%addParameter(ip,'C',1)
addParameter(ip,'gamma_I',1)
addParameter(ip,'gamma_A',1e-5)
addParameter(ip,'knn',5)
addParameter(ip, 'GraphDistanceFunction', 'euclidean')
addParameter(ip, 'multiclass', 'one-vs-one')
addParameter(ip, 'roboss', 0)

parse(ip,varargin{:});
t=ip.Results.Kernel;
g=ip.Results.KernelParam;
%c=ip.Results.C;
gamma_I=ip.Results.gamma_I;
gamma_A=ip.Results.gamma_A;
knn=ip.Results.knn;
DistanceFunction=ip.Results.GraphDistanceFunction;
multiclass=ip.Results.multiclass;
roboss=ip.Results.roboss;

%% kernels
if t == 0
    Kernel = 'linear'; 
elseif t == 1
    Kernel = 'polym'; 
elseif t == 2
    Kernel = 'rbf';
%     KernelParam = sqrt(1/2/g);
elseif t == 3
    Kernel = 'tanh';
elseif t == 4
    Kernel = 'sig';
elseif t == 5
    Kernel = 'log';
else 
    error('wrong ker'); 
end

%% make options
options = make_options('gamma_I',gamma_I,'gamma_A',gamma_A,'NN',knn,...
	'Kernel',Kernel,'KernelParam',g,'GraphDistanceFunction',DistanceFunction);
options.Verbose = 0;%default:0
options.UseBias = 1;%default:0
options.UseHinge = 1;%default:1
if roboss
    options.roboss = 1;
options.LaplacianNormalize = 0;%default:0
options.NewtonLineSearch = 1;%default:1
if t == 1
    options.polyDegree = 2;
end

%% train the classifier and predict
options.Cg = 0; % 0: newton 1: PCG
options.MaxIter = 1000; % upper bound
options.CgStopType = 1; % 'stability' early stop default:1
options.CgStopParam = 0.0015; % tolerance: 1.5% default:0.015
options.CgStopIter = 3; % check stability every 3 iterations

nCls = max(Yl);
nTest = size(Xu,1);
%% multi class
if nCls > 2 
    % one-vs-one
    if multiclass=='one-vs-one'
        pred =zeros(nTest,nCls+1);
        idx=1;
        for iCls = 0:nCls
            for jCls=iCls+1:nCls
                X_temp = []; Y_temp=[];
                for i = 1:size(Xl,1)
                    current_label = Yl(i);
                    if (current_label == iCls)
                        current_feature = Xl(i);
                        X_temp{size(X_temp, 2)+1} = cell2mat(current_feature);
                        Y_temp=[Y_temp,1];
        
                    elseif (current_label ==jCls)
                        current_feature = Xl(i);
                        X_temp{size(X_temp, 2)+1} = cell2mat(current_feature);
                        Y_temp=[Y_temp,-1];
                    end
                end
                % create the 'data' structure
                X = [X_temp';Xu];
                data.X = X;
                data.K = liekernel(options,X,X);
                data.L = laplacian(options,X);

                %data.Y = [(Yl-1.5)*2;zeros(nTest,1)];
                data.Y = [Y_temp';zeros(nTest,1)];
	            classifier = lapsvmp(options,data);
	            
	            out = data.K(:,classifier.svs)*classifier.alpha+classifier.b;
                prob = out(end-nTest+1:end);
                pred_temp = sign(prob);
	            model{idx} = classifier;
                idx=idx+1;
                for i = 1:nTest
                    if pred_temp(i)==1
                        pred(i,iCls+1)=pred(i,iCls+1)+1;
                    else
                        pred(i,jCls+1)=pred(i,jCls+1)+1;
                    end
                end
            end
        end
        [~,pred] = max(pred,[],2);
        pred=pred-1;
                     
    % one-vs-all
    elseif multiclass=='one-vs-all'
        % create the 'data' structure
        X = [Xl;Xu];
        data.X = X;
        data.K = liekernel(options,X,X);
        data.L = laplacian(options,X);

	    % warning('currently using one-vs-all strategy on lapSVM, one-vs-one may be better.');
	    prob = nan(nTest,nCls);
	    for iCls = 0:nCls
		    Y1 = Yl == iCls;
		    data.Y = [(Y1-.5)*2;zeros(nTest,1)];% {0,1} to {-1,0,+1}
            % data.Y = [Y1;zeros(nTest,1)];% {0,1} to {-1,0,+1}
		    classifier = lapsvmp(options,data);
		    % fprintf('It took %f seconds.\n',classifier.traintime);
		    
		    out = data.K(:,classifier.svs)*classifier.alpha+classifier.b;
		    prob(:,iCls+1) = out(end-nTest+1:end);
		    model{iCls+1} = classifier;
	    end
	    [~,pred] = max(prob,[],2);
        pred=pred-1;
    end

%% binary class
else
    % create the 'data' structure
    X = [Xl;Xu];
    data.X = X;
    data.K = liekernel(options,X,X);
    data.L = laplacian(options,X);

	%data.Y = [(Yl-1.5)*2;zeros(nTest,1)];
    data.Y = [Yl;zeros(nTest,1)];
	classifier = lapsvmp(options,data);
	
	out = data.K(:,classifier.svs)*classifier.alpha+classifier.b;
    prob = out(end-nTest+1:end);
	%prob = out(end-nTest+1:end-size(Xt,1));
	%pred = sign(prob)/2+1.5;
    pred = sign(prob);
	model = classifier;

end

end