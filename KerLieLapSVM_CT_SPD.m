% clear 
% close all  
% clc
addpath(genpath('lapsvmp_lie'))
load CT.mat;
train_images_matrix=data.trainX;
train_labels=data.trainY;
test_images_matrix=data.testX;
test_labels=data.testY;

% train
times=1;
for t =1:times

    n=randperm(2600);
    m=randperm(400);

    train_list = []; 
    num_tra=400;
    train_label=zeros(num_tra,1);

    for i = 1:num_tra

        current_image = train_images_matrix(:,:,n(i));
        train_label(i)=train_labels(n(i));
        current_feature = genSPD(current_image);
        current_feature=current_feature+0.01*trace(current_feature)*eye(size(current_feature,1));
        train_list{size(train_list, 2)+1} = current_feature;

    end

    train_list=train_list';

    test_list=[];
    num_t=2000;
    test_label=zeros(num_t,1);

    for j = num_tra+1:num_tra+num_t

        current_image = train_images_matrix(:,:,n(j));
        test_label(j-num_tra)=train_labels(n(j));
        current_feature = genSPD(current_image);
        current_feature=current_feature+0.01*trace(current_feature)*eye(size(current_feature,1));
        test_list{size(test_list, 2)+1} = current_feature;

    end

    test_list=test_list';


    [pred2,model,prob2] = KLieLapSVM(train_list,train_label,test_list,...
            'Kernel',0,'KernelParam',1,'gamma_I',0.01,'gamma_A',0.1,'knn',5,...
            'GraphDistanceFunction','geodesic_distance','roboss',1);
    acc(t) = sum(pred2== test_label)/length(test_label);

end

disp(acc);