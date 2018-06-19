% Rank Pooling ����ʵ����Ƶ���б���
% ���� :vlfeat-0.9.18, liblinear-2.20, libsvm-3.22 

function ChaLearn_Rank_Pooling()    
    % ���vlfeat·��������
    addpath('H:/environment/vlfeat-0.9.18/toolbox');
    vl_setup();	
	% ���libnear·��
	addpath('G:/MATLAB 2016b/MATLAB 2016b/toolbox/liblinear-2.20/liblinear-2.20/matlab');
	% ���libsvm·��
    addpath('G:/MATLAB 2016b/MATLAB 2016b/toolbox/libsvm-3.22/libsvm-3.22/matlab');
    % ����ChaLearn��Ϊʶ�����ݼ�   
    load('ChaLearn_train_test_split.mat');    
    if exist('ChaLearn_gesture_Data','dir') == 7 && numel(dir(sprintf('ChaLearn_gesture_Data/*.mat'))) == 13883
        fprintf('Dataset exists..\n');
    else
        fprintf('Dataset is no exists, please load Dataset exists..\n'); 
    end
    videoname = fnames;	
    feats = {'ChaLearn_gesture_Data'};   
    for f = 1 : numel(feats)
        file = sprintf('%s.mat',feats{f});
        if exist(file,'file') ~= 2
            [RankPooledFeats,MaxPooledFeats,MeanPooledFeats] = getRankPooling(feats{f},videoname);
            save(file,'RankPooledFeats','MaxPooledFeats','AveragePooledFeats');
        else
            load(file);
        end
      ALL_Data_cell{f} = RankPooledFeats;  
%       ALL_Data_cell{f} = MaxPooledFeats;  
%       ALL_Data_cell{f} = MeanPooledFeats;  
    end    
    Options.KERN =0;    % non linear kernel  Options.KERN =0 ʵ����Accuracy=76.5018%,F1=0.76
    Options.Norm =2;     % L2  normalization 
   

    % chi2�������ں�ӳ�䷽��
	if Options.KERN == 3        
        for ch = 1 : size(ALL_Data_cell,2)                
             x = vl_homkermap(ALL_Data_cell{ch}', 2, 'kchi2') ;
             ALL_Data_cell{ch} = x';
             % ʵ������Accuracy=74.4324%,F1=0.74
        end
    end
    
      % SSR����
      if Options.KERN == 4
          for ch = 1 : size(ALL_Data_cell,2)                
          ALL_Data_cell{ch} = sign( ALL_Data_cell{ch}).*sqrt( ALL_Data_cell{ch});
             % ʵ������Accuracy=73.7357%,F1=0.74
          end
      end  
      
    % SER����
    if Options.KERN == 5        
        for ch = 1 : size(ALL_Data_cell,2)                
          ALL_Data_cell{ch} = rootKernelMap(ALL_Data_cell{ch});
             % ʵ������Accuracy=73.7357%,F1=0.74 
        end
    end 
     
    %L2��һ��
	if Options.Norm == 2       
         for ch = 1 : size(ALL_Data_cell,2)                 
            ALL_Data_cell{ch} = normalizeL2(ALL_Data_cell{ch});
         end
    end  
    
    if size(ALL_Data_cell,2) == 1
        weights = 1;
    end

    if size(ALL_Data_cell,2) == 2 || size(ALL_Data_cell,2) == 6 
        weights = [0.5 0.5];
    end

    if size(ALL_Data_cell,2) > 2 && size(ALL_Data_cell,2) ~= 6
        nch = size(ALL_Data_cell,2) ;
        weights = ones(1,nch) * 1/nch;
    end      
    
    %ѵ�����Ͳ��Լ�
    classid = labels2;  
    trn_indx = [cur_train_indx]; % cur_train_indx  cur_val_indx 
    test_indx = [cur_test_indx];  % cur_test_indx     
    
    TrainClass_ALL = classid(trn_indx,:);
    TestClass_ALL = classid(test_indx,:);   
   [~,TrainClass] = max(TrainClass_ALL,[],2);
   [~,TestClass] = max(TestClass_ALL,[],2);   		
      
    for ch = 1 : size(ALL_Data_cell,2)        
        ALL_Data = ALL_Data_cell{ch};
        TrainData = ALL_Data(trn_indx,:);        
        TestData = ALL_Data(test_indx,:);

        TrainData_Kern_cell{ch} = [TrainData * TrainData'];    
        TestData_Kern_cell{ch} = [TestData * TrainData'];                        
        clear TrainData; clear TestData; clear ALL_Data;            
    end
    
    for wi = 1 : size(weights,1)
        TrainData_Kern = zeros(size(TrainData_Kern_cell{1}));
        TestData_Kern = zeros(size(TestData_Kern_cell{1}));
            for ch = 1 : size(ALL_Data_cell,2)     
                TrainData_Kern = TrainData_Kern + weights(wi,ch) * TrainData_Kern_cell{ch};
                TestData_Kern = TestData_Kern + weights(wi,ch) * TestData_Kern_cell{ch};
            end
            [precision(wi,:),recall(wi,:),acc(wi) ] = train_and_classify(TrainData_Kern,TestData_Kern,TrainClass,TestClass); 
    end          
            
    [~,indx] = max(acc);    
   %precision��recall�Լ�F1����Ԥ��
    precisionRP = precision(indx,:);
    recallRP = recall(indx,:); 
    F1RP = 2*(precisionRP .* recallRP)./(precisionRP+recallRP);
    fprintf('Mean F1RP score = %1.2f\n',mean(F1RP));
    save(sprintf('resultsRP.mat'),'precisionRP','recallRP','F1RP'); 
    
%     precisionMP = precision(indx,:);
%     recallMP = recall(indx,:); 
%     F1MP = 2*(precisionMP .* recallMP)./(precisionMP+recallMP);
%     fprintf('Mean F1MP score = %1.2f\n',mean(F1MP));
%     save(sprintf('resultsMP.mat'),'precisionMP','recallMP','F1MP');
%    
%     precisionAP = precision(indx,:);
%     recallAP = recall(indx,:); 
%     F1AP = 2*(precisionAP .* recallAP)./(precisionAP+recallAP);
%     fprintf('Mean F1AP score = %1.2f\n',mean(F1AP));
%     save(sprintf('resultsAP.mat'),'precisionAP','recallAP','F1AP');
%    
%   
end

%Rank Pooling������ʵ�ִ�����v�����������Ĳ���u�Ĳ���
function W = genRepresentation(data,CVAL,noLin)
    Data =  zeros(size(data,1)-1,size(data,2));
    % smooth������ʱ���ֵʸ������
    for j = 2 : size(data,1)                
        Data(j-1,:) = mean(data(1:j,:));
    end 
    W_fow = liblinearsvr(getNonLinearity(Data,noLin),CVAL,2);
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    Data =  zeros(size(data,1)-1,size(data,2));
    for j = 2 : size(data,1)                
        Data(j-1,:) = mean(data(1:j,:));
    end            
    W_rev = liblinearsvr(getNonLinearity(Data,noLin),CVAL,2); 			              
    W = [W_fow ; W_rev];  %W��u��������֡˳��
end

%�����Ժ���ӳ�䣺ʵ�����������������֡xtת��Ϊ֡vt
function Data = getNonLinearity(Data,noLin)
    switch nonLin
        case ''
            Data = rootExpandKernelMap(Data);
        case 'ssr'
            Data = sign(Data).*sqrt(abs(Data));
        case 'chi2'
            Data = vl_homkermap(Data',2,'kchi2');
            Data = Data';         
        case 'chi2exp'
            u = vl_homkermap(Data',1,'kchi2')';	
            Data = rootExpandKernelMap(u);
        case 'ser'
            Data = rootExpandKernelMap(Data);    
            
    end  
end

%����������ѧϰ����ģ��
function w = liblinearsvr(Data,C,normD)
    if normD == 2
        Data = normalizeL2(Data);
    end  
    N = size(Data,1);
    Labels = [1:N]';
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %d -s 11 -q',C) );%sparse���������ǻ�DataΪϡ�����
    w = model.w';
end

%������
function [ALL_Data,MAX_Data,MEAN_Data] = getRankPooling(featType,Videos)  
   
    CVAL = 1; 
    noLin = 'chi2';
    % C value for the ranking function    
	TOTAL = size(Videos,2);  
    for i = 1:TOTAL
        name = Videos{i};         
        MATFILE = fullfile(featType,sprintf('%s.mat',name));        
        load(MATFILE);
        data  = clustDist';  clear clustDist;%dataΪ֡xt
        W = genRepresentation(data,CVAL,noLin); 
        maxPooled = max(data);            
        meanPooled = mean(data);
        if i == 1
             ALL_Data =  zeros(TOTAL,size(W,1)) ;          
             MAX_Data =  zeros(TOTAL,size(maxPooled,2)) ;
             MEAN_Data =  zeros(TOTAL,size(meanPooled,2)) ;
        end
        if mod(i,100) == 0 
            fprintf('.')
        end
        ALL_Data(i,:) = W';
        MAX_Data(i,:) = maxPooled';
        MEAN_Data(i,:) = meanPooled';
    end
   fprintf('Complete...\n')
end

%L2��һ������
function X = normalizeL2(X)
    for i = 1 : size(X,1)
		if norm(X(i,:)) ~= 0
			X(i,:) = X(i,:) ./ norm(X(i,:));
		end
    end	   
end

%SER�����Ժ���
function X = rootKernelMap(X)
    X = sqrt(X);
end

%����ѵ������
function [trn,tst] = generateTrainTest(classid)
    trn = zeros(numel(classid),1);
    tst = zeros(numel(classid),1);
    maxC = max(classid);
    for c = 1 : maxC
        indx = find(classid == c);
        n = numel(indx);
        tindx = indx(1:4);
        testindx = indx(5:end);
        trn(tindx,1) = 1;
        tst(testindx,1) = 1;
    end
end

%label��ȡ����
function [X] = getLabel(classid)
    X = zeros(numel(classid),max(classid))-1;
    for i = 1 : max(classid)
        indx = find(classid == i);
        X(indx,i) = 1;
    end
end

%ʹ��svm��u���з������
function [precision,recall,acc ] = train_and_classify(TrainData_Kern,TestData_Kern,TrainClass,TestClass)
         nTrain = 1 : size(TrainData_Kern,1);
         TrainData_Kern = [nTrain' TrainData_Kern];         
         nTest = 1 : size(TestData_Kern,1);
         TestData_Kern = [nTest' TestData_Kern];         
         C = [1 10 100 500 1000 ];
         for ci = 1 : numel(C)
             model(ci) = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f -v 2 -q ',C(ci)));               
         end                
         [~,max_indx]=max(model);         
         C = C(max_indx);        
         for ci = 1 : numel(C)
             model = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f  -q ',C(ci)));
             [predicted, acc, scores{ci}] = svmpredict(TestClass, TestData_Kern ,model);	                 
             [precision(ci,:) , recall(ci,:)] = perclass_precision_recall(TestClass,predicted);
             accuracy(ci) = acc(1,1);
         end             
        [acc,cindx] = max(accuracy);   
        scores = scores{cindx};
        precision = precision(cindx,:);
        recall = recall(cindx,:);
end

% ���� precision �� recall ��ֵ
function [precision , recall] = perclass_precision_recall(label,predicted)    
    for cl = 1 : 20
        true_pos = sum((predicted == cl) .* (label == cl));
        false_pos = sum((predicted == cl) .* (label ~= cl));
        false_neg = sum((predicted ~= cl) .* (label == cl));
        true_neg = sum((predicted ~= cl) .* (label ~= cl));
        precision(cl) = true_pos / (true_pos + false_pos);
        recall(cl) = true_pos / (true_pos + false_neg);  
       
    end
end


