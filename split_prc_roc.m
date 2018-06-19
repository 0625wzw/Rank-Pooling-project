close all;
%%����Precision-Recalll Curve����
x=xlsread('H:\graduate project\code\x1.xlsx','Sheet1');
y=xlsread('H:\graduate project\code\y1.xlsx','Sheet1');
h=plot(x,y);
legend('Rank Pooling','Max Pooling','Average Pooling');
set(h,'LineWidth',2);
set(h(1),'color','r');
set(h(1),'Linestyle','--');
set(h(2),'color','m');
set(h(3),'color','y');
set(h(3),'color','b');
title('Precision-Recalll Curve');
xlabel('recall');
axis([0,1, 0, 1]);
ylabel('precision');
grid on;            %��������
box on;

%% ����Precision-Recalll Curve����
% x=xlsread('H:\graduate project\code\x2.xlsx','Sheet1');
% y=xlsread('H:\graduate project\code\y2.xlsx','Sheet1');
% h=plot(x,y);
% legend('Rank Pooling','Max Pooling','Average Pooling','�ο���');
% set(h,'LineWidth',2);
% set(h(1),'color','r');
% set(h(1),'Linestyle','--');
% set(h(2),'color','m');
% set(h(3),'color','y');
% set(h(3),'color','b');
% title('ROC Curve');
% xlabel('FPR(false positive rate)');
% axis([0,1, 0, 1]);
% ylabel('TPR(true positive rate)');
% grid on;            %��������
% box on;
%%