%不同池化方法的recall比较
%
 plot(x,recallRP,'-b','LineWidth',2);hold on;plot(x,recallAP,'-g','LineWidth',2);hold on;plot(x,recallMP,'-r','LineWidth',2);
legend('Recall-RankPool曲线','Recall-AvePool曲线','Recall-MaxPool曲线'); % legend 会自动根据画图顺序分配图形
hold off; 