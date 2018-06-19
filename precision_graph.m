% 不同池化方法的precision比较  
%
plot(x,precisionRP,'-b','LineWidth',2);hold on;plot(x,precisionAP,'-g','LineWidth',2);hold on;plot(x,precisionMP,'-r','LineWidth',2);
legend('Precision-RankPool曲线','Precision-AvePool曲线','Precision-MaxPool曲线'); % legend 会自动根据画图顺序分配图形
hold off;