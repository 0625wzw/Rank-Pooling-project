% 不同池化方法的F1分数比较
%
plot(x,F1RP,'-b','LineWidth',2);hold on;plot(x,F1AP,'-g','LineWidth',2);hold on;plot(x,F1MP,'-r','LineWidth',2);
legend('F1-RankPool曲线','F1-AvePool曲线','F1-MaxPool曲线'); % legend 会自动根据画图顺序分配图形
hold off;