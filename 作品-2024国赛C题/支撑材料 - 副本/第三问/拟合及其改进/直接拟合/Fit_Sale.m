clear,clc
load Fit_Data.mat

%提取自变量
x1=Fit_Data(:,1);
x2=Fit_Data(:,2);
y=Fit_Data(:,3);

% 拟合: '销售单价与预期销售量'。
[xData, yData] = prepareCurveData( x2, y );

% 设置 fittype 和选项。
ft = fittype( 'log' );

% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft );

% 输出销售单价与预期销售量的拟合优度R²
fprintf('销售单价与预期销售量的拟合优度 R²: %.4f\n', gof.rsquare);

% 绘制数据拟合图。
figure( 'Name', '销售单价与预期销售量' );
h = plot( fitresult, xData, yData );
legend( h, 'y vs. x2', '销售单价与预期销售量', 'Location', 'NorthEast', 'Interpreter', 'none' );
% 为坐标区加标签
xlabel( 'x2', 'Interpreter', 'none' );
ylabel( 'y', 'Interpreter', 'none' );
grid on

% 拟合: '种植成本与销售量'。
[xData, yData] = prepareCurveData( x1, y );

% 设置 fittype 和选项。
ft = fittype( 'log' );

% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft );

% 输出种植成本与销售量的拟合优度R²
fprintf('种植成本与销售量的拟合优度 R²: %.4f\n', gof.rsquare);

% 绘制数据拟合图。
figure( 'Name', '种植成本与销售量' );
h = plot( fitresult, xData, yData );
legend( h, 'y vs. x1', '种植成本与销售量', 'Location', 'NorthEast', 'Interpreter', 'none' );
% 为坐标区加标签
xlabel( 'x1', 'Interpreter', 'none' );
ylabel( 'y', 'Interpreter', 'none' );
grid on
