clc;
clear;
close all;

%% 1. Генерація набору даних

rng(42);               % Для повторюваності результатів
n = 1000;              % Кількість спостережень

% Незалежна змінна X у діапазоні 0...1000
X = 1000 * rand(n,1);

% Випадковий шум
noise = 50 * randn(n,1);

% Формування залежної змінної Y
Y = 0.5 * X + 100 * sin(X / 100) + noise;

% Виведення перших 10 значень
disp('Перші 10 значень X та Y:');

disp(table( ...
    X(1:10), ...
    Y(1:10), ...
    'VariableNames', {'X','Y'} ...
));

fprintf('Кількість спостережень: %d\n', n);

% Графік початкових даних
figure;

scatter(X, Y, 15, 'filled');

xlabel('X');
ylabel('Y');

title('Початковий набір даних');

grid on;


%% 2. Нормалізація значень

% Нормалізація X у діапазон [0; 1]
X_norm = (X - min(X)) / (max(X) - min(X));

% Нормалізація Y у діапазон [0; 1]
Y_norm = (Y - min(Y)) / (max(Y) - min(Y));

% Виведення перших 10 нормалізованих значень
disp(' ');
disp('Перші 10 нормалізованих значень:');

disp(table( ...
    X_norm(1:10), ...
    Y_norm(1:10), ...
    'VariableNames', {'X_norm','Y_norm'} ...
));

fprintf('Мінімум X_norm = %.4f\n', min(X_norm));
fprintf('Максимум X_norm = %.4f\n', max(X_norm));

fprintf('Мінімум Y_norm = %.4f\n', min(Y_norm));
fprintf('Максимум Y_norm = %.4f\n', max(Y_norm));

% Графік нормалізованих даних
figure;

scatter( ...
    X_norm, ...
    Y_norm, ...
    15, ...
    'filled' ...
);

xlabel('Нормалізоване X');
ylabel('Нормалізоване Y');

title('Нормалізований набір даних');

grid on;

%% 3. Розділення на навчальну і тестову вибірки

rng(42);

% Створюємо випадковий поділ:
% 80 % - навчальна вибірка
% 20 % - тестова вибірка
cv = cvpartition(length(X_norm), 'HoldOut', 0.20);

train_idx = training(cv);
test_idx  = test(cv);

% Формування вибірок
X_train = X_norm(train_idx);
Y_train = Y_norm(train_idx);

X_test = X_norm(test_idx);
Y_test = Y_norm(test_idx);

% Перевірка
fprintf('\nКількість навчальних записів: %d\n', length(X_train));
fprintf('Кількість тестових записів: %d\n', length(X_test));

fprintf('X_train: %.4f ... %.4f\n', ...
    min(X_train), max(X_train));

fprintf('X_test:  %.4f ... %.4f\n', ...
    min(X_test), max(X_test));

%% Графік

figure;

scatter(X_train, Y_train, 18, 'filled');
hold on;

scatter(X_test, Y_test, 35, 'x');

xlabel('Нормалізоване X');
ylabel('Нормалізоване Y');

title('Навчальна і тестова вибірки');

legend( ...
    'Навчальна вибірка', ...
    'Тестова вибірка', ...
    'Location', 'best' ...
);

grid on;
hold off;
%% 4. Навчання KNN-регресора для різних значень K

K_values = 5:30;

MSE_values = zeros(size(K_values));

fprintf('\nРезультати KNN-регресії:\n');
fprintf('K\tMSE тестової вибірки\n');

for i = 1:length(K_values)

    K = K_values(i);

    % Пошук K найближчих сусідів
    idx = knnsearch( ...
        X_train, ...
        X_test, ...
        'K', K ...
    );

    % Прогнозоване значення Y
    % як середнє значення K найближчих сусідів
    Y_pred = mean(Y_train(idx), 2);

    % Середньоквадратична похибка
    MSE_values(i) = mean( ...
        (Y_test - Y_pred).^2 ...
    );

    fprintf( ...
        '%d\t%.6f\n', ...
        K, ...
        MSE_values(i) ...
    );

end

%% 5. Вибір оптимального значення K

[min_MSE, min_idx] = min(MSE_values);

best_K = K_values(min_idx);

fprintf('\nОптимальне значення K = %d\n', best_K);
fprintf('Мінімальна MSE = %.6f\n', min_MSE);

%% 6. Візуалізація отриманих результатів

%% 6.1. Залежність MSE від K

figure;

plot( ...
    K_values, ...
    MSE_values, ...
    '-o', ...
    'LineWidth', 1.5 ...
);

hold on;

plot( ...
    best_K, ...
    min_MSE, ...
    'o', ...
    'MarkerSize', 10, ...
    'LineWidth', 2 ...
);

xlabel('Кількість сусідів K');
ylabel('MSE');

title('Залежність MSE від значення K');

legend( ...
    'MSE для різних K', ...
    'Оптимальне K', ...
    'Location', 'best' ...
);

grid on;
hold off;


%% 6.2. KNN-регресія для оптимального K

X_grid = linspace( ...
    min(X_norm), ...
    max(X_norm), ...
    500 ...
)';

idx_grid = knnsearch( ...
    X_train, ...
    X_grid, ...
    'K', best_K ...
);

Y_grid_pred = mean( ...
    Y_train(idx_grid), ...
    2 ...
);

figure;

scatter( ...
    X_train, ...
    Y_train, ...
    15, ...
    'filled' ...
);

hold on;

scatter( ...
    X_test, ...
    Y_test, ...
    30, ...
    'x' ...
);

plot( ...
    X_grid, ...
    Y_grid_pred, ...
    'LineWidth', 2.5 ...
);

xlabel('Нормалізоване X');
ylabel('Нормалізоване Y');

title( ...
    sprintf( ...
        'KNN-регресія для оптимального K = %d', ...
        best_K ...
    ) ...
);

legend( ...
    'Навчальна вибірка', ...
    'Тестова вибірка', ...
    'KNN-регресія', ...
    'Location', 'best' ...
);

grid on;
hold off;