% Load the dataframe from the CSV file
data = readtable('C:\Users\giord\OneDrive\Desktop\UNI\MAGISTRALE\Intelligent Systems For Pattern Recognition\Assignments\PatternRecognition\1\data\1.csv');

col = 2;

% Assuming each column is a variable you want to analyze with CWT

% Extract the column data. This method works if the data are numeric.
columnData = table2array(data(:, col));

% Perform the Continuous Wavelet Transform on the column data
[cfs, frequencies] = cwt(columnData, 'bump'); % Example uses the Morlet (analytic Morlet) wavelet

% Plot the CWT result for this column
figure; % Open a new figure window
imagesc(abs(cfs));  % Visualize the absolute value of the CWT coefficients
colorbar;  % Show a color bar indicating the coefficient magnitudes
title(['CWT of Column ', num2str(col)]);
xlabel('Time');
ylabel('Scale');
axis tight; % Fit the axes tightly around the data


% Extract unique activity labels from the 5th column
activity_labels = unique(data{:, 5});
activity_labels = activity_labels(2:end);

% Loop through each unique activity label
for activity_label_index = 1:numel(activity_labels)
    % Extract data corresponding to the current activity label
    activity_data = data(data{:, 5} == activity_labels(activity_label_index), :);

    % Extract the column data for the current variable
    columnData = table2array(activity_data(:, col));
    
    % Perform the Continuous Wavelet Transform on the column data
    [cfs, frequencies] = cwt(columnData, 'bump'); % Example uses the Morlet (analytic Morlet) wavelet

    % Plot the CWT result for this column
    figure; % Open a new figure window
    imagesc(abs(cfs));  % Visualize the absolute value of the CWT coefficients
    colorbar;  % Show a color bar indicating the coefficient magnitudes
    title(['CWT of Column ', num2str(col), ' for Activity ', (activity_labels(activity_label_index))]);
    xlabel('Time');
    ylabel('Scale');
    axis tight; % Fit the axes tightly around the data

end

