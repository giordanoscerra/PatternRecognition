% Load the dataframe from the CSV file
data = readtable('C:\Users\giord\OneDrive\Desktop\UNI\MAGISTRALE\Intelligent Systems For Pattern Recognition\Assignments\PatternRecognition\1\data\1.csv');

col = 2; %from 2 to 4: x, y, z 

% Define a structure where each field represents an activity ID and its value is the corresponding description
activity_descriptions = {
    'Working at Computer';                             % Activity 1
    'Standing Up, Walking and Going up/down stairs';   % Activity 2
    'Standing';                                        % Activity 3
    'Walking';                                         % Activity 4
    'Going Up/Down Stairs';                            % Activity 5
    'Walking and Talking with Someone';                % Activity 6
    'Talking while Standing'                           % Activity 7
};

axis_descriptions = {
    'duh';
    'X';
    'Y';
    'Z';
};

% Create a folder to save the plots
folder_path = 'C:\Users\giord\OneDrive\Desktop\UNI\MAGISTRALE\Intelligent Systems For Pattern Recognition\Assignments\PatternRecognition\1\plots\';

% Assuming each column is a variable you want to analyze with CWT

% Extract the column data. This method works if the data are numeric.
columnData = table2array(data(:, col));

% Perform the Continuous Wavelet Transform on the column data
[cfs, frequencies] = cwt(columnData, 'bump'); % Example uses the Morlet (analytic Morlet) wavelet

% Plot the CWT result for this column
figure; % Open a new figure window
imagesc(abs(cfs));  % Visualize the absolute value of the CWT coefficients
colorbar;  % Show a color bar indicating the coefficient magnitudes
title(['Complete CWT of ', axis_descriptions{col}, ' axis']);
xlabel('Time');
ylabel('Scale');
axis tight; % Fit the axes tightly around the data

% Save the plot
saveas(gcf, fullfile(folder_path, ['Complete_CWT_', axis_descriptions{col}, '_axis.png']));

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
    [cfs, frequencies] = cwt(columnData, 'amor'); % Example uses the Morlet (analytic Morlet) wavelet

    % Plot the CWT result for this column
    figure; % Open a new figure window
    imagesc(abs(cfs));  % Visualize the absolute value of the CWT coefficients
    colorbar;  % Show a color bar indicating the coefficient magnitudes
    title(['CWT of ', axis_descriptions{col}, ' axis for Activity ', activity_descriptions{activity_label_index}]);
    xlabel('Time');
    ylabel('Scale');
    axis tight; % Fit the axes tightly around the data
    
    % Save the plot
    saveas(gcf, fullfile(folder_path, ['CWT_', axis_descriptions{col}, '_axis_Activity_', num2str(activity_label_index), '.png']));
end
