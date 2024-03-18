baseFolder = 'C:\Users\giord\OneDrive\Desktop\UNI\MAGISTRALE\Intelligent Systems For Pattern Recognition\Assignments\PatternRecognition\';
dataFolder = fullfile(baseFolder, '1\data\');
plotsBaseFolder = fullfile(baseFolder, '1\plots\');

% Define activity and axis descriptions
activity_descriptions = {
    'Working at Computer';
    'Standing Up, Walking and Going up/down stairs';
    'Standing';
    'Walking';
    'Going Up/Down Stairs';
    'Walking and Talking with Someone';
    'Talking while Standing'
};

axis_descriptions = {
    'duh';
    'X';
    'Y';
    'Z';
};

% Loop through each CSV file
for fileNum = 1:15
    dataFileName = fullfile(dataFolder, [num2str(fileNum), '.csv']);
    data = readtable(dataFileName);
    
    % Create a folder for the current CSV's plots
    currentPlotFolder = fullfile(plotsBaseFolder, ['CSV_', num2str(fileNum)]);
    if ~exist(currentPlotFolder, 'dir')
        mkdir(currentPlotFolder);
    end
    
    % Loop through the columns 2 to 4 (x, y, z)
    for col = 2:4
        % Extract the column data
        columnData = table2array(data(:, col));
        
        % Perform CWT on the column data
        [cfs, frequencies] = cwt(columnData, 'bump');  % Or use 'amor' for Morlet
        
        % Plot and save the complete CWT result
        figure;
        imagesc(abs(cfs));
        colorbar;
        title(['Complete CWT of ', axis_descriptions{col}, ' axis']);
        xlabel('Time');
        ylabel('Scale');
        axis tight;
        
        saveas(gcf, fullfile(currentPlotFolder, ['Complete_CWT_', axis_descriptions{col}, '_axis.png']));
        
        % Extract unique activity labels from the 5th column, skipping the first if it's a header or NaN
        activity_labels = unique(data{:, 5});
        activity_labels = activity_labels(2:end);
        
        % Loop through each activity label
        for activity_label_index = 1:numel(activity_labels)
            activity_data = data(data{:, 5} == activity_labels(activity_label_index), :);
            columnData = table2array(activity_data(:, col));
            
            [cfs, frequencies] = cwt(columnData, 'amor');  % Or 'bump', depending on preference
            
            figure;
            imagesc(abs(cfs));
            colorbar;
            title(['CWT of ', axis_descriptions{col}, ' axis for Activity ', activity_descriptions{activity_label_index}]);
            xlabel('Time');
            ylabel('Scale');
            axis tight;
            
            saveas(gcf, fullfile(currentPlotFolder, ['CWT_', axis_descriptions{col}, '_axis_Activity_', num2str(activity_label_index), '.png']));
        end
    end
end
