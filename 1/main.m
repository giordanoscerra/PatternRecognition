% Define relative paths
dataFolder = fullfile('data');
plotsBaseFolder = fullfile('plots');

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

% Define axis descriptions
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
        [cfs, frequencies] = cwt(columnData, 'bump');  % Here we use bump for more localized results
        
        % Plot and save the complete CWT result
        figure;
        imagesc(abs(cfs));
        colorbar;
        title(['Complete CWT of ', axis_descriptions{col}, ' axis']);
        xlabel('Time');
        ylabel('Scale');
        axis tight;        
        saveas(gcf, fullfile(currentPlotFolder, ['Complete_CWT_', axis_descriptions{col}, '_axis.png']));
        
        % Extract unique activity labels from the 5th column, skipping the first (0 value is for the end of the file)
        activity_labels = unique(data{:, 5});
        activity_labels = activity_labels(2:end);
        
        % Loop through each activity label
        for activity_label_index = 1:numel(activity_labels)
            % Extract the column data for the current activity label
            activity_data = data(data{:, 5} == activity_labels(activity_label_index), :);
            columnData = table2array(activity_data(:, col));
            
            % Here we can use Morlet for more frequency localized results
            [cfs, frequencies] = cwt(columnData, 'amor'); 
            
            % Plot and save the complete CWT result
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
