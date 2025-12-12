% Set the root folder containing subfolders with branch and trunk OBJ files
rootFolder = 'D:\Data\LPy\LTB81_Test\new_tree\obj\new_tree_json';

% Initialize an empty cell array to store trees
trees = {};

% Define X-axis offset for each tree
xOffset = 1; % Adjust as needed

% Loop through all subfolders in the root folder
subfolders = dir(rootFolder);
for i = 1:numel(subfolders)
    if subfolders(i).isdir && ~strcmp(subfolders(i).name,'.') && ~strcmp(subfolders(i).name,'..')
        folderName = fullfile(rootFolder, subfolders(i).name);
        branchFolder = fullfile(folderName, 'branch');
        trunkFolder = fullfile(folderName, 'trunk');
        
        % Check if both branch and trunk folders exist
        if exist(branchFolder, 'dir') && exist(trunkFolder, 'dir')
            % Read OBJ files in the branch folder
            branchFiles = dir(fullfile(branchFolder, '*.obj'));
            branchVertices = [];
            for j = 1:numel(branchFiles)
                branchVertices = [branchVertices; readObjFile(fullfile(branchFolder, branchFiles(j).name))];
            end
            
            % Read OBJ files in the trunk folder
            trunkFiles = dir(fullfile(trunkFolder, '*.obj'));
            trunkVertices = [];
            for j = 1:numel(trunkFiles)
                trunkVertices = [trunkVertices; readObjFile(fullfile(trunkFolder, trunkFiles(j).name))];
            end
            
            % Apply X-axis offset to tree vertices
            branchVertices(:,1) = branchVertices(:,1) + xOffset*(i-3);
            trunkVertices(:,1) = trunkVertices(:,1) + xOffset*(i-3);
            
            % Assemble the tree
            trees{end+1} = struct('branch', branchVertices, 'trunk', trunkVertices);
            
            % Visualize assembled tree
            % Visualize branch
            scatter3(branchVertices(:,1), branchVertices(:,2), branchVertices(:,3), '.', 'b');
            hold on;
            
            % Visualize trunk
            scatter3(trunkVertices(:,1), trunkVertices(:,2), trunkVertices(:,3), '.', 'g');
            
            % Additional visualization settings (labels, etc.) can be added here
            
            xlabel('X');
            ylabel('Y');
            zlabel('Z');
            title('Assembled Tree Visualization');
            grid on;
            axis equal;
            
            % You can choose to pause or wait for a key press between visualizations
            % pause;
            % input('Press Enter to continue...');
        end
    end
end

%% Set up the video writer
video_filepath = fullfile("C:\Users\tq42\OneDrive - Cornell University\Tree_Completion_2024", "LPy_Tree_Video_V4.avi");
writerObj = VideoWriter(video_filepath); % specify the file name and format
writerObj.FrameRate = 10; % set the frame rate (frames per second)
open(writerObj); % open the video writer

% capture each frame of the plot and write it to the video file
for t = 1:100 % Change the range as needed
    % rotate the plot for each frame (optional)
    view(3*t, 20);
    % capture the current frame
    frame = getframe(gcf);
    % write the frame to the video file
    writeVideo(writerObj, frame);
end
close(writerObj); % close the video writer

%%
function vertices = readObjFile(filepath)
    % Function to read OBJ file and extract vertices
    fid = fopen(filepath, 'r');
    vertices = [];
    while ~feof(fid)
        line = fgetl(fid);
        if startsWith(line, 'v ')
            vertex = sscanf(line, 'v %f %f %f');
            vertices = [vertices; vertex'];
        end
    end
    fclose(fid);
end
