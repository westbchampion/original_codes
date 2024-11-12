clear all;close all;
le1={'train_data_128','train_data_256','train_data_512'};
le2={'CD11b','MPO','PCNA'};

for k1=1:3
    for k2=1:3
        folderPath =strcat([le1{1,k1},'\',le2{1,k2}]);
        subFolders=GetFolders(folderPath);
        foldnum=length(subFolders);
        for k3=1:foldnum
            subfolderPath=strcat([folderPath,'\',subFolders{1,k3},'\masks']);
            subfiles=GetFiles(subfolderPath);
            
            filenum=length(subfiles);
            for k4=1:filenum
                subfile=strcat([subfolderPath,'\', subfiles{1,k4}]); %待复制文件
                %% 批量处理文件
                %1. 产生文件夹
                    subfold_new=strcat([folderPath,'\',...
                        subFolders{1,k3},'-',num2str(k4)]);
                   pathnew=strcat(['cut\',subfold_new,'\masks']);
                   filen=subfiles{1,k4}(1:end-4);
                   mkdir(pathnew);
                   cutting(subfile,filen,pathnew)
            end
        end
    end
end


function subFolders=GetFolders(folderPath)
% 指定文件夹路径
%folderPath = 'tarin_data_128\CD11b';
% 使用dir函数列出文件夹下的所有内容
contents = dir(folderPath);
% 过滤出文件夹名
subFolders = {contents([contents.isdir]).name};
% 移除 "." 和 ".." 文件夹
subFolders = subFolders(~ismember(subFolders, {'.', '..'}));
% 显示文件夹名
disp('所有子文件夹名列表：');
disp(subFolders);
end

function fileNames=GetFiles(folderPath)
% 指定文件夹路径
%folderPath = 'C:\Path\To\Your\Folder';

% 使用dir函数列出文件夹下的所有文件和文件夹
filesAndFolders = dir(folderPath);

% 过滤出文件名
fileNames = {filesAndFolders(~[filesAndFolders.isdir]).name};

% 显示文件名
disp('文件名列表：');
disp(fileNames);
end

function cuting_fig(imagePath)
% 读取图片
%imagePath = 'your_image.jpg'; % 替换为您的图片路径
image = imread(imagePath);

% 指定切割尺寸
sizes = [128, 256, 512];

% 遍历不同的切割尺寸
for i = 1:numel(sizes)
    sizeX = sizes(i);
    sizeY = sizes(i);
    
    % 计算切割的行数和列数
    numRows = floor(size(image, 1) / sizeX);
    numCols = floor(size(image, 2) / sizeY);

    % 切割图片
    for row = 1:numRows
        for col = 1:numCols
            startX = (row - 1) * sizeX + 1;
            endX = row * sizeX;
            startY = (col - 1) * sizeY + 1;
            endY = col * sizeY;
            
            % 提取子图像
            subImage = image(startX:endX, startY:endY, :);

            % 在这里，您可以对每个子图像进行进一步处理或保存
            % 例如，将子图像保存为文件
            subImageFileName = sprintf('subimage_%dx%d_%d.jpg', sizeX, sizeY, (row - 1) * numCols + col);
            imwrite(subImage, subImageFileName);
        end
    end
end
end