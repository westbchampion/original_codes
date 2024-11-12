clear all;close all;clc
le1={'train_data_128','train_data_256','train_data_512'};
le2={'CD11b','MPO','PCNA'};

for k1=1:3
    for k2=1:3
        folderPath =strcat([le1{1,k1},'\',le2{1,k2}]);
        subFolders=GetFolders(folderPath);
        foldnum=length(subFolders);
        for k3=1:foldnum
            subfolderPath=strcat([folderPath,'\',subFolders{1,k3},'\images']);
            subfiles=GetFiles(subfolderPath);
            
            filenum=length(subfiles);
            for k4=1:filenum
                subfile=strcat([subfolderPath,'\', subfiles{1,k4}]); %待复制文件
                subfold=strcat([folderPath,'\',subFolders{1,k3},'\masks\*.*']); %待复制文件夹
                %% 批量处理文件
                %1. 产生文件夹
                    subfold_new=strcat([folderPath,'\',...
                        subFolders{1,k3},'-',num2str(k4)]);
                   pathnew1=strcat(['danew\',subfold_new,'\images']);
                   pathnew2=strcat(['danew\',subfold_new,'\masks']);
                  
                   mkdir(pathnew1);
                   copyfile(subfile,pathnew1);
                   mkdir(pathnew2);
                   copyfile(subfold,pathnew2);
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


