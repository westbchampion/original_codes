 function cutting(imagePath,filen,pathnew)
% 读取图片
%imagePath = subfile; % 替换为您的图片路径
image = imread(imagePath);

% 指定切割尺寸
sizes = [128, 256, 512];

% 遍历不同的切割尺寸
for i = 1:numel(sizes)
    sizeX = sizes(i);
    sizeY = sizes(i);
    
    % 计算切割的行数和列数
    numRows = floor(size(image, 1) / sizeX)+1;
    numCols = floor(size(image, 2) / sizeY)+1;

    % 切割图片
    for row = 1:numRows
        for col = 1:numCols
            startX = (row - 1) * sizeX + 1;
            endX = row * sizeX;
            startY = (col - 1) * sizeY + 1;
            endY = col * sizeY;
            if row<numRows && col<numCols 
                subImage = image(startX:endX, startY:endY, :);
            end
            if row<numRows && col==numCols 
                subImage = image(startX:endX,832-sizeX:end, :);
            end
            if row==numRows && col<numCols
                subImage = image(832-sizeX:end,startY:endY, :);
            end
            if col==numCols  && row==numRows
             subImage = image(832-sizeX:end,832-sizeX:end, :);
            end 

            % 在这里，您可以对每个子图像进行进一步处理或保存
            % 例如，将子图像保存为文件
            subImageFileName = sprintf('subimage_%dx%d_%d.jpg', sizeX, sizeY, (row - 1) * numCols + col);
            imwrite(subImage, strcat([pathnew,'\',num2str(filen),subImageFileName]));
        end
    end
end
 end