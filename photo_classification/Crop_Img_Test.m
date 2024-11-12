clc
clear all
close all
Img_Path=dir('./Big_Img_Test\background/*.tif');
step=80;
New_Width=128;
New_Height=128;
tt=1;
figure;
for i=1:length(Img_Path)
    Center_x=ceil(New_Width/2)+1;
    Center_y=ceil(New_Height/2)+1;
    Img=imread(strcat(Img_Path(i).folder,'/',Img_Path(i).name));
    [h,w,~]=size(Img);
    while 1
        Center_x=ceil(New_Width/2)+1;
        if (Center_y-floor(New_Height/2)<1||Center_y+floor(New_Height/2)>h)
            break;
        end
        while 1
            if (Center_x+floor(New_Width/2)>w||Center_x-floor(New_Width/2)<1)
                break;
            end
            Img_Crop=Img(Center_y-floor(New_Height/2):Center_y+floor(New_Height/2)-1,Center_x-floor(New_Width/2):Center_x+floor(New_Width/2)-1,:);
%             imshow(Img_Crop);
            imwrite(Img_Crop,strcat('./data/temp_file_test/background/',num2str(tt),'.jpg'));
%             pause(0.2);
            Center_x=Center_x+step;
            tt=tt+1;
        end
        Center_y=Center_y+step;
    end
end

% 
% Img_Path=dir('./Big_Img_Test\muscle/*.tif');
% tt=1;
% figure;
% for i=1:length(Img_Path)
%     Center_x=ceil(New_Width/2)+1;
%     Center_y=ceil(New_Height/2)+1;
%     Img=imread(strcat(Img_Path(i).folder,'/',Img_Path(i).name));
%     [h,w,~]=size(Img);
%     while 1
%         Center_x=ceil(New_Width/2)+1;
%         if (Center_y-floor(New_Height/2)<1||Center_y+floor(New_Height/2)>h)
%             break;
%         end
%         while 1
%             if (Center_x+floor(New_Width/2)>w||Center_x-floor(New_Width/2)<1)
%                 break;
%             end
%             Img_Crop=Img(Center_y-floor(New_Height/2):Center_y+floor(New_Height/2)-1,Center_x-floor(New_Width/2):Center_x+floor(New_Width/2)-1,:);
% %             imshow(Img_Crop);
%             imwrite(Img_Crop,strcat('./data/temp_file_test/background/',num2str(tt),'.jpg'));
% %             pause(0.2);
%             Center_x=Center_x+step;
%             tt=tt+1;
%         end
%         Center_y=Center_y+step;
%     end
% end








clc
clear all
close all
Img_Folder=dir('./data/temp_file_test/');

train_Index=0;
val_Index=0;
test_Index=0;
for j=3:length(Img_Folder)
    Img_Path=dir(strcat(Img_Folder(j).folder,'/',Img_Folder(j).name,'/*.jpg'));
% %     Index=1:length(Img_Path);
%     Index=randperm(length(Img_Path));
%     train=Index(1:round(0.8*length(Img_Path)));
%     val=Index(round(0.8*length(Img_Path))+1:round(0.9*length(Img_Path)));
%     test=Index(round(0.9*length(Img_Path))+1:end);

    for i=1:length(Img_Path)
        Img=imread(strcat(Img_Path(i).folder,'/',Img_Path(i).name));
        imwrite(Img,strcat('./data/test_yanshou/',num2str(train_Index),'_',Img_Folder(j).name,'.jpg'));
        train_Index=train_Index+1;
    end
end





