file_path = 'D:\黄梓萌\智能物联网\test-codes\test2\test\';% 图像文件夹路径  
img_path_list = dir(strcat(file_path,'*.jpg'));%获取该文件夹中所有bmp格式的图像  
img_num = length(img_path_list);%获取图像总数量 
% I=cell(1,img_num);
% [max_h,max_w] = size(imread("./division/1_gray.jpg"));
max_w = 0;
max_h = 0;
% I = zeros(img_num,max_w,max_h);
if img_num > 0 %有满足条件的图像  
    for j = 1:img_num %逐一读取图像  
        image_name = img_path_list(j).name;% 图像名  
        image =imread(strcat(file_path,image_name));
        [max_h_c,max_w_c] = size(image);
        if max_h_c > max_h
            max_h = max_h_c;
        end
        if max_w_c > max_w
            max_w = max_w_c;
        end
%         I(j,:,:) = image;
        fprintf('%d %s\n',j,strcat(file_path,image_name));% 显示正在处理的图像名  
        %图像处理过程 省略  
        %这里直接可以访问细胞元数据的方式访问数据
    end
end
fprintf('%d %d\n',max_h,max_w);
