file_path = 'D:\黄梓萌\智能物联网\test-codes\cleanv3\';% 图像文件夹路径  
img_path_list = dir(strcat(file_path,'*.jpg'));%获取该文件夹中所有bmp格式的图像  
img_num = length(img_path_list);%获取图像总数量 
% I=cell(1,img_num);
max_h = 317;
max_w = 702;
I = zeros(img_num,max_w,max_h,3);
if img_num > 0 %有满足条件的图像  
    for j = 1:img_num %逐一读取图像  
        image_name = img_path_list(j).name;% 图像名  
        image =imread(strcat(file_path,image_name));
        I(j,:,:,:) = image;
        fprintf('%d %s\n',j,strcat(file_path,image_name));% 显示正在处理的图像名  
        %图像处理过程 省略  
        %这里直接可以访问细胞元数据的方式访问数据
    end
end
I = reshape(I,img_num,max_w*max_h*3);
[idx,C] = kmeans(I,9);


file_path_origin = 'D:\黄梓萌\智能物联网\test-codes\test2\test\';% 图像文件夹路径  
img_path_list_origin = dir(strcat(file_path_origin,'*.jpg'));%获取该文件夹中所有bmp格式的图像 
mkdir('resultv5_9');
for i = 1:img_num
    cur_idx = idx(i,1);
%     image_name = img_path_list_origin(i).name;% 图像名
    image_name = img_path_list(i).name;% 图像名
    image =imread(strcat(file_path_origin,image_name));
    dir_path = sprintf('resultv5_9/%d/',cur_idx);
    mkdir(dir_path);
    cur_path = sprintf('resultv5_9/%d/%s',cur_idx,image_name);
    fprintf('%d %s\n',j,strcat(file_path_origin,image_name));% 显示正在处理的图像名  
    imwrite(image,cur_path);
end