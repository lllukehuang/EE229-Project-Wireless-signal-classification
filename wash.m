max_w = 702;
max_h = 317;

file_path = 'D:\黄梓萌\智能物联网\test-codes\test2\test\';% 图像文件夹路径  
img_path_list = dir(strcat(file_path,'*.jpg'));%获取该文件夹中所有bmp格式的图像  
img_num = length(img_path_list);%获取图像总数量
mkdir('cleanv5_blue')
save_path = './cleanv5_blue/';
if img_num > 0 %有满足条件的图像  
    for j = 1:img_num %逐一读取图像
        cur_image = zeros(max_w,max_h,3);
        for m = 1:max_w
            for n = 1:max_h
                cur_image(m,n,:) = [30,144,255];
            end
        end
        image_name = img_path_list(j).name;% 图像名
        image =imread(strcat(file_path,image_name));
        [cur_w,cur_h,channels] = size(image);
%         if cur_w <10 || cur_h < 10
        if cur_w * cur_h < 20
            continue;
        end
        save_name = strcat(save_path,image_name);
        left_length = floor(cur_w/2);
        left_start = 351 - left_length;
%         right_length = cur_w - left_length;
        up_length = floor(cur_h/2);
        up_start = 158 - up_length;
%         down_length = cur_h - up_length;
        for m = 1:cur_w
            for n = 1:cur_h
%                 cur_image(left_start+m,up_start+n,:) = image(m,n,:);
                cur_image(m,n,:) = image(m,n,:);
            end
        end
        cur_image = uint8(cur_image);
        imwrite(cur_image,save_name);
        fprintf('%d %s\n',j,strcat(file_path,image_name));% 显示正在处理的图像名  
    end
end