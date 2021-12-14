function img_list=GenerateList(dir_name,ext_name,list_name)
list=dir(dir_name);
list=list(3:end);
if exist('list_name','var')
    fid=fopen(list_name,'w');
end
img_list={};
count=1;
for i=1:size(list)
    [pathstr, name, ext] = fileparts(list(i).name);
    if strcmp(ext,ext_name)
        if exist('list_name','var')
            fprintf(fid,'%s\\%s\n',dir_name,list(i).name);
        end
        img_list{count}=[dir_name,'\',list(i).name];
        count=count+1;
    end
end

if exist('list_name','var')
    fclose(fid);
end