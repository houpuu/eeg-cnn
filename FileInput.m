function orig_data=FileInput(filename,file_option)
switch file_option.type
    case 'single'
        orig_data.type='single';
        orig_data.data=imread(filename);
    case 'dir'
        img_list=GenerateList(filename,file_option.ext);
        orig_data.type='lst';
        orig_data.data=img_list;
    case 'lst'
        fid=fopen(filename,'r');
        img_list={};
        [list,count]=fgets(fid);
        i=1;
        while(fid>0 && ~isempty(count))
            img_list{i}=list;
            i=i+1;
            [list,count]=fgets(fid);
        end
        orig_data.type='lst';
        orig_data.data=img_list;
    otherwise
        error('wrong file type');
end
