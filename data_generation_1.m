%数据集是处理过的不等长数据　格式为１６＊ＸＸ

close all
clear all
clc

% train_input=[];
% train_output=[];

train_x=[];
train_y=[];

file_name='C:\Users\bohua\OneDrive\eeg\eeg_DATA\2017.12.12实验数据（数字）\可用\数字\i1';

input_option.type='dir';
input_option.ext='.mat';
pos_file_list=FileInput(file_name,input_option);
length_data=length(pos_file_list.data);
length_N=1000;
star_n=0;

for k=1:length_data
    
    S_signalname=char(pos_file_list.data(1,k));
    load(S_signalname);
    dimention=length(size(EEG));
   
    
    switch dimention
        case 3
            train_x=[train_x;EEG];
            train_y=[train_y;train_output];
        case 2
            data=double(EEG);
            kk=fix(size(data,2)/length_N);
            for i=1:kk-1
                star_n=star_n+1;
                EEG_data=data(:,(i-1)*length_N+1:i*length_N);
                train_input(:,:,star_n)=[EEG_data];
                train_output(star_n,1)=k-1;
            end
    end
     
end

switch dimention
        case 2
            train_input=permute(train_input,[3,2,1]);
%             train_output=[repmat(0,size(train_input,1),1)];


% load('E:\PycharmProjects\eeg\data\picture\picture1000\EEG_picture_spider5.mat');
% % train_input=[train_input;EEG_data];
% train_output=[repmat(5,size(train_input,1),1)];
% 
            train_input1=train_input(:,1:4:end,:);
            train_input2=train_input(:,2:4:end,:);
            train_input3=train_input(:,3:4:end,:);
            train_input4=train_input(:,4:4:end,:);
% 
            train_input=[train_input1;train_input2;train_input3;train_input4];
            train_output=[train_output;train_output;train_output;train_output];

    case 3
        
            train_input=train_x;
            train_output=train_y;
end
% 
%  save('E:\PycharmProjects\eeg\data\shuzi\s_number_ver2.mat','train_input','train_output');
%  save('E:\eeg_lab\2017.12.12实验数据（数字）\可用\数字\s_number_ver2.mat','train_input','train_output')

a=squeeze(train_input(:,:,1));
