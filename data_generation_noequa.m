%数据集是处理过的不等长数据　格式为１６＊ＸＸ

close all
clear all
clc

train_input=[];
train_output=[];

file_name='E:\eeg_lab\color_blue';

input_option.type='dir';
input_option.ext='.mat';
pos_file_list=FileInput(file_name,input_option);
length_data=length(pos_file_list.data);
length_N=1000;
% length_data=1;
star_n=0;

for k=1:length_data
    
    S_signalname=char(pos_file_list.data(1,k));
    load(S_signalname);
    dimention=length(size(EEG_data));
    switch dimention
        case 3
            train_input=[train_input;EEG_data];
            train_output=[train_output;repmat(0,size(EEG_data,1),1)];
            star_n=star_n+1;
        case 2
            kk=fix(size(data,2)/length_N);
            for i=1:kk-1
                star_n=star_n+1;
                EEG_data=data(:,(i-1)*length_N+1:i*length_N);
                train_input(:,:,star_n)=[EEG_data];
            end
                y1=0;
                train_output=[train_output;repmat(y1,size(EEG_data,1),1)];
    end
end
% train_output=[train_output;repmat(y1,size(EEG_data,1),1)];
train_input=permute(train_input,[1,3,2]);
% %train_output=[repmat(0,size(train_input,1),1)];

save('E:\PycharmProjects\eeg\data\EEG_blue.mat','train_input','train_output');

% load('E:\PycharmProjects\eeg\data\EEG_color2.mat','train_input','train_output');
% x=train_input;
% y=train_output;
% load('E:\PycharmProjects\eeg\data\EEG_color2.mat','train_input','train_output');
% train_input=[train_input;x];
% train_output=[train_output;y];
% save('E:\PycharmProjects\eeg\data\EEG_color.mat','train_input','train_output');

