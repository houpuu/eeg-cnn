%���ݼ��Ǵ�����ĵȳ�����
%�������ⳤ���ݣ�
%���������������sample_ratio,��������sample_length;

close all
clear all
clc

sample_ratio=10;
sample_length=100;

train_input=[];
train_output=[];

file_name='C:\Users\bohua\OneDrive\eeg\eeg_DATA\2017.12.12ʵ�����ݣ����֣�\����\����\i1';

input_option.type='dir';
input_option.ext='.mat';
pos_file_list=FileInput(file_name,input_option);
length_data=length(pos_file_list.data);


for k=1:1%length_data
    
    S_signalname=char(pos_file_list.data(1,k));
    load(S_signalname);
    L=size(EEG,2);
%     train_input=[train_input;EEG_data];
%     y1=k-1;
%     train_output=[train_output;repmat(y1,size(EEG_data,1),1)];
end

% train_input=permute(train_input,[1,3,2]);
% save('C:\PycharmProjects\eeg\data\color_cly.mat','train_input','train_output');

figure,
a=EEG(1,:)
plot(a);
