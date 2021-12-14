%数据集是处理过的等长数据
%输入任意长数据，
%将其间隔采样，间隔sample_ratio,样本长度sample_length;
% 输出为（样本个数，样本长度，通道数）

close all
clear all
clc

addpath('C:\FangCloudV2\personal_space\eeg_program\function');

sample_ratio=5;
sample_length=200;
S_length=sample_ratio*sample_length;

train_input=[];
train_output=[];

file_name='C:\Users\bohua\OneDrive\eeg\eeg_DATA\2017.12.12实验数据（数字）\可用\数字\s2';
save_name='C:\PycharmProjects\eeg\data\shuzi\s_number_200_2.mat';

input_option.type='dir';
input_option.ext='.mat';
pos_file_list=FileInput(file_name,input_option);
length_data=length(pos_file_list.data);


for k=1:length_data
    
    S_signalname=char(pos_file_list.data(1,k));
    load(S_signalname);
    EEG1=function_IIR_bandpass_16(EEG,45,50,1000);
    L=fix(size(EEG1,2)/(sample_length*sample_ratio));
    EEG1=EEG1(:,21:end);
    EEG1=EEG1';
    for i=1:sample_ratio
        for j=1:L
            input((i-1)*L+j,:,:)=EEG1((j-1)*S_length+i:sample_ratio:j*S_length,:);
        end
    end
     train_input=[train_input;input];
     y1=k-1;
     train_output=[train_output;repmat(y1,size(input,1),1)];
end
save(save_name,'train_input','train_output');



