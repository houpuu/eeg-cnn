%���ݼ��Ǵ�����ĵȳ�����
%�������ⳤ���ݣ�
%���������������sample_ratio,��������sample_length;
% ���Ϊ�������������������ȣ�ͨ������

close all
clear all
clc

sample_ratio=4;
sample_length=250;
S_length=sample_ratio*sample_length;

train_input=[];
train_output=[];

file_name='C:\Users\bohua\OneDrive\eeg\eeg_DATA\2017.12.12ʵ�����ݣ����֣�\����\����\i1';

input_option.type='dir';
input_option.ext='.mat';
pos_file_list=FileInput(file_name,input_option);
length_data=length(pos_file_list.data);


for k=1:length_data
    
    S_signalname=char(pos_file_list.data(1,k));
    load(S_signalname);
    a=squeeze(EEG(1,:));
    L=fix(size(EEG,2)/(sample_length*sample_ratio));
    EEG=EEG';
    for i=1:sample_ratio
        for j=1:L
            input((i-1)*L+j,:,:)=EEG((j-1)*S_length+i:sample_ratio:j*S_length,:);
        end
    end
     train_input=[train_input;input];
     y1=k-1;
     train_output=[train_output;repmat(y1,size(input,1),1)];
end

save('C:\PycharmProjects\eeg\data\shuzi\i_number_250_1.mat','train_input','train_output');



