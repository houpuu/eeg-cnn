close all
clear all
clc

x=[];
y=[];

load('E:\eeg_lab\number_i\EEGi0.mat');

% load('E:\PycharmProjects\eeg\data\shuzi\EEG_number_down4.mat','train_input','train_output');
% %------------------------------------------
% x=[x;train_input];
% y=[y;train_output];
%-------------------------------------------------
% 2 ,picture
% load('E:\PycharmProjects\eeg\data\shuzi\s_number.mat','train_input','train_output');
% x=[x;train_input];
% y=[y;train_output];
% 
% 
% train_input=x;
% train_output=y;

% k=fix(rand*5000)+1;
a=squeeze(EEG_data(1,1,:));
figure,
plot(a);
% save('E:\PycharmProjects\eeg\data\shuzi\all_number_ver1.mat','train_input','train_output');
