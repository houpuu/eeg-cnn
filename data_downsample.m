close all
clear all
clc

load('E:\PycharmProjects\eeg\data\EEG_orange.mat','train_input','train_output');
train_input1=train_input(:,1:4:end,:);
train_input2=train_input(:,2:4:end,:);
train_input3=train_input(:,3:4:end,:);
train_input4=train_input(:,4:4:end,:);

train_input=[train_input1;train_input2;train_input3;train_input4];
train_output=[train_output;train_output;train_output;train_output];

save('E:\PycharmProjects\eeg\data\EEG_orange_down4.mat','train_input','train_output');
% figure,
% subplot(2,1,1),plot(train_input(1,:,1));
% subplot(2,1,2),plot(train_input1(1,:,1));
% hold on
% plot(train_input2(1,:,1));

