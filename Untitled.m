close all
clear all
clc


% addpath('E:\eeg_lab\图片诱发情绪的数据包\数据包\数据包');
% load('wj-exin3.mat');

datax=[];
datay=[];

load('E:\PycharmProjects\eeg\data\EEG_music_sad0.mat','train_input','train_output');
datax=[datax;train_input];
datay=[datay;train_output];

load('E:\PycharmProjects\eeg\data\EEG_music_excited1.mat','train_input','train_output');
datax=[datax;train_input];
datay=[datay;train_output];

load('E:\PycharmProjects\eeg\data\EEG_music_happy2.mat','train_input','train_output');
datax=[datax;train_input];
datay=[datay;train_output];

load('E:\PycharmProjects\eeg\data\EEG_music_peace3.mat','train_input','train_output');
datax=[datax;train_input];
datay=[datay;train_output];

load('E:\PycharmProjects\eeg\data\EEG_music_terrifying4.mat','train_input','train_output');
datax=[datax;train_input];
datay=[datay;train_output];

% load('E:\PycharmProjects\eeg\data\EEG_picture_spider5.mat','train_input','train_output');
% datax=[datax;train_input];
% datay=[datay;train_output];

train_input=datax;
train_output=datay;


save('E:\PycharmProjects\eeg\data\EEG_music.mat','train_input','train_output');


