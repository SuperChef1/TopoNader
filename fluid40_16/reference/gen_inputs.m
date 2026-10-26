% n=641;
% seed_gen = 3;
% base=[];
% nx=40;
% ny=16;
% inputs = func_inputs_gen(n,seed_gen,base,nx,ny);
% outputs= func_outputs_nowrite(inputs,nx,ny);

% base=single(zeros(10, 640));


% contrast

clc;
clear;
% input_all=[];
% for i=1:100
%     input=randi([0, 1], 2000000, 640);
%     % input=double(rand(20000, 640)>0.9);
%     [sa,sb]=size(input);
%     rowsToDelete = [];
%     for k=1:sa
%         import py.utils.isvalid
%         if ~isvalid(reshape(input(k,:),40,16)') 
%             rowsToDelete = [rowsToDelete, k]; 
%         end
%     end
%     input(rowsToDelete,:)=[];
%     input_all=[input_all;input];
% end
% input=input_all;

input=double(rand(20000, 640)>0.8);
[sa,sb]=size(input);
rowsToDelete = [];
for k=1:sa
    import py.utils.isvalid
    if ~isvalid(reshape(input(k,:),40,16)') 
        rowsToDelete = [rowsToDelete, k]; 
    end
end
input(rowsToDelete,:)=[];

inputs=reshape(inputs,size(inputs,1),640);
input=[inputs;input];
inputs=reshape(input(1:11000,:),11000,40,16);
outputs= func_outputs_nowrite(inputs,40,16);

index=find(outputs==inf);
inputs(index,:,:)=[];
outputs(index,:)=[];

