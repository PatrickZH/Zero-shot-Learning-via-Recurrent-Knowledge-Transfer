function [accuracy,Labels_predict,acc_each] = classifier_nearest(Predicted,Signatures,label_list,groundtruth, k)

N_labels = size(Signatures,1);
Labels_predict = zeros(size(groundtruth,1),k);
for i = 1:size(Predicted,1)
    dis = sum((repmat(Predicted(i,:),N_labels,1)-Signatures).^2,2);
    [~,ind_label] = sort(dis);
    Labels_predict(i,:) = label_list(ind_label(1:k));
end
Ground = repmat(groundtruth,1,k);
Label_predict_equal = sum(Labels_predict == Ground,2);
accuracy = sum(Label_predict_equal >= 1) / length(groundtruth);

