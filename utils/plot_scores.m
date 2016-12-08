function [ ] = plot_scores( test_scores )
%ACC_PLOT plots Loss and Accuracy vs Epoch 

m = csvread(test_scores);
epochs = length(m);
loss = m(:, 1);
acc = m(:, 2);

%Plot Loss
figure % new figure
ax1 = subplot(2,1,1); % top subplot
ax2 = subplot(2,1,2); % bottom subplot

plot(ax1,1:epochs,loss,'-o');
title(ax1,'Loss')

%Plot accuracy
plot(ax2, 1:epochs,acc,'-o');
title(ax2,'Accuracy')

axis([ax1 ax2],[0 20 0 1]);


% Compute best model Loss and Accuracy
best_acc = max(acc);
best_loss = min(loss);

fprintf('Accuracy: %f \n',best_acc);
fprintf('Loss: %f \n',best_loss);

end
