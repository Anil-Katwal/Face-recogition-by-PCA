clear;

% Load the face images and preprocess them
train_data = load('train.mat');
test_data = load('test.mat');
X_train = double(train_data.train);
X_test = double(test_data.test);
m = mean(X_train, 2);
A = X_train - repmat(m, 1, size(X_train, 2));
num_components = 80; 
[coeff, ~, ~, ~, explained] = pca(A', 'NumComponents', num_components);
ProjectedImages = A' * coeff;

% Project the test images onto the subspace
Difference = X_test - repmat(m, 1, size(X_test, 2));
Projected_TestImages = Difference' * coeff;

% Initialize matrices for distances and recognized indices
Euc_dist = pdist2(ProjectedImages, Projected_TestImages, 'euclidean');
[~, Recognized_index_per_image] = min(Euc_dist);

% Display the original images, recognized faces, and eigenfaces
for i = 1:size(X_test, 2)
    figure;

    % Original image
    subplot(1, 3, 1);
    imshow(uint8(reshape(X_test(:, i), [112, 92])));
    title(['Test Image ' num2str(i)]);

    % Recognized face
    subplot(1, 3, 2);
    recognized_face_index = Recognized_index_per_image(i);
    recognized_face = reshape(X_train(:, recognized_face_index), [112, 92]);
    imshow(uint8(recognized_face));
    title(['Recognized as Image ' num2str(recognized_face_index)]);

    % Display eigenface
    subplot(1, 3, 3);
    eigenface = reshape(coeff(:, i), [112, 92]);
    eigenface = eigenface - min(eigenface(:));
    eigenface = eigenface / max(eigenface(:));
    imshow(eigenface, []);
    title(['Eigenface ' num2str(i)]);
end
