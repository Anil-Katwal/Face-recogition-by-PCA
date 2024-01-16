clear;
% Load the face images and preprocess them
train_data = load('train.mat');
test_data = load('test.mat');
X_train = double(train_data.train);
X_test = double(test_data.test);
m = mean(X_train, 2);
Train_Number = size(X_train, 2);
temp_m = repmat(m, 1, Train_Number);
A = X_train - temp_m;
L = (A' * A)/(Train_Number-1);
[V, D] = eig(L);
Eig_vec = [];
threshold = 1000;

for i = 1:size(V, 2)
    if (D(i, i) > threshold)
        Eig_vec = [Eig_vec V(:, i)];
    end
end

Eigenfaces = A * Eig_vec;

ProjectedImages = Eigenfaces' * A;
Euc_dist = zeros(Train_Number, size(X_test, 2));
Recognized_index_per_image = zeros(1, size(X_test, 2));

for img = 1:size(X_test, 2)
    InputImage = X_test(:, img);
    Difference = InputImage - m;
    Projected_TestImage = Eigenfaces' * Difference;
    for i = 1:Train_Number
        q = ProjectedImages(:, i);
         q = q / norm(q);
        temp = norm(Projected_TestImage - q)^2;
        Euc_dist(i, img) = temp;
    end
    [~, Recognized_index_per_image(img)] = min(Euc_dist(:, img));
end
for i = 1:size(X_test, 2)
    figure;

    % Original image
    subplot(1, 2, 1);
    imshow(uint8(reshape(X_test(:, i), [112, 92])));
    title(['Test Image ' num2str(i)]);

    % Recognized face
    subplot(1, 2, 2);
    recognized_face_index = Recognized_index_per_image(i);
    recognized_face = reshape(X_train(:, recognized_face_index), [112, 92]);
    imshow(uint8(recognized_face));
    title(['Recognized as Image ' num2str(recognized_face_index)]);
end

% Display eigenfaces
Num_Eigenvalue = size(Eigenfaces, 2);
figure('Name', 'Eigenface');
for i = 1:min(Num_Eigenvalue, 9)
    img = reshape(Eigenfaces(:, i), [112, 92]);
    subplot(3, 3, i);
    imshow(img', []);
end
