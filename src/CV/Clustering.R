#####
# Objective: getting an idea of potential clusters of pictures
# We use a pre-trained neural network - VGG16 - available from Keras to extract meaningful features 
# We apply k-means on theses features and on the pricipal componnets computed on these features
#####


library(keras)
library(magick) 
library(tidyverse)
library(imager)


#Load model weights without last layer (makes class predictions)
model <- application_vgg16(weights = "imagenet", 
                           include_top = FALSE)


#Image preprocessing - change target size and arrange them into arrays
image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(224, 224))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

# Get List of files
image_files_path <- "House Pictures Example/"
file_list <- list.files(image_files_path, full.names = TRUE, recursive = TRUE)
head(file_list)



#Get synthetic features from VGG16 Network and stack them into a DF
vgg16_feature_list <- data.frame()
for (image in file_list) {
  #image = file_list[1]
  print(image)
  cat("Image", which(file_list == image), "from", length(file_list))
  vgg16_feature <- predict(model, image_prep(image))
  flatten <- as.data.frame.table(vgg16_feature, responseName = "value") %>%
    select(value)
  flatten <- cbind(image, as.data.frame(t(flatten)))
  vgg16_feature_list <- rbind(vgg16_feature_list, flatten)
}

save(vgg16_feature_list, file = "vgg16_feature_list.RData")

#Run PCA to get a view of the cluster on the 2 principal component 
pca <- prcomp(vgg16_feature_list[, -1],
              center = TRUE,
              scale = FALSE)

#Viz of clusters 
data.frame(PC1 = pca$x[, 1], 
           PC2 = pca$x[, 2]) %>%
  ggplot(aes(x = PC1, y = PC2)) +
  geom_point()
#Not much cluster based on first glance

nb_cluster_pca <- 4
nb_cluster_feature <- 4
cluster_pca <- kmeans(pca$x[, 1:10], nb_cluster_pca)
cluster_feature <- kmeans(vgg16_feature_list[, -1], nb_cluster_feature)

