# loading edx & validation datasets that were saved after the R script provider from the capstone course
load("C:/Users/Home/projects/Capstone Course/Train & Validation ds.RData")

library(tidyverse)
library(caret)


# setting seed so that the results are always the same for everyone
set.seed(1,sample.kind = "Rounding")

# Partitioning edx into train and test sets, for tuning
# test set will be 20% of edx set
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)

train_set <- edx[-test_index,]
temp <- edx[test_index,]


# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

########################

# First model is just the average of all the training set
mu<-mean(train_set$rating)

# defining RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# building a dataframe to track the RMSE as we go
# the first row for the first model (The Average)
rmse<-data.frame(method="The Average", RMSE=RMSE(test_set$rating,mu))
rmse

########################

# the second model will be adding the user effect to the average model
# here we remove the average from all the ratings and then calculate the average of every user
user_specific<-train_set %>% mutate(rating=rating-mu) %>%
  group_by(userId) %>% summarise(user_avg=mean(rating))

# extracting the predictions using our model on test set
y_hat_user<-test_set %>% left_join(user_specific) %>% mutate(y_hat=mu+user_avg) %>% pull(y_hat)

# adding another row on our tracking dataframe for RMSE
rmse<-rbind(rmse,data.frame(method="Average + user effect", RMSE=RMSE(test_set$rating,y_hat_user)))
rmse

########################

# the third model will be adding the movie effect to our model
# here we calculate the residual from the last model and then calculate the average of every movie
movie_specific<-train_set %>% left_join(user_specific) %>% mutate(rating=rating-mu-user_avg) %>%
  group_by(movieId) %>% summarise(movie_avg=mean(rating))

# extracting the predictions using our model on test set
y_hat_user_movie<-test_set %>% left_join(user_specific) %>% 
  left_join(movie_specific) %>%
  mutate(y_hat=mu+user_avg+movie_avg) %>% pull(y_hat)

# adding another row on our tracking dataframe for RMSE
rmse<-rbind(rmse,data.frame(method="Average + user effect + Movie effect", RMSE=RMSE(test_set$rating,y_hat_user_movie)))
rmse

########################

# the fourth model will be using regularization on users and movies, 
# to decrease the effect of the few ratings a user or a movie got
# we will use the same paramenter alpha on the users and movies
# here we will simulate the model with different alpha to tune alpha to the min RMSE
alpha_seq=seq(1,30)

rmse_alpha=sapply(alpha_seq, function(a){
  
  user_specific<-train_set %>% mutate(rating=rating-mu) %>%
    group_by(userId) %>% summarise(user_avg=sum(rating)/(n()+a))

  movie_specific<-train_set %>% left_join(user_specific) %>% mutate(rating=rating-mu-user_avg) %>%
    group_by(movieId) %>% summarise(movie_avg=sum(rating)/(n()+a))
  
  y_hat_user_movie<-test_set %>% left_join(user_specific) %>% 
    left_join(movie_specific) %>%
    mutate(y_hat=mu+user_avg+movie_avg) %>% pull(y_hat)
  
  RMSE(test_set$rating,y_hat_user_movie)
})
# ploting the effect of different alpha on RMSE
plot(rmse_alpha~alpha_seq)

# choosing the final alpha
alpha=which.min(rmse_alpha)
alpha

min(rmse_alpha)

# adding another row on our tracking dataframe for RMSE
rmse<-rbind(rmse,data.frame(method="Average + Regularization of (user_specific + Movie_specific) ", RMSE=min(rmse_alpha)))
rmse

# preparing the new user and movie effects using the regularization to move forward
user_specific<-train_set %>% mutate(rating=rating-mu) %>%
  group_by(userId) %>% summarise(user_avg=sum(rating)/(n()+alpha))

movie_specific<-train_set %>% left_join(user_specific) %>% mutate(rating=rating-mu-user_avg) %>%
  group_by(movieId) %>% summarise(movie_avg=sum(rating)/(n()+alpha))


########################

# the final trial model will be using Matrix Factorization on the residual of the last model
# recosystem is a package for recommendation systems, which is fast and efficient on memory usage
library(recosystem)

# calculating the residuals for train & test sets
residual_training<- train_set %>% left_join(user_specific) %>% left_join(movie_specific) %>%
  mutate(res=rating-mu-user_avg-movie_avg)

residual_test<- test_set %>% left_join(user_specific) %>% left_join(movie_specific) %>%
  mutate(res=rating-mu-user_avg-movie_avg)


# preparing the datasource for the model
train_reco <-  with(residual_training, data_memory(user_index = userId, 
                                   item_index = movieId, 
                                   rating = res))
test_reco  <-  with(residual_test, data_memory(user_index = userId, 
                                                  item_index = movieId, 
                                                  rating = res))

# Create the model object
model <-  recosystem::Reco()

# tune the model using the common usage of the method from help file (this will take several minutes)
model$tune(train_reco, opts = list(dim      = c(10L, 20L),
                               costp_l1 = c(0, 0.1),
                               costp_l2 = c(0.01, 0.1),
                               costq_l1 = c(0, 0.1),
                               costq_l2 = c(0.01, 0.1),
                               lrate    = c(0.01, 0.1),
                               nthread = 4, niter = 20))

# training the model 
model$train(train_reco, opts = c( nthread = 4, niter = 20))
# extract of prediction on the residual
residual_prediction <-  model$predict(test_reco, out_memory())

# adding the rest of the moel to the prediction to extract the full prediction
y_hat<-test_set %>% left_join(user_specific) %>% left_join(movie_specific) %>%
  mutate(residual_prediction=residual_prediction) %>% 
  mutate(y_hat=mu+user_avg+movie_avg+residual_prediction) %>%
  pull(y_hat)
# adding another row on our tracking dataframe for RMSE
rmse<-rbind(rmse,data.frame(method="Average + Regularization of (user_specific + Movie_specific) + Matrix Factorization", RMSE=RMSE(test_set$rating,y_hat)))
rmse

########################

# we will go ahead and use the last model as our model of choice
# so we will train the model on all the edx set and then predict the model on validation set to get the final RMSE value

# the new average (on edx)
mu<-mean(edx$rating)
# the new user effect (on edx) using the same alpha 
user_specific<-edx %>% mutate(rating=rating-mu) %>%
  group_by(userId) %>% summarise(user_avg=sum(rating)/(n()+alpha))
# the new movie effect (on edx)using the same alpha 
movie_specific<-edx %>% left_join(user_specific) %>% mutate(rating=rating-mu-user_avg) %>%
  group_by(movieId) %>% summarise(movie_avg=sum(rating)/(n()+alpha))
# repeating the prearation of Matrix Factorization
residual_training<- edx %>% left_join(user_specific) %>% left_join(movie_specific) %>%
  mutate(res=rating-mu-user_avg-movie_avg)

residual_test<- validation %>% left_join(user_specific) %>% left_join(movie_specific) %>%
  mutate(res=rating-mu-user_avg-movie_avg)

train_reco <-  with(residual_training, data_memory(user_index = userId, 
                                                   item_index = movieId, 
                                                   rating = res))
test_reco  <-  with(residual_test, data_memory(user_index = userId, 
                                               item_index = movieId, 
                                               rating = res))

# Create the model object
model <-  recosystem::Reco()
# tune the model using the common usage of the method (this will take several minutes)
model$tune(train_reco, opts = list(dim      = c(10L, 20L),
                                   costp_l1 = c(0, 0.1),
                                   costp_l2 = c(0.01, 0.1),
                                   costq_l1 = c(0, 0.1),
                                   costq_l2 = c(0.01, 0.1),
                                   lrate    = c(0.01, 0.1),
                                   nthread = 4, niter = 20))


# training the model 
model$train(train_reco, opts = c( nthread = 4, niter = 20))
# extract of prediction on the residual
residual_prediction <-  model$predict(test_reco, out_memory())

# the final prediction (on validation)
y_hat<-validation %>% left_join(user_specific) %>% left_join(movie_specific) %>%
  mutate(residual_prediction=residual_prediction) %>% 
  mutate(y_hat=mu+user_avg+movie_avg+residual_prediction) %>%
  pull(y_hat)


# the final result
rmse<-rbind(rmse,data.frame(method="Applying on Validation dataset", RMSE=RMSE(validation$rating,y_hat)))
rmse

list_to_keep<-c("alpha", "rmse", "alpha_seq", "rmse_alpha", "train_set","test_set")
rm(list=setdiff(ls(),list_to_keep))
save.image("C:/Users/Home/projects/Capstone Course/after training.RData")
