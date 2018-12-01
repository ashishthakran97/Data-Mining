require(tm)
require(data.table)

#Importing the file
file <- read.csv('file:///E:/ANALYTIXLABS/DATA SCIENCE USING R/BA CLASSES/6. TEXT MINING - CLASSIFICATION/yelp.csv')

#data cleanup
file1 <- file[,-c(1,2,3,6,7)]
apply(is.na(file1[,]),2,sum)


#understanding data
prop.table(table(file1$stars))
file1$type <- ifelse(file1$stars>2,1,0)

file1$stars <- factor(file1$type)

require(caret)
#splitting data
set.seed(50000)
index <- createDataPartition(file1$type,times = 1,p=0.8,list = FALSE)
train <- file1[index,]
test <- file1[-index,]

#CHECKING PROPORTIONS
prop.table(table(train$type))
prop.table(table(test$type))

#RESAMPLING
require(ROSE)
smote_bl <- ROSE(type ~ ., data = train, seed = 5)$data
prop.table(table(smote_bl$type))

#CREATE CORPUS
docs <- Corpus(VectorSource(smote_bl$text))
#summary(docs)
#inspect(docs)             

#CLEAN CORPUS
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stripWhitespace)
#require(SnowballC)
docs <- tm_map(docs, stemDocument, language = "english")
#docs <- tm_map(docs,PlainTextDocument)
#xgboost use xgtree

#UDF TO REMOVE COLUMNS
#toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))

#docs <- tm_map(docs, toSpace, " everyth")
#docs <- tm_map(docs, toSpace, " mani")
#docs <- tm_map(docs, toSpace, " next")
#docs <- tm_map(docs, toSpace, " ever")
#docs <- tm_map(docs, toSpace, " someth")
#docs <- tm_map(docs, toSpace, " last")
#docs <- tm_map(docs, toSpace, " around")
#docs <- tm_map(docs, toSpace, " though")
#docs <- tm_map(docs, toSpace, " big")
#docs <- tm_map(docs, toSpace, " visit")


#feature engineering
#DTM
dtm <-DocumentTermMatrix(docs,control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE))) 
inspect(dtm)



#remove sparse
new_docterm_corpus <- removeSparseTerms(dtm,sparse = 0.92)
colsum <- colSums(as.matrix(new_docterm_corpus))

doc_feature <- data.table(name=attributes(colsum)$names,count=colsum)  #$names
#attributes(colsum)

processed_data <- as.data.table(as.matrix(new_docterm_corpus))
train1 <- cbind(data.table(type=train$type,cool=train$cool,useful=train$useful,funny=train$funny,processed_data))
#train1$out <- relevel(train1$type,ref=0)

#data partitiom
set.seed(50000)
index1 <- createDataPartition(train1$type, times = 1,p = 0.7, list = FALSE)
                             
train2=train1[index1,]
test2=train1[-index1,]

#MODEL BUILDING
require(nnet)

mnomial_logmodel <- nnet::multinom(type~.,data = train2)  

require(mgcv)
saveRDS(mnomial_logmodel,file = 'rating_logistic_model.rda')

#load and validate model
rate_model <- readRDS('E:/tm....ashishthakran97@gmail.com/rating_logistic_model.rda')

#predict
#multinomial logistic regression
pred <- predict(rate_model,test2,type = 'class')

#confusion matrix
cm_categ <- table(predict(rate_model),train2$type)
cm_categ1 <- table(pred,test2$type)

#accuracy for training
accuracy_categ=sum(diag(cm_categ))/sum(cm_categ)  
accuracy_categ   #83.66%

#accuracy for testing.
accuracy_categ1=sum(diag(cm_categ1))/sum(cm_categ1)  
accuracy_categ1  #82.83%
