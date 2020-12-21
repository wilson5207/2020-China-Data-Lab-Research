library(stm)
library(quanteda)
library(dplyr)
library(readr)
library(stringr)
library(wordcloud)

#Author: Yifan Huang, Che-wei Lin, Xueru Xie
setwd("/Users/16413/Desktop")

news <- read.csv("newspaper_data.csv",header = F)
colnames(news) = c("id","newspaper","title","date","content")


set.seed(92037)
news$rnd = rnorm(29432)
labelsample = news[order(news$rnd),]
labelsample = labelsample[1:500,]
write.csv(labelsample,file = "labelsample.csv")


#separating sets
labeled_sample = read.csv("labeled_sample.csv")[,2:8]
news$protest_related = NA

news_labeled = rbind(labeled_sample, news)
news_labeled = news_labeled[!duplicated(news_labeled$id), ]

news_labeled$content <- as.character(news_labeled$content)

corpus = corpus(news_labeled$content,docvars=news_labeled)

tok <- tokens(corpus, what = "fasterword")
doc.features <- dfm(tok, 
                    remove=c(stopwords(language = "zh", source = "misc"), "https", "t.co", "rt"), 
                    remove_numbers = TRUE,
                    stem=T, remove_punct=T,
                    min_termfreq = 10)

textplot_wordcloud(doc.features,
                   font = "HYQuanTangShiJ", max_words = 200)







#supervised
docvars(corpus, "id_numeric") <- 1:ndoc(corpus)


test <- which(is.na(news_labeled$protest_related))
labeled <- which(!is.na(news_labeled$protest_related))
training <- sample(labeled, round(length(labeled)*0.75))
validation <- labeled[!labeled%in%training]

dfmat_train <- doc.features[training,]
dfmat_val <- doc.features[validation,]
dfmat_test <- doc.features[test,]

traindata <- convert(dfmat_train, "data.frame")[,c(-1)]
valdata <- convert(dfmat_val, "data.frame")[,c(-1)]

tmod_nb <- textmodel_nb(dfmat_train, docvars(dfmat_train, "protest_related"))
predict.train.nb <- predict(tmod_nb, dfmat_train)

predict.train.nb <- as.numeric(predict.train.nb)-1

nb_train <- table(docvars(dfmat_train, "protest_related"), predict.train.nb)

#recall
diag(nb_train)/colSums(nb_train)
#precision
diag(nb_train)/rowSums(nb_train)


#How well does this prediction do out of sample?  Validation
predict.val.nb <- predict(tmod_nb, newdata = dfmat_val)

tab_val <- table(docvars(dfmat_val, "protest_related"), predict.val.nb)
tab_val

#recall
diag(tab_val)/colSums(tab_val)
#precision
diag(tab_val)/rowSums(tab_val)



#apply on test
predict.test.nb <- predict(tmod_nb, newdata = dfmat_test)
test_news <- as.data.frame(predict.test.nb)

test_news$predict.test.nb <- as.numeric(test_news$predict.test.nb)-1
news_labeled_2 <- news_labeled
news_labeled_2$protest_related[501:29432] <- test_news$predict.test.nb
news_labeled_2 <- news_labeled_2[order(news_labeled_2$id),]
news_labeled_2$dummy <- 1



# plotting: grouping by week
news_labeled_2$date <- as.Date(news_labeled_2$date)
news_labeled_2$week <- as.Date(cut(news_labeled_2$date,breaks = "week"))
news_labeled_2$month <- as.Date(cut(news_labeled_2$date,breaks = "month"))

protest_news_byweek <- tapply(news_labeled_2$protest_related, news_labeled_2$week, sum)
protest_news_freqbyweek <- protest_news_byweek/(tapply(news_labeled_2$dummy, news_labeled_2$week, sum))
par(mar = c(5, 4, 4, 2) + 0.1)

plot(as.Date(names(protest_news_byweek)), 
     protest_news_byweek, 
     type="l", 
     main = "Protest News by Week", 
     xlab=" ",
     ylab="Number of Piece of news")
abline(v = as.Date("2019-6-9"), col = "blue")


# news concentration rate: cosine distance
news_labeled_2$simil = NA
simil = data.frame("date" = seq(as.Date("2019-3-15"), as.Date("2019-12-29"), by=1), "similarity" = NA)
for (i in seq(as.Date("2019-3-15"), as.Date("2019-12-29"), by=1)){
  news_sub = doc.features[news_labeled_2$date == i,]
  news_sub = rbind(news_sub,doc.features[news_labeled_2$date == (i+1),])
  news_sub = rbind(news_sub,doc.features[news_labeled_2$date == (i+2),])
  simil_index = as.dist(textstat_dist(news_sub))
  simil$similarity[simil$date == i] = mean(simil_index)
}

plot(as.Date(simil$date), 
     simil$similarity, 
     type="l", 
     main = "News Similarity by Day", 
     xlab=" ",
     ylab="Average Cosine Similarity across the day")
abline(v = as.Date("2019-6-9"), col = "blue")







#topic models

out <- convert(doc.features, to = "stm", docvars = news)
out$meta$date <- as.Date(out$meta$date, format="%Y-%m-%d")
hist(out$meta$date, breaks="day")

#Run STM model
stm.out = stm(out$documents, out$vocab, K=10, 
              prevalence = ~ s(date), 
              data=out$meta, init.type="Spectral")

labelTopics(stm.out)

