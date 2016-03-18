ROOT = "/home/chase/Dropbox/streetview-classifier"
setwd(ROOT)
require(stringr)

data = read.csv('data/2016_tax_sale_list_cleaned.csv',header=TRUE,stringsAsFactor=FALSE,sep=",")
data = subset(data,select=c('street_view_rating','address','parcel_number'))

# re-label street_view_ratings to 0=bad house, 1=good house, 2=un-labeled
data$street_view_rating[data$street_view_rating < 2] = 0
data$street_view_rating[data$street_view_rating > 2] = 1
data$street_view_rating[is.na(data$street_view_rating) == TRUE] = 2

# parcel numbers with no dashes or dots
data$parcel_number = str_replace_all(data$parcel_number,"[[:punct:]]","")

# write data
write.table(data,file='data/2016_streetview_input.csv',row.names=F,col.names=T,sep=",")