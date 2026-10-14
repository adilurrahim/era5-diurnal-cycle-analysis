#install.packages("matrixStats")

library(matrixStats)
library(raster)

setwd('D:/ERA5/Data')
file = list.files('.')

#read the nc file
nc_file = brick(file)
#par(mar=c(1,1,1,1))
#plot(nc_file[[700]])

for (k in seq(1,nc_file@data@nlayers,24)){
  nc_slice = nc_file[[k]]
  sample_df = (as.data.frame(nc_slice,na.rm=T, xy=T))
  
  for(i in 1:23){ 
    nc_slice = nc_file[[k+i]]
    df = (as.data.frame(nc_slice,na.rm=T, xy=F))
    sample_df = cbind(sample_df,df)
  }

  #average of 24 hourly values
  averages = round(data.frame(rowMeans(as.matrix(sample_df[,3:26]), na.rm=TRUE)),2)
  average_df = cbind(sample_df[,1:2],averages)
  colnames(average_df) = c('x','y','average')
  
  #portion of hour that is greater, equal, or less than the average
  above_avg = data.frame(rowSums(sample_df[,3:26] > average_df[,3]))
  colnames(above_avg) = c('Above')
  equal_avg = data.frame(rowSums(sample_df[,3:26] == average_df[,3]))
  colnames(equal_avg) = c('Equal')
  below_avg = data.frame(rowSums(sample_df[,3:26] < average_df[,3]))
  colnames(below_avg) = c('Below')
  
  average_final_df= cbind(average_df,above_avg,equal_avg,below_avg)
  
  setwd('D:/ERA5/Processed_Data')
  filename = paste(substr(nc_slice@z[[1]],1,10),'_average_data.csv',sep="")
  write.csv(average_final_df,filename,row.names = FALSE)
  
  #mean of max and min values
  mins = rowMins(as.matrix(sample_df[,3:26]), na.rm = TRUE)
  maxs= rowMaxs(as.matrix(sample_df[,3:26]), na.rm = TRUE)
  
  mean = round(data.frame(rowMeans(cbind(data.frame(mins),data.frame(maxs)), na.rm=TRUE)),2)
  mean_df = cbind(sample_df[,1:2],mean)
  colnames(mean_df) = c('x','y','mean')
  
  #portion of hour that is greater, equal, or less than the mean
  above_mean = data.frame(rowSums(sample_df[,3:26] > mean_df[,3]))
  colnames(above_mean) = c('Above')
  equal_mean = data.frame(rowSums(sample_df[,3:26] == mean_df[,3]))
  colnames(equal_mean) = c('Equal')
  below_mean = data.frame(rowSums(sample_df[,3:26] < mean_df[,3]))
  colnames(below_mean) = c('Below')
  
  mean_final_df= cbind(mean_df,above_mean,equal_mean,below_mean)
  
  filename = paste(substr(nc_slice@z[[1]],1,10),'_mean_data.csv',sep="")
  write.csv(mean_final_df,filename,row.names = FALSE)
  
  gc()

}



