library(caret)
library(kernlab)
library(foreign)
library(ggplot2)
library(lattice)
library(pscl)

#please change the two paths according to your project path
#datasetpath is the promise dataset path, and resultpath is the path of the predicted number of defects via the following 5 regression algorithms
datasetpath="/Users/xiao/Documents/project/Python/Revisiting-IST/CrossversionData1/"
resultfold="/Users/xiao/Documents/project/Python/Revisiting-IST/Rresult/"

folder_names<- list.files(datasetpath)
dir<-paste(finalpath,sep = "",folder_names)

for (i in 1:length(dir)){
  files<-list.files(dir[i])
  if(files[1]<files[2]){
    trainingdataname=files[1]
    testingdataname=files[2]
  }else{
    trainingdataname=files[2]
    testingdataname=files[1]
  }
  # print("trainingdataname:")
  cat("trainingdataname:",trainingdataname,'\n')
  # print("testingdataname:")
  cat("testingdataname:",testingdataname,'\n')
  
  NBRresultname=paste("NBR",sep = "",testingdataname)
  PRresultname=paste("PR",sep = "",testingdataname)
  ZINBRresultname=paste("ZINBR",sep = "",testingdataname)
  ZIPRresultname=paste("ZIPR",sep = "",testingdataname)
  HRresultname=paste("HR",sep = "",testingdataname)
  
  a = data.frame(dataFrame <- read.csv(paste(dir[i],sep = "/",trainingdataname)))
  #The test data variable 'b'
  b = data.frame(dataFrameTest <- read.csv(paste(dir[i],sep = "/",testingdataname)))
  library(MASS)
  set.seed(123)
  
  
  print("--------------Negative binomial regression analysis----------------")
  m1 <- glm.nb(bug~., data = a)
  predictm1 <- predict(m1, b, ncomp = 1)
  t = data.frame(predictm1)
  write.csv(t, file = paste(resultfold,sep = "",NBRresultname))
  print("--------------End of Negative binomial regression analysis----------------")
  
  
  print("--------------Possion regression analysis----------------")
  m2 <- glm(bug~., data = a, family="poisson")
  predictm2 <- predict(m2, b, type="response")
  q = data.frame(predictm2)
  write.csv(q, file = paste(resultfold,sep = "",PRresultname))
  print("--------------End of Possion regression analysis----------------")
  
  
  print("--------------Zero-Inflated Negative Binomial Regression----------------")
  #install.packages("pscl")
  require(ggplot2)
  require(pscl)
  
  require(MASS)
  require(boot)
  m3 <- zeroinfl(bug~.| 1, data = a, dist = "negbin", 
                 model = TRUE, link = "logit",
                 y = TRUE, x = TRUE)
  predictm3 <- predict(m3,b)
  o <- data.frame(predictm3)
  write.csv(o, file = paste(resultfold,sep = "",ZINBRresultname))
  print("--------------End of Zero-Inflated Negative Binomial Regression----------------")
  
  
  
  print("--------------Zero-Inflated Poisson Regression (ZIPR)----------------")
  m4 <- zeroinfl(bug~.| 1, data = a, dist = "poisson", 
                 model = TRUE, link = "logit",
                 y = TRUE, x = TRUE)
  predictm4 <- predict(m4,b)
  w <- data.frame(predictm4)
  write.csv(w, file = paste(resultfold,sep = "",ZIPRresultname))
  print("--------------End of Zero-Inflated Poisson Regression (ZIPR)----------------")
  
  
  
  print("-------------- Hurdle Regression (HR)----------------")
  m5 <- hurdle(bug~., data = a, dist = "negbin")
  predictm5 <- predict(m5, b)
  n <- data.frame(predictm5)
  write.csv(n, file = paste(resultfold,sep = "",HRresultname))
  print("-------------- End for Hurdle Regression (HR)----------------")
  
  
}
