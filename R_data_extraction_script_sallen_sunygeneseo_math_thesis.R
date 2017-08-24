#Title: Data Extraction Script for Nepal Data
#Author: Stephanie Allen
#Project: SUNY Geneseo Undergrad Thesis
#Data Modified: 4/20/2017
#Resources:
#-Dplyr documentation: https://cran.rstudio.com/web/packages/dplyr/vignettes/introduction.html
#-Stringr Documentation: https://cran.r-project.org/web/packages/stringr/vignettes/stringr.html
#-List Reference: http://adv-r.had.co.nz/Data-structures.html#vectors, http://www.r-tutor.com/r-introduction/list
#Description: This script extracts all of the lat/long values, finds the locations the vehicles traveled
#to each day, and writes csv files for each day that contain all the locations visited by HDRVG (on that day).
#The script also finds the total amount of supplies distributed each day (had to assign a kg value
#to each categories of supplies).

library(dplyr)
library(stringr)

###Importing Data and Extracting Lat/Long Values
Full_Data <- read.csv("C:/Users/StephanieAllen/Dropbox/Allen Stephanie/Stephanie_Allen_Capstone/Data/Sorted_CHECK_Nepal_Distribution_Data_Set_1.csv", na.strings = "NA")

Full_Data$Date <- as.character(Full_Data$Date)
#Full_Data$FSC...KGs..sack.25..pack.1..box.4.5. <- if(Full_Data$FSC...KGs..sack.25..pack.1..box.4.5. == "Undefined"){Full_Data$FSC...KGs..sack.25..pack.1..box.4.5. <- 1}
Lats_Longs <- Full_Data[,14:15]
min_lat <- min(Lats_Longs[,1])
max_lat <- max(Lats_Longs[,1])
min_long <- min(Lats_Longs[,2])
max_long <- max(Lats_Longs[,2])

####Finding Averages for Each Type of Item Delivered#####
test <- group_by(Full_Data, Unit) #when apply functions, they will be based upon the group
means <- summarise(test, averages = mean(FSC...KGs..sack.25..pack.1..box.4.5.), minimum = min(FSC...KGs..sack.25..pack.1..box.4.5.), maximum = max(FSC...KGs..sack.25..pack.1..box.4.5.))

###Dates HDRVG ran missions
dates <- list('^1-May-15', '^10-May-15', '^11-May-15', '^12-May-15', '^13-May-15', '^14-May-15', '^15-May-15', '^16-May-15', '^17-May-15', '^18-May-15', '^19-May-15', '^2-May-15', '^20-May-15', '^21-May-15', '^22-May-15', '^23-May-15', '^24-May-15', '^29-Apr-15',  '^3-May-15', '^30-Apr-15',  '^4-May-15',  '^5-May-15', '^6-May-15',  '^7-May-15',  '^8-May-15',  '^9-May-15') 

###Preallocating data space
total_kilos <- vector('list',26)
distinct_missions <- vector('list',26)
tracker <- vector('list',26)

########Function for type_to_kilo: using the 'type' column to help inform the kilo amount##########
type_to_kilo <- function(name)
{
  switch(as.character(name),
         "box" = 4.5, #good
         "bottle" = 1, #from the internet: 32oz = 0.907kg
         "pack" = 1, #good
         "rolls" = 1, 
         "sack" = 25, #good
         "tabs" = 1,
         "tubes" = 1,
         "unit" = 1,
         1  )
}

####MAKE SURE TO FIX FACT THAT CALLING OTHER STUFF BESIDES 1-MAY (like 11-May)

for (i in 1:26){
  rm(place_holder) #clearing variables
  rm(total_kilos_PH)
  rm(distinct_missions_PH)
  place_holder <- filter(Full_Data, str_detect(Date, dates[[i]])) #get data for each date
  
  tracker[[i]] <- nrow(place_holder)
  
  location_data <- place_holder[, c(10,14:15)]
  relief_resources <- place_holder[,c(6:7)]
  relief_resources[,3] <- 0
  #RUN THE FUNCTION BELOW
  for (j in 1:length(relief_resources[,1]))
  {
    relief_resources[j,3] <- type_to_kilo(relief_resources[j,1])
  } 
  total_kilos_PH <- relief_resources[,2] %*% relief_resources[,3]
  #amount_per_site <- total_kilos_PH / 16
  distinct_missions_PH <- distinct(location_data,Mission) #after the data name, the labels indicate the columns through which to establish uniqueness
  #if did two, you could have a repeat of something with respect to one column, but you wouldn't have a repeated pair
  distinct_missions_PH <- distinct_missions_PH[,2:3]
  distinct_missions_PH[(nrow(distinct_missions_PH)+1), 1:2] <- rbind(27.6825, 85.3059) #adding the Yellow House (which is the depot) coordinates (which I took from Google)
  
  #Put stuff in Lists
  total_kilos[[i]] <- total_kilos_PH
  distinct_missions[[i]] <- distinct_missions_PH
  
  } #end of final for loop


#####Writing all of the Data to CSV Files#######
# for (h in 1:26){
#   write.table(distinct_missions[[h]], paste("C:/Users/StephanieAllen/Dropbox/Allen Stephanie/Stephanie_Allen_Capstone/MATLAB_Code/Lat_Long", h,".csv",sep="_"),row.names=FALSE,col.names=FALSE)
# }
# 
# 

####Write total supplies for each day to a file#### 
total_kilos_2 <- as.matrix(total_kilos)

write.table(total_kilos_2, paste("C:/Users/StephanieAllen/Dropbox/Allen Stephanie/Stephanie_Allen_Capstone/MATLAB_Code/Total_supplies_each_Day.csv"),row.names=FALSE,col.names=FALSE)
