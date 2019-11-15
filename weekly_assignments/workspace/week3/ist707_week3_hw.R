# Homework Week 1: 

#--- Data Details -------------------------------------------------------------------
# 
# 

#--- Preprocess Steps: ----------------------------------------------------------------------

### Clear objects from Memory
rm(list=ls())
### Clear Console:
cat("\014")
### Set Working Directory
setwd("C:\\Users\\Ryan\\OneDrive\\workspaces\\ms_datascience_su\\IST707-DataAnalytics\\homework\\w1")

#---- Global Variable Assignments --------------------------------------------


#---- Load Required Packages -------------------------------------------------
if(!require("devtools")) {install.packages("devtools")}
if(!require("ggplot2")) {install.packages("ggplot2")}
#if(!require("tidyverse")) {install.packages("tidyverse")}

# ---- Step 1: Load the data -----------------------------------------------------------------------------