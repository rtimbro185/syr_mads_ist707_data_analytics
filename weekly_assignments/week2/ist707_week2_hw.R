# Title: Homework Week 2 - Job Interview Question
# Author: Ryan Timbrook
# Due Date: 7/16/2019

#--- Instructions -------------------------------------------------------------------
# Real job interview question from a data analysis company
# Explore your story using data exploration and transformation techniques appropriately.
# Small data set to tell story with:
#   Each of 5 schools (A, B, C, D and E) is implementing the same math course for this semester,
#   with 35 lessions. There are 30 sections total. The semester is about 3/4 of the way through.
#   For each section, we record the number of students who are:
#     - very ahead (more than 5 lessons ahead)
#     - middling (5 lessons ahead to 0 lessons ahead)
#     - behind (1 to 5 lessons behind)
#     - more behind (6 to 10 lessons behind)
#     - very behind (more than 10 lessons behind)
#     - completed (finished with the course)
#
#   What's the story (or stories) in this data?
#

# Investigate Student Progress by Section, by School.
# Possible questions the data may be able to answer or give insights to:
# - Based on the time left in the term, are there students who are at risk of not completing a specific section?
# - Can we apply ranking levels to these schools based on the students progress factors? Is the student's progress level  high or low, issolated to a specific school?
# - Are there certain sections, where the schools student progress levels could show if a section appears to be easier or harder, regardless of the school?
# - 


#--- Data Details -------------------------------------------------------------------
# - Number of Schools: 5
# - Schools Names: A, B, C, D, E
# - Course: Math
# - Number of Unique Courses: 1
# - Lessions Count: 35
# - Sections Total: 30
# - Student Section Progress Categories:
# -   - VeryAhead   (+6 to +34)
# -   - Middling    (0 to +5)
# -   - Behind      (-1 to -5)
# -   - MoreBehind  (-6 to -10)
# -   - VeryBehind  (-10 to -35)
# -   - Completed   
# - Semester Schedule Progress Now: ~.75
####



#--- Preprocess Steps: -------------------------------------------------------------

### Clear objects from Memory
rm(list=ls())
### Clear Console:
cat("\014")
### Set Working Directory
setwd("C:\\Users\\rt310\\OneDrive\\workspaces\\ms_datascience_su\\IST707-DataAnalytics\\homework\\w2")

#---- Global Variable Assignments --------------------------------------------


#---- Load Required Packages ----
if(!require("tidyverse")) {install.packages("tidyverse")}
if(!require("ggplot2")) {install.packages("ggplot2")}
if(!require("psych")) {install.packages("psych")}
if(!require("reshape")) {install.packages("reshape")}
if(!require("gridExtra")) {install.packages("gridExtra")}


# ---- Step 1: Obtain the data ----
schoolsDf <- read.csv(file="data-storyteller.csv")
schoolsDf

# ---- Step 2: Data Exploration ----
str(schoolsDf)
colnames(schoolsDf) <- NULL
colnames(schoolsDf) <- c('School','Section','VeryAhead','Middling','Behind','MoreBehind','VeryBehind','Completed')
names(schoolsDf)
h_school <- head(schoolsDf)
write.table(h_school, file = "h_school.csv", append = FALSE, quote = TRUE, sep = ",",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")

s <- summary(schoolsDf)
s

d <- describe(schoolsDf)
d
write.table(d, file = "describe_school_df.csv", append = FALSE, quote = TRUE, sep = ",",
            eol = "\n", na = "NA", dec = ".", row.names = TRUE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")

# Students Progress Recorded per school
studentsBySchool <- aggregate(cbind(schoolsDf$VeryAhead,schoolsDf$Middling,schoolsDf$Behind,schoolsDf$MoreBehind,schoolsDf$VeryBehind,schoolsDf$Completed),
                              by=list(Schools=schoolsDf$School), FUN=sum)
colnames(studentsBySchool) <- NULL
colnames(studentsBySchool) <- c('School','VeryAhead','Middling','Behind','MoreBehind','VeryBehind','Completed')
studentsBySchool$Totals <- rowSums(studentsBySchool[,-1])
studentsBySchool <- studentsBySchool[order(-studentsBySchool$Totals),]
studentsBySchool
write.table(studentsBySchool, file = "students_progress_by_school_df.csv", append = FALSE, quote = TRUE, sep = ",",
            eol = "\n", na = "NA", dec = ".", row.names = TRUE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")


# column graph of student's by school
g_bar_students_by_school <- ggplot(data=studentsBySchool) +
  geom_col(aes(x=School,y=Totals)) +
  geom_line(group=1,aes(x=School,y=Totals), color='red') +
  theme(panel.background = element_blank()) +
  theme(plot.background = element_blank()) + 
  theme(panel.grid.major.y = element_line(color = 'grey')) +
  theme(panel.grid.minor.y = element_line(color = 'grey')) +
  ggtitle("Count of Student Lessons Reports by School") +
  xlab("School Name") +
  ylab("Total Student Records Covering all Lessons")
g_bar_students_by_school
ggsave(filename='g_bar_students_by_school.png',width=8 ,height=8)

# apply categories alerting by progess by School
# - Schools at risk of students not completing their lessons: 
# - Groups: GREEN, YELLOW, RED
# -     GREEN: Completed, Very Ahead, Middling
#-      YELLOW: Behind
#-      RED: More Behind, Very Behind

alert_categories <- c('School','GREEN','YELLOW','RED')
studentsBySchool <- studentsBySchool[order(studentsBySchool$School),]
s <- studentsBySchool
# School A
school <- 'A'
a_alerts_green <- as.numeric(s[s$School==school,7]+s[s$School==school,2]+s[s$School==school,3])
a_alerts_yellow <- as.numeric(s[s$School==school,4])
a_alerts_red <- as.numeric(s[s$School==school,5]+s[s$School==school,6])
# School B
school <- 'B'
b_alerts_green <- as.numeric(s[s$School==school,7]+s[s$School==school,2]+s[s$School==school,3])
b_alerts_yellow <- as.numeric(s[s$School==school,4])
b_alerts_red <- as.numeric(s[s$School==school,5]+s[s$School==school,6])
# School C
school <- 'C'
c_alerts_green <- as.numeric(s[s$School==school,7]+s[s$School==school,2]+s[s$School==school,3])
c_alerts_yellow <- as.numeric(s[s$School==school,4])
c_alerts_red <- as.numeric(s[s$School==school,5]+s[s$School==school,6])
# School D
school <- 'D'
d_alerts_green <- as.numeric(s[s$School==school,7]+s[s$School==school,2]+s[s$School==school,3])
d_alerts_yellow <- as.numeric(s[s$School==school,4])
d_alerts_red <- as.numeric(s[s$School==school,5]+s[s$School==school,6])
# School E
school <- 'E'
e_alerts_green <- as.numeric(s[s$School==school,7]+s[s$School==school,2]+s[s$School==school,3])
e_alerts_yellow <- as.numeric(s[s$School==school,4])
e_alerts_red <- as.numeric(s[s$School==school,5]+s[s$School==school,6])

# alert vectors
alert_green <- c(a_alerts_green,b_alerts_green,c_alerts_green,d_alerts_green,e_alerts_green)
alert_yellow <- c(a_alerts_yellow,b_alerts_yellow,c_alerts_yellow,d_alerts_yellow,e_alerts_yellow)
alert_red <- c(a_alerts_red,b_alerts_red,c_alerts_red,d_alerts_red,e_alerts_red)

# update student's df with alert categories
studentsBySchool$GREEN <- alert_green
studentsBySchool$YELLOW <- alert_yellow
studentsBySchool$RED <- alert_red
school_alerts <- studentsBySchool[c('School','GREEN','YELLOW','RED')]
melt_school_alerts <- melt(school_alerts, id='School')
melt_school_alerts

#stacked col graph by school
g_col_stacked_alerts <- ggplot(data=melt_school_alerts) +
  geom_col(aes(x=School,y=value,fill=variable)) +
  theme(panel.background = element_blank()) +
  theme(plot.background = element_blank()) + 
  theme(panel.grid.major.y = element_line(color = 'grey')) +
  theme(panel.grid.minor.y = element_line(color = 'grey')) +
  scale_fill_manual(values=c("Green","Yellow","Red"),
                    guide=guide_legend(title="Alert Level")) +
  ggtitle("Alert Counts Stacked by School") +
  xlab("School Name") +
  ylab("Stacked Alerts Counts")
g_col_stacked_alerts
ggsave(filename='g_col_stacked_alerts.png',width=8 ,height=8)
  

# 
a_prop <- round(studentsBySchool[studentsBySchool$School=='A',8]/sum(studentsBySchool$Totals),4)
b_prop <- round(studentsBySchool[studentsBySchool$School=='B',8]/sum(studentsBySchool$Totals),4)
c_prop <- round(studentsBySchool[studentsBySchool$School=='C',8]/sum(studentsBySchool$Totals),4)
d_prop <- round(studentsBySchool[studentsBySchool$School=='D',8]/sum(studentsBySchool$Totals),4)
e_prop <- round(studentsBySchool[studentsBySchool$School=='E',8]/sum(studentsBySchool$Totals),4)
school_props <- c(a_prop,b_prop,c_prop,d_prop,e_prop)
sum(school_props)
school_names <- c('A','B','C','D','E')
schoolPropsDf <- data.frame(school_names,school_props)
colnames(schoolPropsDf) <- NULL
colnames(schoolPropsDf) <- c('School','Total.PROP')
write.table(schoolPropsDf, file = "school_prop_df.csv", append = FALSE, quote = TRUE, sep = ",",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")


# Bivariate Plot
g_records_prop_school <- ggplot(data=schoolPropsDf) +
  geom_point(aes(x=School, y=Total.PROP), size=5, color="Black") +
  geom_line(aes(x=School, y=Total.PROP), group=1, color="Black") +
  theme(panel.background = element_blank()) +
  theme(plot.background = element_blank()) + 
  theme(panel.grid.major.y = element_line(color = 'grey')) +
  theme(panel.grid.minor.y = element_line(color = 'grey')) +
  theme(legend.position = "") +
  labs(title='Proportion of All Student Records by School',x='School',y='Student Records Proportion')
ggsave(filename='g_student_records_prop_school.png', width=8, height=8)
g_records_prop_school


# Student Progress Records as proportion of total student records by school
totals_titles <- c('Students','VeryAhead','Middling','Behind','MoreBehind','VeryBehind','Completed')
totalStudents <- sum(studentsBySchool$Totals)
totalVeryAhead <- sum(studentsBySchool$VeryAhead)
totalMiddling <- sum(studentsBySchool$Middling)
totalBehind <- sum(studentsBySchool$Behind)
totalMoreBehind <- sum(studentsBySchool$Middling)
totalVeryBehind <- sum(studentsBySchool$VeryBehind)
totalCompleted <- sum(studentsBySchool$Completed)
totals <- c(totalStudents,totalVeryAhead,totalMiddling,totalBehind,totalMoreBehind,totalVeryBehind,totalCompleted)
totalsDf <- data.frame(totals_titles,totals)
totalsDf
totalsDf2 <- totalsDf[totalsDf$totals_titles!='Students',]
orderedTotalsDf2 <- totalsDf2[order(-totalsDf2$totals),]
orderedTotalsDf2
write.table(orderedTotalsDf2, file = "progress_totals_df.csv", append = FALSE, quote = TRUE, sep = ",",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")

# col plot totals by progress category
g_progess_totals <- ggplot(data=orderedTotalsDf2) +
  geom_col(aes(x=totals_titles,y=totals)) +
  theme(panel.background = element_blank()) +
  theme(plot.background = element_blank()) + 
  theme(panel.grid.major.y = element_line(color = 'grey')) +
  theme(panel.grid.minor.y = element_line(color = 'grey')) +
  ggtitle("Progess Totals by Categores", subtitle = "All five schools, A, B, C, D, E") +
  xlab("Lesson Progress Categories") +
  ylab("All Schools, Lessons Progress Records Count")
g_progess_totals
ggsave(filename='g_progess_totals.png',width=8 ,height=8)

propStudentsBySchool <- data.frame(studentsBySchool %>% group_by(School) %>%
  summarise(VeryAhead.PROP=VeryAhead/Totals, Middling.PROP=round(Middling/Totals,2), Behind.PROP=round(Behind/Totals,2), 
            MoreBehind.PROP=round(MoreBehind/Totals,2), VeryBehind.PROP=round(VeryBehind/Totals,2), 
            Completed.PROP=round(Completed/Totals,2), GREEN.PROP=round(GREEN/Totals,2), YELLOW.PROP=round(YELLOW/Totals,2), RED.PROP=round(RED/Totals,2)))
propStudentsBySchool
propStudentsBySchool <- merge(propStudentsBySchool,schoolPropsDf, by='School')
propStudentsBySchool
write.table(propStudentsBySchool, file = "prop_stud_by_school_df.csv", append = FALSE, quote = TRUE, sep = ",",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")

s_prop_stud_by_schol <- summary(propStudentsBySchool)
write.table(s_prop_stud_by_schol, file = "s_prop_stud_by_schol.csv", append = FALSE, quote = TRUE, sep = ",",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")

describe(propStudentsBySchool$Total.PROP)
describe(propStudentsBySchool$GREEN.PROP)
describe(propStudentsBySchool$YELLOW.PROP)
describe(propStudentsBySchool$RED.PROP)

# visualize proportions of alerts by school
prop_school_alerts <- propStudentsBySchool[c('School','GREEN.PROP','YELLOW.PROP','RED.PROP')]
melt_prop_school_alerts <- melt(prop_school_alerts, id='School')
melt_prop_school_alerts

#stacked col graph by school
g_col_stacked_alerts <- ggplot(data=melt_prop_school_alerts) +
  geom_col(aes(x=School,y=value,fill=variable)) +
  theme(panel.background = element_blank()) +
  theme(plot.background = element_blank()) + 
  theme(panel.grid.major.y = element_line(color = 'grey')) +
  theme(panel.grid.minor.y = element_line(color = 'grey')) +
  scale_fill_manual(values=c("Green","Yellow","Red"),
                    guide=guide_legend(title="Alert Level")) +
  ggtitle("Alert Proportions Stacked by School") +
  xlab("School Name") +
  ylab("Stacked Alert Proportions")
g_col_stacked_alerts
ggsave(filename='g_col_stacked_alert_propor.png',width=8 ,height=8)


# Visualizations plots of Progress Levels by School
# Middling
# Boxplot
g_box_middling <- ggplot(data=propStudentsBySchool) +
  geom_boxplot(aes(x=factor(0),y=Middling.PROP), fill='Green',notch = FALSE) +
  ylab('Middling Proportion of Schools') + xlab('Middling') +
  ggtitle('Boxplot Middling Proportion of Schools')
g_box_middling

ggplot(data=propStudentsBySchool) +
  geom_col(aes(x=School,y=Middling.PROP))

g_combined_progress <- ggplot(data=propStudentsBySchool,aes(x=School,y=Total.PROP)) +
  # Baseline Total Population Proportion by School
  geom_point(size=4, alpha=.7) +
  geom_line(group=1,color='black') + 
  # Middling
  geom_point(aes(x=School,y=Middling.PROP),size=4,color='green', alpha=.5) +
  geom_line(group=1,color='green',aes(x=School,y=Middling.PROP)) + 
  # Behind.PROP
  geom_point(aes(x=School,y=Behind.PROP),size=4,color='blue', alpha=.4) + 
  geom_line(group=1,color='blue',aes(x=School,y=Behind.PROP)) +
  # MoreBehind.PROP
  geom_point(aes(x=School,y=MoreBehind.PROP),size=4,color='red', alpha=.4) + 
  geom_line(group=1,color='red',aes(x=School,y=MoreBehind.PROP)) +
  # VeryBehind.PROP
  geom_point(aes(x=School,y=VeryBehind.PROP),size=4,color='purple', alpha=.4) + 
  geom_line(group=1,color='purple',aes(x=School,y=VeryBehind.PROP)) +
  # Completed.PROP
  geom_point(aes(x=School,y=Completed.PROP),size=4,color='orange', alpha=.4) + 
  geom_line(group=1,color='orange',aes(x=School,y=Completed.PROP)) +
  theme(panel.background = element_blank()) +
  theme(plot.background = element_blank()) + 
  theme(panel.grid.major.y = element_line(color = 'grey')) +
  theme(panel.grid.minor.y = element_line(color = 'grey')) +
  annotate("text",label="Black: Proportion of Total Student Records per School\nBaseline reference",x='C',y=.57) +
  labs(title='Normalized Proportion of Students Lessons Progress Report',x='School',y='Preportion of Progress Category by School')
ggsave(filename='g_combined_progress.png', width=8, height=8)
g_combined_progress

# Bivariate Plot: x-axis: School, y-axis: Proportion of students
g_middlingProp_school <- ggplot(data=propStudentsBySchool) +
  geom_point(aes(x=School, y=Middling.PROP), size=4, color="Green") +
  geom_line(group=1, color='green', aes(x=School, y=Middling.PROP)) +
  geom_point(aes(x=School, y=Total.PROP), size=3, color="Black") +
  geom_line(group=1,color='black', aes(x=School, y=Total.PROP)) +
  annotate("text",label='Proportion of Total Student Records per School',x='C',y=.6) +
  geom_hline(yintercept=mean(propStudentsBySchool$Middling.PROP), color='red') +
  annotate("text",label="Avg Middling", x='A', y=mean(propStudentsBySchool$Middling.PROP)+.01) +
  labs(title='Proportion of \'MIDDLING\' Students by School',x='School',y='Green: Middling Proportion | Black: Total Proportion')

ggsave(filename='g_middlingProp_school.png', width=8, height=8)
g_middlingProp_school


# Behind
g_box_behind <- ggplot(data=propStudentsBySchool) +
  geom_boxplot(aes(x=factor(0),y=Behind.PROP), fill='yellow',notch = FALSE) +
  ylab('Behind Proportion of Schools') + xlab('Behind') +
  ggtitle('Boxplot Behind Proportion of Schools')
g_box_behind

ggplot(data=propStudentsBySchool) +
  geom_col(aes(x=School,y=Behind.PROP))

# Bivariate Plot:
g_behindProp_school <- ggplot(data=propStudentsBySchool) +
  geom_point(aes(x=School, y=Behind.PROP), size=3, color="orange") +
  geom_line(group=1, color='orange', aes(x=School, y=Behind.PROP)) +
  geom_point(aes(x=School, y=Total.PROP), size=3, color="Black") +
  geom_line(group=1,color='black', aes(x=School, y=Total.PROP)) +
  annotate("text",label='Black: Proportion of Total Student Records per School',x='C',y=.6) +
  geom_hline(yintercept=mean(propStudentsBySchool$Behind.PROP), color='red') +
  annotate("text",label="Avg Behind", x='A', y=mean(propStudentsBySchool$Behind.PROP)+.01) +
  #theme(legend.position = "") +
  labs(title='Proportion of BEHIND Students by School',x='School',y='Orange: Behind Proportion | Black: Total Proportion')
ggsave(filename='g_behindProp_school.png', width=8, height=8)
g_behindProp_school


# MoreBehind
# Boxplot
g_box_morebehind <- ggplot(data=propStudentsBySchool) +
  geom_boxplot(aes(x=factor(0),y=MoreBehind.PROP), fill='Red',notch = FALSE) +
  ylab('MoreBehind Proportion of Schools') + xlab('MoreBehind') +
  ggtitle('Boxplot MoreBehind Proportion of Schools')
g_box_morebehind

ggplot(data=propStudentsBySchool) +
  geom_col(aes(x=School,y=MoreBehind.PROP))

#Bivariate Plot
g_moreBehindProp_school <- ggplot(data=propStudentsBySchool) +
  geom_point(aes(x=School, y=MoreBehind.PROP), size=3, color="purple") +
  geom_line(group=1, color='purple', aes(x=School, y=MoreBehind.PROP)) +
  geom_point(aes(x=School, y=Total.PROP), size=3, color="Black") +
  geom_line(group=1,color='black', aes(x=School, y=Total.PROP)) +
  annotate("text",label='Black: Proportion of Total Student Records per School',x='C',y=.6) +
  geom_hline(yintercept=mean(propStudentsBySchool$MoreBehind.PROP), color='red') +
  annotate("text",label="Avg MoreBehind", x='A', y=mean(propStudentsBySchool$MoreBehind.PROP)-.01) +
  labs(title='Proportion of MoreBehind Students by School',x='School',y='Purple: MoreBehind Proportion | Black: Total Proportion')
ggsave(filename='g_moreBehindProp_school.png', width=8, height=8)
g_moreBehindProp_school

# VeryBehind
# Boxplot
g_box_verybehind <- ggplot(data=propStudentsBySchool) +
  geom_boxplot(aes(x=factor(0),y=VeryBehind.PROP), fill='Red',notch = FALSE) +
  ylab('VeryBehind Proportion of Schools') + xlab('VeryBehind') +
  ggtitle('Boxplot VeryBehind Proportion of Schools')
g_box_verybehind

ggplot(data=propStudentsBySchool) +
  geom_col(aes(x=School,y=VeryBehind.PROP))

# Bivariate Plot
g_veryBehindProp_school <- ggplot(data=propStudentsBySchool) +
  geom_point(aes(x=School, y=VeryBehind.PROP), size=3, color="purple") +
  geom_line(group=1, color='purple', aes(x=School, y=VeryBehind.PROP)) +
  geom_point(aes(x=School, y=Total.PROP), size=3, color="Black") +
  geom_line(group=1,color='black', aes(x=School, y=Total.PROP)) +
  annotate("text",label='Black: Proportion of Total Student Records per School',x='C',y=.6) +
  geom_hline(yintercept=mean(propStudentsBySchool$VeryBehind.PROP), color='red') +
  annotate("text",label="Avg VeryBehind", x='A', y=mean(propStudentsBySchool$VeryBehind.PROP)-.01) +
  labs(title='Proportion of \'VERY BEHIND\' Students by School',x='School',y='Red: VeryBehind Proportion | Black: Total Proportion')
ggsave(filename='g_veryBehindProp_school.png', width=8, height=8)
g_veryBehindProp_school

# Completed
# Boxplot
g_box_completed <- ggplot(data=propStudentsBySchool) +
  geom_boxplot(aes(x=factor(0),y=Completed.PROP), fill='Green',notch = FALSE) +
  ylab('Completed Proportion of Schools') + xlab('Completed') +
  ggtitle('Boxplot Completed Proportion of Schools')
g_box_completed

ggplot(data=propStudentsBySchool) +
  geom_col(aes(x=School,y=Completed.PROP))

# Bivariate Plot
g_completedProp_school <- ggplot(data=propStudentsBySchool) +
  geom_point(aes(x=School, y=Completed.PROP), size=3, color="green") +
  geom_line(group=1, color='green', aes(x=School, y=Completed.PROP)) +
  geom_point(aes(x=School, y=Total.PROP), size=3, color="Black") +
  geom_line(group=1,color='black', aes(x=School, y=Total.PROP)) +
  annotate("text",label='Black: Proportion of Total Student Records per School',x='C',y=.6) +
  geom_hline(yintercept=mean(propStudentsBySchool$Completed.PROP), color='red') +
  annotate("text",label="Avg Completed", x='A', y=mean(propStudentsBySchool$Completed.PROP)-.01) +
  labs(title='Proportion of \'COMPLETED\' Students by School',x='School',y='Green: Completed Proportion | Black: Total Proportion')
ggsave(filename='g_completedProp_school.png', width=8, height=8)
g_completedProp_school

# Show all results (charts) in one window, using the grid.arrange function
ga_bi <- grid.arrange(g_middlingProp_school,g_behindProp_school,g_moreBehindProp_school,g_veryBehindProp_school,g_completedProp_school, ncol=3, nrow=2)
ga_bi
ggsave(file="Grid_Arrange_School_ProgressLevel.jpg", ga_bi, width = 17, height = 10)

ga_box <- grid.arrange(g_box_completed,g_box_middling,g_box_behind,g_box_morebehind,g_box_verybehind,ncol=3,nrow=2)
ga_box
ggsave(file="ga_boxplots_progress.jpg",ga_box,width=17, height=10)


meltPropStudentsBySchool <- melt(propStudentsBySchool,id='School')
meltPropStudentsBySchool
ggplot(data=meltPropStudentsBySchool, aes(x=meltPropStudentsBySchool[,meltPropStudentsBySchool$variable=='Middling.PROP'],y=))

# completedBySchool
completedAvgBySchool <- aggregate(list(CompletedAvg=schoolsDf$Completed), by=list(School=schoolsDf$School), FUN=mean)
completedAvgBySchool

# very ahead average by school
veryAheadAvgBySchool <- aggregate(list(VeryAheadAvg=schoolsDf$VeryAhead), by=list(School=schoolsDf$School), FUN=mean)
veryAheadAvgBySchool


# Exploration Visualizations
boxplot(schoolsDf$Behind)
boxplot(schoolsDf$Middling)


# graph bar chart - x axis: progress level, y axis: value
schoolProgress <- melt(schoolsDf[,-2],id='School')

  

# - Based on the time left in the term, are there students who are at risk of not completing a specific section?



# ---- Step 3: Data Transformation ----


# ---- Step 4: Data Analysis Conclusions ----




