import pandas as pd
import numpy as np

students_performance = pd.read_csv('http://stepik.org/media/attachments/course/4852/StudentsPerformance.csv')


students_performance


students_performance.head()

students_performance.head(10)


students_performance.tail()

students_performance.tail(10)

students_performance.describe()


students_performance.dtypes


students_performance.loc[:7] #-
students_performance.head(7) #+
students_performance.tail(7) #-
students_performance.iloc[0:7] #+
students_performance.loc[:6] #+
students_performance.iloc[:7] #+



#

students_performance.columns # descibes columns
students_performance.get_dtype_counts
students_performance.shape
students_performance.index

students_performance.size


#
# titanic dataset
#



tit = pd.read_csv('https://stepik.org/media/attachments/course/4852/titanic.csv')
tit.columns
tit.shape
tit.dtypes


#
# students_performance[students_performance['writing score'] > 100 and students_performance.gender == 'female']
#


