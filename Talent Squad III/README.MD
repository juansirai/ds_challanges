# Background


## Influences on academic performance

A study has been carried out to see if the academic performance of children is influenced by the academic level of their parents. Therefore, the academic results of the students will be evaluated based on several variables

Data
* Dataset of 800 rows for the training (train) of the prediction algorithm and 200 for the test (test).

Variables
* 'gender' : Student's sex
* 'parental level of education': Educational level of the father
* 'lunch': If you have dining scholarships
* 'test preparation course': Attend academy
* 'math score': Math score
* 'reading score': Reading Comprehension
* 'writing score': Writing score


# Problem

Predict if the student's academic results are influenced by the educational level of the parents.<br>
These have a number that represents the following educational level:
*  high school: 0,
*  some high school: 1,
*  some college: 2,
*  associate's degree: 3,
*  bachelor's degree: 4,
*  master's degree: 5


# Results

The code is delivered in a py file named app.py. which contains all the functions used and the necessary code together with documentation to reproduce the solution.
The exploratory analysis is delivered in an ipynb.

# Analysis

We performed a One-Way ANOVA test, which is a test that is used to find out whether there exists a statistically significant difference between the mean values of more than one group.

Hypothesis involved: A one-way ANOVA has the below given null and alternative hypotheses:

H0 (null hypothesis): μ1 = μ2 = μ3 = … = μk (It implies that the means of all the population are equal
H1 (null hypothesis): It states that there will be at least one population mean that differs from the rest

The F statistic and p-value turn out to be equal to 5.171 and 0.0 respectively. Since the p-value is less than 0.05 hence we would reject the null hypothesis. This implies that we have sufficient proof to say that there exists a difference in the performance of students, depends of parental education level

# Solution

We tested several classification models, but without enough results. <br>
However, we decided to use MLPClassifier who had better results in the test set.


# License

Copyright (c) 2012-2022 Scott Chacon and others

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
