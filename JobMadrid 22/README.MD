# Job Madrid Hackaton
⚠ Link to explanation here -> https://www.loom.com/share/acd1cd7db59244bfa16faf5d8ffc8c57

<img src="https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/81f1f131049961.563f0d5d52dde.jpg">

## Index
* [Loom Explanation](https://www.loom.com/share/acd1cd7db59244bfa16faf5d8ffc8c57)
* [Jupyter Notebook](https://github.com/juansirai/ds_challanges/blob/master/JobMadrid%2022/Main.ipynb)
* [Additional Functions](https://github.com/juansirai/ds_challanges/blob/master/JobMadrid%2022/preprocessing.ipynb)
* [Dashboard](https://github.com/juansirai/ds_challanges/blob/master/JobMadrid%2022/Dashboard.pbix)

------------------------------
## Business Analytics
-------------------------------------

## 🛫 Background

You works at an airline called Nuwe Air and is part of the customer success team. As you know, Nuwe Air always seeks to achieve maximum passenger satisfaction and the mission of your department is to detect defects and areas for improvement and make decisions to increase overall satisfaction.

Your head of department launched an initiative a few months ago to find out the opinion of our passengers after flying with Nuwe Air. Now it's your turn to enter the scene and find out what customers think of the service we offer them and where we have to focus to improve our results.

What is asked of you in this case is to identify which are the services that require the most urgency, which types of clients are the best and which types of clients are the worst, which are the fields that most affect satisfaction total of the client (it can be the delay in leaving, in arriving, the food provided during the flights...), etc., and build a dashboard that helps your head of department to understand this data. In addition, since he is very confident in your skills and knowledge of the sector, he asks you to also give him some recommendation on what areas or what type of client we have to focus on in order to improve the average satisfaction of our clients.


## 📚 Dataset

* Gender: Gender of passengers (female, male)
* Customer Type: The type of customer (Loyal Customer, Disloyal Customer)
* Age: The actual age of the passengers
* Type of trip: Purpose of the passenger's flight (Personal trip, Business trip)
* Class: Class of travel on the passenger plane (Business, Eco, Eco Plus)
* Flight Distance: The flight distance of this trip
* Wi-Fi service on board: Level of satisfaction with the Wi-Fi service on board (0: Not applicable; 1-5)
* Convenient departure/arrival time: Level of satisfaction regarding the convenience of the departure/arrival time
* Ease of online booking: Satisfaction level of online booking
* Door location: Satisfaction level of the door location
* Food and drink: Level of satisfaction with food and drink
* Online boarding: Level of satisfaction with online boarding
* Seat comfort: Satisfaction level of seat comfort
* On-board entertainment: Satisfaction level of in-flight entertainment
* On-board service: Level of satisfaction with the on-board service
* Waiting room service: Service satisfaction level in the waiting room
* Baggage handling: Satisfaction level of the baggage service
* Billing service: Satisfaction level of the billing service
* On-board service: Level of satisfaction with the on-board service
* Cleanliness: Level of satisfaction with cleanliness
* Delay in departure in minutes: Minutes of delay in departure
* Delay in arrival in minutes: Minutes of delay in arrival
* Satisfaction: Level of satisfaction of the airline (Satisfaction, neutral or dissatisfaction)


## ⚡ Challenge 

* What areas do we need to improve to achieve greater customer satisfaction?
* What type of customers do we have to focus on to improve their satisfaction, especially

Information that the dashboard/presentation may contain:
*  What kind of clients do we have the most?
*  What kind of customers are the most dissatisfied?
*  What is the area where we have the lowest satisfaction?
*  What is the average satisfaction of the areas and in which are we most below?
*  What areas most affect the customer's bottom line being "not satisfied"?
*  Is there any noticeable difference in how some types of clients perceive some services vs how others perceive it?

-----------------------------------

## Libraries
In order to run the notebook, you should run the following line

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
---------------------------------
## Conclussions
When analyzing the general performance of the airline, we can see that in general terms it has a 
considerable proportion of loyal customers, and is efficient in its flight schedules (+76% with a delay of less than 30 minutes).

So... why we have more than 55% of customers who are neutral or dissatisfied with our service? 

By analyzing our consumers demographically, as well as their survey responses, we see that the greatest dissatisfaction is in economy class, and personal travel.

If we analyze the correlation between features, we suggest to pay attention in:
* Inflight WIFI service: since is the one with less score in surveys, and is also correlated in +0.28 with a positive score.
* Food and Drink: since is also lower than the total average, and has a correlation of 0.22 with satisfaction.
* Online Boarding: this feature is at the median value, but the correlation with the satisfaction is the highest of the subset. We should try to improve 
* it in order to achieve a better score.

Eventhough today have a good performance, due to its high correlation we cannot forget:
* Inflight Entertainment
* Seat Comfort

So...Our business seems to do very well in the business travel category, but considering that not all of our passengers can afford business class,
we should consider improving our service in economy class, so as not to lose our loyal customers.

We should also consider to Increase investment in digital modernization of facilities, since they were the worst performers in the surveys, 
and they can help reverse the experience in leisure trips and young people.
