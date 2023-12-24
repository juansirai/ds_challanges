## 🖼️ Background

Once upon a time, in the early days of industrialization, there was little to no regulation on the use and handling of materials in the supply chain. As a result, hazardous materials were often used without proper safety precautions, putting workers and consumers at risk.
In the mid-20th century, public awareness of the dangers of hazardous materials began to grow, leading to the creation of regulations and safety standards. However, enforcing these standards was difficult, as there was no reliable way to quickly and accurately detect whether a material was hazardous or not.

This led to the development of various testing methods and technologies, such as chemical analysis and toxicity testing. However, these methods were often time-consuming and expensive, making them impractical for widespread use in the supply chain.

As technology advanced, new possibilities emerged. In the 21st century, machine learning and artificial intelligence algorithms became increasingly powerful and sophisticated. This opened up the possibility of using these algorithms to develop a faster, more efficient method of detecting hazardous materials in the supply chain.

The user, recognizing the potential of this technology, began working on developing an algorithm that could detect hazardous materials in real-time. They worked with experts in the field of chemistry and toxicology to gather data and train their algorithm to accurately identify hazardous materials.

After months of hard work, the user's algorithm was finally ready for testing. They integrated it into a supply chain monitoring system, which could automatically analyze incoming materials and alert workers if any hazardous materials were detected.

Thanks to the user's innovation and hard work, the supply chain became much safer, as workers and consumers could trust that the materials they were handling and using were not hazardous. This led to a significant reduction in workplace injuries and illnesses, and helped protect the health and safety of communities around the world.

## 📁 Dataset

For this challenge, you will have 2 CSV and a XSLX file:

* state_code_to_name.csv
* supply_chain_2012.csv (Train and Test)
* cfs-2012-pum-file-users-guide-app-a-jun2015.xlsx

You will have to create a prediction model based on the HAZMAT variable found in the supply_chain_2012.csv

The dataset can be found in the XSLX file explained.

There is the following file to be downloaded:

data.zip: It is a table that relates the data with the fiiles described above

## 📌 Task
The tasks are going to be based on answering and coding the following questions:

Create a prediction Model that can detect when a material in the Supply Chain is Hazardous and its type.

## ✅ Submission

File 1: predictions.json

Predictions must be in a JSON file named as predictions.json, an example can be found in the following link

In this predictions file, in json format, each row will correspond to the predicted value of the idx_test , i.e. if the first value is a 2 it means that this 2 corresponds to the first file of the test dataset. It is IMPORTANT to call the column target as specified in the format. Remember that you can use the to_json function of pandas to convert your dataframe to json, length of predictions has to be the same as in test.csv.

The objectives score will come from applying the f1-score of the predictions you have made to the testing dataset with our ground truth.

IMPORTANT: Your predictions must be in int format (0, 1 or 2).

Encoding of Hazardous:

* H -> 0
* N -> 1
* P -> 2

## 💡 Evaluation
The evaluation will be taken into consideration the following:

1200/1200:(OBJECTIVES) This will be obtained from the f1-score(macro) of the predictive model. Comparing the predictions your model has made about versus ground truth.