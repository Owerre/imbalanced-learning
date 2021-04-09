# Canada Motor Vehicle Collisions (1999 - 2017)

![fig0](image/photo.jpg)

In many real-world data sets, class imbalance is a common problem. An imbalanced data set occurs when one class (majority or negative class) vastly outnumbered the other (minority or positive class). The class imbalance problem is manifested when the positive class is the class of interest. We have obtained a real-world dataset of motor vehicle collisions in Canada, with an inherent imbalanced class problem.

##  Dataset Information:   [National Collision Database](https://open.canada.ca/data/en/dataset/1eb9eba7-71d1-4b30-9fb1-30cbdab7e63a)


## Exploratory Data Analysis

### 1. Fatality Rate By Hour
The figure below shows the fatality rate is increased at 5 p.m. on Sundays and the average age involved is around 32 years old.

![fig1](image/fig_h.png)


### 2. Fatality Rate By Month
The figure below shows the fatality rate is increased in the month of July and August, which is obviously the summer season.

![fig1](image/fig9.png)

### 3. Fatality Rate By Vehicle Model Year
This figure below shows that the 2000 & 2001 vehicle models caused the most fatal collisions and they were driven mostly by males.
![fig5](image/fig5.png)


### 4. Fatality Rate By Age

This figure below shows that young male drivers caused the most fatal collision.
![fig3](image/fig3.png)

## Dimensionality Reduction

![fig4](image/pca.png)

##  Model Results

The result shown below is based on Random Forest Classifier

![fig5](image/sup.png)

## WebApp

This project is accompanied by a web app [CollisionPredictor](https://collisionapp.herokuapp.com/)
