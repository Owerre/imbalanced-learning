# Canada Motor Vehicle Collisions (1999 - 2017)

![fig0](canada-collision/image/photo.jpg)

In many real-world data sets, class imbalance is a common problem. An imbalanced data set occurs when one class (majority or negative class) vastly outnumbered the other (minority or positive class). The class imbalance problem is manifested when the positive class is the class of interest. We have obtained a real-world dataset of motor vehicle collisions on public roads in Canada, with an inherent imbalanced class problem.

##  Dataset Information:   [National Collision Database](https://open.canada.ca/data/en/dataset/1eb9eba7-71d1-4b30-9fb1-30cbdab7e63a)


## Exploratory Data Analysis

### 1. Fatality Rate By Hour
The figure below shows the total fatal collisions is high at 5 p.m. on Fridays. The average age involved is around 36 years old and the average number of vehicles involved is around 2 vehicles.

![fig1](canada-collision/image/fig_h.png)

### 2. Total fatality  By Vehicle Model Year
This figure below shows that the 2000 & 2001 vehicle models caused the most fatal collisions and they were driven mostly by males.
![fig5](canada-collision/image/fig5.png)


### 3. Fatality Rate By Age

This figure below shows that young male drivers caused the most fatal collision.
![fig3](canada-collision/image/fig3.png)

### 4. Fatality Rate By Month

The figure below shows that fatal collisions are most likely in the month of July and August, which is the summer season in Canada. They also occur mostly on weekends.

![fig1](canada-collision/image/fig9.png)

## Dimensionality Reduction

The PCA plot of the data is shown below
![fig4](canada-collision/image/pca.png)

##  Model Results

The result shown below is based on Random Forest Classifier

![fig5](canada-collision/image/sup.png)

## WebApp

This project is accompanied by a web app [CollisionPredictor](https://collisionapp.herokuapp.com/)
