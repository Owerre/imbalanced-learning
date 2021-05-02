# Canada Motor Vehicle Collisions (1999 - 2017)

![fig0](canada-collision/image/photo.jpg)

In many real-world data sets, class imbalance is a common problem. An imbalanced data set occurs when one class (majority or negative class) vastly outnumbered the other (minority or positive class). The class imbalance problem is manifested when the positive class is the class of interest. We have obtained a real-world dataset of motor vehicle collisions on public roads in Canada, with an inherent imbalanced class problem.

## Dataset Information:   [National Collision Database](https://open.canada.ca/data/en/dataset/1eb9eba7-71d1-4b30-9fb1-30cbdab7e63a)

## Exploratory Data Analysis

### Fatality rate by month

- Fatal collisions are most likely in the months of July, June, and August respectively, which are the summer season in Canada.
- Fatal collisions occur mostly on weekends (Sundays & Saturdays respectively), and are predominant in males.

![fig1a](canada-collision/image/fig9a.png)

![fig1a](canada-collision/image/fig9b.png)

### Fatality rate by collision year

- Fatal collisions were mostly in the year 1999 & 2006 and the weather condition was visibility limited.
- Fatality rate also peaked in July 2003

  ![fig1a](canada-collision/image/fig1a.png)

  ![fig1b](canada-collision/image/fig1b.png)

### Fatality rate by collision hour

- Total fatal collisions is high at 5 p.m. on Fridays.
- The average age involved is around 36 years old.
- The average number of vehicles involved is around 2 vehicles.

![fig1](canada-collision/image/fig_h.png)

### Total fatality  by vehicle model year

- The light duty 2000s (i.e. 2000-2009) model vehicles are involved in the most fatal collisions and they were driven by mostly males.![fig5](canada-collision/image/fig5.png)![fig5](canada-collision/image/fig7.png)

### Fatality rate by age group

- Young people in their 20s (mostly males) are involved in most fatal collisions
- Fatality rate increases in older people![fig3a](canada-collision/image/fig3a.png)![fig3b](canada-collision/image/fig3b.png)
- ## Dimensionality Reduction

The PCA plot of the data is shown below
![fig4](canada-collision/image/pca.png)

## Model results on small data set

The result shown below is based on a small sample of the dataset due to lack of computational resources to train on the entire dataset.

![fig5](canada-collision/image/sup.png)

## WebApp

This project is accompanied by a web app [CollisionPredictor](https://collisionapp.herokuapp.com/)
