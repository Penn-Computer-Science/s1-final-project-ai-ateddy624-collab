**Data**
---

**Data Collection Method**

I used the website Teachable Machine to take the pictures, exported them from the site and imported it into VS Code

---

**Potential Sources of Bias**

Some potential sources of bias can include the background of where I took the pictures of my brother and I, as well as the clothes we were wearing and the fact that we have the same haircut

---
**How the dataset might impact model perfomance**

Because the dataset is relatively small, with a lot of similarity between photos, and a consistant background, the model may not perform quite as well/accurately

---

**Model Training**
-
***NOTE:*** When I tried 1 epoch, nothing showed up on the graph, so this is all I have to show for it

---

**Reflection and Analysis**
-

My three epoch run went a little better, as it had approximately equal accuracy and less loss.
I think that the hyperparameter that mattered the most was my Conv2d's and my max pooling, as the drop outs didn't really prevent overfitting, and even if it did, it doesn't matter too much if validation accuracy and loss are still improving.
I was surprised by how long each epoch took to train.
My dataset quality affected the length of time that training takes, as well as the accuracy (how often it got the right answer) and my loss (how sure/unsure it was about each answer). I was somewhat surprised by how well my model performed. If I had more time, I would have gotten better data, with more classes, greater differences between each image, and different backgrounds to keep the algorithm from memorizing backgrounds.