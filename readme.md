# Text Classification

Text Classification is the task of assigning a label or class to a given text. Some use cases are sentiment analysis, natural language inference, and assessing grammatical correctness.

![Alt text](assents/1_T8WWibd7u8b7gfgeG0LgAA.gif)



Stanford University has obtained characteristics of billions of words that have strange proportions. Click on the [link](https://nlp.stanford.edu/projects/glove/) to read more and download.

---

Using NLP, I want to classify the sentences and display a special emoji based on the emotion that the sentence evokes.

- Dataset

![!\[Alt text\](image.png)](assents/spaces_u_image.png)

- Network

![Alt text](assents/spaces.png)

## How to install
```
pip install -r requirements.txt
```

## How to run
```
python Emoji_Text_Classification.py
```

## Accuracy and Loss

|Vectors|Accuracy|Loss|Accuracy plot|Loss plot|
|-------|--------|----|----|------|
|  50d  | 0.81   |0.6 |![Alt text](assents/acc_50d.png)|![Alt text](assents/loss_50d.png)|
| 100d  | 0.90   |0.4 |![Alt text](assents/acc_100d.png)|![Alt text](assents/loss_100d.png)|
| 200d  | 0.94   |0.2 |![Alt text](assents/acc_200d.png)|![Alt text](assents/loss_200d.png)|
| 300d  | 0.99   |0.1 |![Alt text](assents/acc_300d.png)|![Alt text](assents/loss_300d.png) |

## Results
![Alt text](assents/result1.PNG)
![Alt text](assents/result2.PNG)
![Alt text](assents/result3.PNG)
![Alt text](assents/result4.PNG)