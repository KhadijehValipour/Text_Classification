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
python Emoji_Text_Classification.py --dimension 100 --sentence "I am very well today"
```

## Accuracy and Loss

|Vectors|Accuracy|Loss|Accuracy with dropout|Loss with dropout|Inference average time|
|-------|--------|----|----|------|------|
|  50d  | 0.81   |0.6 |0.80|0.72|0.11s|
| 100d  | 0.90   |0.4 |0.82|0.61|0.13s|
| 200d  | 0.94   |0.2 |0.82|0.48|0.12s|
| 300d  | 0.99   |0.1 |0.96|0.41|0.12s|

## Results
![Alt text](assents/result1.PNG)
![Alt text](assents/result2.PNG)
![Alt text](assents/result3.PNG)
![Alt text](assents/result4.PNG)