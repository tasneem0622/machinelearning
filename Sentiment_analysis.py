#importing files
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

#Taking tweet as an input
tweet=input("Enter a tweet: ")


#preprocessing
tweet_words=[]

for word in tweet.split(' '):
    if word.startswith("@") and len(word)>1:
        word='@user'
    elif word.startswith("http"):
        word='http'
    tweet_words.append(word)
tweet_proc=" ".join(tweet_words)
#print(tweet_proc)

roberta='cardiffnlp/twitter-roberta-base-sentiment'
model=AutoModelForSequenceClassification.from_pretrained(roberta)

tokenizer=AutoTokenizer.from_pretrained(roberta)

labels=labels = {'Negative': 1, 'Neutral': 2,"Positive": 3}

#scores of negative positive and neutral
encoded_tweet=tokenizer(tweet_proc,return_tensors='pt')
output=model(**encoded_tweet)
scores=output[0][0].detach().numpy()
scores=softmax(scores)
update_dict = {key: value for key, value in zip(labels.keys(), scores)}
labels.update(update_dict)
max_key = max(labels, key=labels.get)
max_value = labels[max_key]
print(max_key,max_value)