# Extracting Movie Tags from a Plot Synopsis
We choose to watch a movie to relax and get disconnected from the usual schedule. And we expect it to be good according to our taste, like and dislike. 

When we plan to take 2-3 hours out of our busy schedule, we want that to be utilized completely for relaxation or entertainment. To achieve so, we go by the filters like actors, director, genres and more. Using those filters will help us to set an expectation as well as help us to choose a movie to watch.

Genres in movies are based on similarities either in narrative elements or in the emotional response to the movie. We can have a good idea about narrative elements and possible emotional response just by analyzing a movie plot synopsis.

In this blog, we will discuss a problem of movie synopsis analysis to extract genre(s) of a movie. For this exercise, we will be using a dataset provided by RiTUAL (Research in Text Understanding and Analysis of Language) Lab. More information is available here: [http://ritual.uh.edu/mpst-2018/](http://ritual.uh.edu/mpst-2018/).

Such automated genre extracting system will also help to build better recommendation systems to predict similar movie and help us to know what to expect from a movie. This dataset contains around 14k movie synopsis derived into train, validation and test set. All the plots are categorized into one or multiple genres. Here, there is a total of 71 unique genres.

We would first import the dataset into a pandas data frame. The data can be downloaded from [here](https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-with-tags). Below is how dataset looks like in raw form:

![MPST Data](https://raw.githubusercontent.com/anandborad/MPST/master/images/MPST%20data.png)

## 1. Understanding the dataset

From the above dataset, let's take a close look at different columns:

1. imdb_id: Internet Movie Database (IMDb) is the most popular and authoritative source to learn about movie or TV series. For every digital content, IMDb generates a unique identifier which is accepted all around the internet. Here, imdb_id is a unique identifier and should be unique for each datapoint. If there are duplicate imdb_id, that simply means we do have duplicates data-points in our data and we need to remove those.

2. title: the title of the movie
3. Plot_synopsis: Plot synopsis is the narrative explanation of a movie plot means a summary of a screenplay. It introduced a lead character and what it does in a movie. Below is the plot synopsis for the movie ‘Hansel and Gretel’ (imdb_id: tt1380833):

```
‘Hansel and Gretel are the young children of a poor woodcutter. When a great famine settles over the land, the woodcutter\'s second, abusive wife decides to take the children into the woods and leave them there to fend for themselves, so that she and her husband do not starve to death, because the kids eat too much. The woodcutter opposes the plan but finally, and reluctantly, submits to his wife\'s scheme. They were unaware that in the children\'s bedroom, Hansel and Gretel......’
```

4. split: this column defines whether a data-points belongs to a train, test or validation set.

5. synopsis_source: It gives information about the synopsis source, either IMDb or Wikipedia.

6. tags: Tags are labelled genres for a movie. It may take multiple values for a single movie. This will be our prediction labels.
If we take a closure look, a single tag may have space or a ‘-’. We want our tags to be similar form and thus we will replace whitespace and a dash with an underscore (‘_’). Also, we will separate tags by space instead of the comma. Below is how it looks:

![processsed tags](https://raw.githubusercontent.com/anandborad/MPST/master/images/2_MPST_ptags.png)

## 2. Check for missing and duplicate data

Fortunately, there is no missing text in any of the columns but there is for sure, duplicate data in the dataset. 

As discussed earlier, if an ‘imdb_id’ column has duplicate, then data must be duplicate. But here, there are few data-points where ‘imdb_id’ is different but the content for ‘title’, ‘plot_synopsis’, and ‘synopsis_source’ are same. Take a look at below image:

![duplicate](https://raw.githubusercontent.com/anandborad/MPST/master/images/3_MPST_duplicates.png)

We will be removing such duplicate points with below code:

```python
data= mpst_df.drop_duplicates(['title','plot_synopsis', 'ptags'])
```

Above code will remove all duplicate rows which have same 'title','plot_synopsis', and 'ptags' excluding the first record.

## 3. Exploring data

### 3.1 Tags per movie

As discussed earlier, a movie may consist of more than one genre and this will be interesting information to look into.

```python
# tags_count is an array containing number of tags for each movie
sns.countplot(tags_count)
plt.title("Number of tags in the synopsis ")
plt.xlabel("Number of Tags")
plt.ylabel("Number of synopsis")
plt.show()
```
![No_of_tags](https://raw.githubusercontent.com/anandborad/MPST/master/images/4_Tag_Analysis_no_of_tags.png)

There are 5516 movies which contain only one genre and 1 movie which is labelled for 25 tags.

### 3.2 Tags frequency analysis

It will be a good idea to analyse tag frequencies to know about frequent and rare tags. Here we can conclude that "murder" is the most frequent tag (5782 occurrences) while "christian film" is the least frequent tag (42 occurrences).
```python
sorted_freq_df=freq_df.sort_values(0, ascending=False)
sorted_freq_df.head(-1).plot(kind='bar', figsize=(16,7), legend=False)
i=np.arange(71)
plt.title('Frequency of all tags')
plt.xlabel('Tags')
plt.ylabel('Counts')
plt.show()
```
![tag_frequency](https://raw.githubusercontent.com/anandborad/MPST/master/images/5_Tag_Analysis_All.png)

If we consider only the top 20 tags, below is how it looks like:

![tag_freq_top20](https://raw.githubusercontent.com/anandborad/MPST/master/images/6_Tag_Analysis_20.png)

### 3.3 WordCloud for tags

Creating a word cloud of a plot synopsis text for a particular tag will help us to understand about the most frequent words for that tag. We will create a word cloud for a murder tag.

```python
# Creating column to indicate whether a murder tag exists or nor for a movie
data['ptags_murder']= [1 if 'murder' in tgs.split() else 0 for tgs in data.ptags]

# creating a corpus for movies with murder tag
murder_word_cloud=''
for plot in data[data['ptags_murder']==1].plot_synopsis:
  murder_word_cloud = plot + ' '
  
from wordcloud import WordCloud

#building a wordcloud
wordcloud = WordCloud(width=800, height=800, collocations=False, 
                     background_color='white').generate(murder_word_cloud)

plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.title("Words in plots for movies with murder tag")
plt.axis("off")
plt.show()
```
![wordcloud](https://raw.githubusercontent.com/anandborad/MPST/master/images/7_MPST_wordclod.png)
Here we can see words like murder, police, attempts, charged, etc do have a connection with a tag murder semantically.
We can do such analysis for all the tags but as we require to cover a lot of other stuff, it won’t be a good idea to increase a length of a blog including all those tag analysis.
## 4. Text preprocessing

Text in a raw format does have things like HTML tags, special characters, etc, which need to be removed before using text to build a machine learning model. Below is the procedure I used for text processing.

1. Removing HTML tags
2. Removing special characters like #, _ , -, etc
3. Converting text to lower case
4. Removing stop words
5. Stemming operation

```python
# function to remove html tags
def striphtml(data):
 cleanr = re.compile('<.*?>')
 cleantext = re.sub(cleanr, ' ', str(data))
 return cleantext
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# function to pre-process the text
def text_preprocess(syn_text):
 syn_processed = striphtml(syn_text.encode('utf-8')) # html tags removed
 syn_processed=re.sub(r'[^A-Za-z]+',' ',syn_processed) # removing special characters
 words=word_tokenize(str(syn_processed.lower())) # device into words and convert into lower

 syn_processed=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and len(j)!=1) #Removing stopwords and joining into sentence
return syn_processed
```
## 5. Train, validation, and test split

This is fairly simple as splitting strategy is already mentioned in the dataset itself.

```python
train = data[data['split']=='train']
val = data[data['split']=='val']
test = data[data['split']=='test']
```
## 6. Featurization of text

We can use multiple text featurization techniques such as a bag of words with n-grams, TFIDF with n-grams, Word2vec (average and weighted), Sentic Phrase, TextBlob, LDA topic Modelling, NLP/Text-based Features, etc.
[An additional resource to learn about text featurization](https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41) 

For simplicity, I have used TFIDF with 1,2,3-grams featurization which gives pretty good result in fact.

```python
vectorizer = TfidfVectorizer(min_df=0.00009, smooth_idf=True, tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,3))
x_train = vectorizer.fit_transform(train['processed_plot'])
x_val = vectorizer.transform(val['processed_plot'])
x_test = vectorizer.transform(test['processed_plot'])
```
TFIDF with (1,3) grams generated few less than 7 million features. Below 2 resources will help you to learn more about TFIDF: 
- [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [https://www.onely.com/blog/what-is-tf-idf/](https://www.onely.com/blog/what-is-tf-idf/)

## 7. Machine learning modelling

Before jumping to modelling, let's discuss the evaluation metrics. Choosing an evaluation metrics is the most essential task as its a bit tricky depending on task objective.

Our problem is a multi-label classification problem where there may be multiple labels for a single data-point. We want our model to predict the right categories as much as it can while avoiding the wrong prediction. The accuracy is not a so good metric for this task. For this task, micro averaged F1 score is the better metric.

### Micro averaged F1 score:

F1-Score is derived from recall and precision values. To calculate micro f1 score, we need to calculate micro averaged precision and recall, hence:

```
Micro F1-score = 2/(micro recall ^-1 + micro precision^-1)
```
Let’s understand how to calculate micro averaged precision and recall from an example. let’s say for a set of data, the system's

```
True positive (TP1)= 12
False positive (FP1)=9
False negative (FN1)=3
```
Then precision (P1) and recall (R1) will be 57.14 and 80

and for a different set of data, the system's

```
True positive (TP2)= 50
False positive (FP2)=23
False negative (FN2)=9
```

Then precision (P2) and recall (R2) will be 68.49 and 84.75

Now, the average precision and recall of the system using the Micro-average method is
```
Micro-average of precision = (TP1+TP2)/(TP1+TP2+FP1+FP2) = (12+50)/(12+50+9+23) = 65.96
Micro-average of recall = (TP1+TP2)/(TP1+TP2+FN1+FN2) = (12+50)/(12+50+3+9) = 83.78
```
Above explanation is borrowed from this [brilliant blog](http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html).

One more thing before jumping to modelling.

Let's look into a research paper by dataset publisher which I have already mentioned (a link to paper) in the beginning. They used a micro F1 score along with tag recall and tag learned as evaluation metrics. Below is the snap of their result:

![from_research_paper](https://raw.githubusercontent.com/anandborad/MPST/master/images/7_MPST_from_paper.png)

Maximum f1-micro they’ve got is 37.8. Let’s see how much we would able to get with a simple model.

To solve a multi-label classification problem, we would use OneVsRest classifier which at a time classifies one class iteratively. [Learn more here](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html). 

I have tried with support vector machine and logistic regression. Logistic regression turns out to be a better option.

```python
classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=1e-5, penalty='l1'), n_jobs=-1)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

print("micro f1 score :",metrics.f1_score(y_test, predictions, average = 'micro'))
print("hamming loss :",metrics.hamming_loss(y_test,predictions))
```
```
micro f1 score : 0.34867686650469504
hamming loss : 0.04710189661231041
```
We got pretty good micro-f1 (0.349 vs 0.378, best result mentioned in paper) with LR model. 

## The final trick! 

Precision and recall depend upon TP, FP, TN, and FN. All that metrics depends on predictions (0 or 1) but not on the probability of a prediction.

What if we figure out a way to use probability and check whether it improves micro f1 score or not. To do so, we will use the fact that the default threshold for prediction is 0.5. That simply means we assign 1 if prediction probability id 0.5 or above and 0 otherwise.

Here we will try different threshold value to learn a threshold that maximizes the micro-f1 score.

Let's try with 0.20 to 0.30 as threshold values.
```python
yhat_val_prob= classifier.predict_proba(x_val)
for t in list(range(20, 31, 1)):
 print(t*0.01)
 pred_lb=np.asarray(yhat_val_prob>t*0.01, dtype='int8')
 print("micro f1 scoore :",metrics.f1_score(y_val, pred_lb, average = 'micro'))
```
```
0.2
micro f1 scoore : 0.37533010563380287
0.21
micro f1 scoore : 0.3761086785674189
0.22
micro f1 scoore : 0.3761457378551788
0.23
micro f1 scoore : 0.37720425084666587
0.24
micro f1 scoore : 0.3766496254904292
0.25
micro f1 scoore : 0.3773150950248154
0.26
micro f1 scoore : 0.378451509747248
0.27
micro f1 scoore : 0.3784528656435954
0.28
micro f1 scoore : 0.37878787878787873
0.29
micro f1 scoore : 0.377741831122614
0.3
micro f1 scoore : 0.3768382352941177
```
From above, we can see that using 0.28 as a threshold value we can get the best micro f1 score around 0.379 on the validation dataset. 

Let's see what result the same threshold can give for test data set.
```python
yhat_test_prob= classifier.predict_proba(x_test)
pred_lb=np.asarray(yhat_test_prob>0.28, dtype='int8')

print("micro f1 score :",metrics.f1_score(y_test, pred_lb, average = 'micro'))
print("hamming loss :",metrics.hamming_loss(y_test,pred_lb))
```
```
micro f1 score : 0.3737731458059294
hamming loss : 0.05899764066633302
```
Here, we improved result on test data from 0.349 to 0.374 just by adjusting threshold values. 

As mentioned in the paper, we can try with different featurization instead of just 1,2,3-grams TFIDF. Various resources suggest that 3,4-grams character based TFIDF performs better with this dataset. We can definitely extend our experiments with threshold adjustment which gave a better result in our experiment. 
