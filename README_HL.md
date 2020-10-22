# What's the Deal with Horror These Days?

## Links to Relevant Notebooks

[Grabbing Data](../grabbing_data.ipynb)

[EDA and Comparison](../eda_and_comparison.ipynb)

[Modeling](../modeling.ipynb)


## Problem Statement

A horror novel publisher hires a data scientist to determine the up-and-coming trends in the horror genre. This particular publisher knows that up-and-coming trends are not just reflected in new media (like movies, TV shows, and books), but also the smaller, self-publishing superfan communities, like WattPad and r/nosleep. Keeping one's finger on the pulse of the direction of the horror genre involves watching trends coming from multiple sources. To do so, the editor wants the data scientist to scrape r/creepypasta and r/nosleep to determine the topics and language most frequently used in recent posts.

Secondarily, the data scientist discovers, through their scraping process and their own research, that r/nosleep seems to have more interesting, original horror content than r/creepypasta, and hopes to convince the horror novel publisher that r/nosleep is the better subreddit from which to get information. The data scientist will develop a model that can discern between the two subreddits in order to prove that they have fundamentally different content. The data scientist will test several pipelines with different vectorizers (CountVectorizer and TF-IDF Vectorizer) and classifier models (Multinomial NB and RandomForest) to determine the best method for differentiating between the two subreddits. A successful model will have both a high accuracy score and a high sensitivity (where the positive class will be r/nosleep), because, after all, the data scientist is pretty sure that r/nosleep is the better horror subreddit and wants a model that accurately places posts in that subreddit.

## Data Description

We'll grab data from [r/nosleep](https://www.reddit.com/r/nosleep/) and [r/creepypasta](https://www.reddit.com/r/creepypasta/) using a custom function with the PushShift API. For reference, nosleep is billed as "a place for authors to share their original horror stories." and creepypasta's description is "For fans of the genre to post original works, discussions, and more." By, "the genre," the writer of that description is referring to the entire CreepyPasta ouevre, which according to the [Wikipedia page](https://en.wikipedia.org/wiki/Creepypasta), dates back to 2001 - predating [Reddit](https://en.wikipedia.org/wiki/Reddit) itself, which was created in 2005. Although the creepypasta [website](https://www.creepypasta.com/) is home to a lot of great original stories (some better than others), the creepypasta subreddit is not as popular and not as widely-used for original content as nosleep. However, comparing them to see the trending topics between them and using a model to tell the difference between written posts is still intriguing.

Back to the function: after starting out with a PushShift API url, we'll start the function. To appease Reddit's rules about pulling, we'll incorporate a sleep function that takes a 5-second rest between pulling 100 posts. In the function, we'll also clean out most of the posts in the 'subreddit' column that don't have the content we need. Since the posts on these two subreddits are pretty text-heavy, we'll just pull a little over a thousand posts from each. We end up with a dataframe with the following features:

| Column Name   | Description                    |
| ------------- |:------------------------------:|
| subreddit   | Which subreddit the post was pulled from - nosleep or creepypasta  |
| title   | Title of the post |
| selftext   | The text of the post - the meat in this sandwich! |


## Comparison & Visualizations

Once I was certain I had a large (atleast 2,000 row), reasonably-equally-split dataset with no null values, especially in the selftext column, I used CountVectorizer on the dataframe to lemmatize, strip, and count the most common words.

My first pass, without the use of any hyperparameters in the CountVectorizer, was nothing great:

![alt text](https://git.generalassemb.ly/hrl-dev/project_3/blob/master/Images/baseline_words.png "Baseline Common Words")

I wanted some more interesting words to present as common topics to the horror novel puvlisher. So, I added stopwords = 'english' as a hyperparameter, making use of scikitlearn's embedded stopwords list. That was slightly better, but still no stunning information:

![alt text](https://git.generalassemb.ly/hrl-dev/project_3/blob/master/Images/stopwords_words.png "Common Words: Stopwords Only")

Now, we have "time," "door," and "eyes," all of which are mildly interesting as common occurences in horror texts (very, very mild... "salt is a spice" mild), but none are enough to bring to the horror publisher. So, I added some new hyperparameters: I set the max_df to .5, and the max_features to 1,000. Setting the max_df meant that I would only catch words that were in - at most - 50% of the selftext posts, which would likely catch less common words. I set max_features mostly because I had more than 38,000 features, and I didn't need all of those. So, let's look at the results:

![alt text](https://git.generalassemb.ly/hrl-dev/project_3/blob/master/Images/max_df_fifty_words.png "Common Words: max_df .5")

Okay, so now we have "face," "head," "night," and "house" - those are definitely closer to "trending horror topics." At the very least, they're close to actual tropes that occur in the horror genre, if not "topics." A story might not be ABOUT night, or about a house, but according to our findings, many stories will contain those elements. So, closer to good information for the publisher - but not quite there yet! Let's skim a bit more off the top by setting the max_df to .25:

![alt text](https://git.generalassemb.ly/hrl-dev/project_3/blob/master/Images/max_df_25_words.png "Common Words: max_df .25")

So, we definitely have a parent theme. Freud would dine out for weeks on how often "mother" comes up in horror stories. That's something to relay to the publisher: "recent self-publishing horror authors have a tendency to lean on parent themes." However, I still think we could get some more interesting information if we set the max_df even lower - let's say .15:

![alt text](https://git.generalassemb.ly/hrl-dev/project_3/blob/master/Images/max_df_15_words.png "Common Words: max_df .15")

Now, we're really getting down to it. "Game" is especially interesting to me - a few recent horror movies have been based around games (Truth or Dare (2018), Ready or Not (2019)). One of the most famous horror film series of the last decade, the Saw films, sees its main antagonist telling victims, "I want to play a game." "Game" as a common, trending topic in recent self-published horror stories is very significant to me. "Creature" and "woods" are also important to note. Going even a little bit more specific:

![alt text](https://git.generalassemb.ly/hrl-dev/project_3/blob/master/Images/max_df_10_words.png "Common Words: max_df .1")

Blatantly ignoring some of the more X-rated topics here, we have some very interesting developments: "basement," "truck," "doctor," and "video." "Video" has been a popular topic in horror films and books since the invention of the VHS format, but it became even more popular after 2002's The Ring (a remake of 1998's Ringu (Japanese), itself an adaptation of Koji Suzuki's 1991 novel of the same name), one of the most popular and well-known horror movies of the 21st century.

Of course, denying the existence of the more explicit topics can't be done forever - it is important for the horror publisher to know that many members of their audience are interested in more Adult content in general.

Refining further, to a max_df of .05, proved to be too much, and brought up a bunch of names. However, it also brought up "cabin," which is a fun horror trope in my opinion.

My next trick was to bring in two-word n-grams, just to see what we could see there. Nothing too interesting, but when I brought the max_df back up to .35 as an experiment, we got some cool new words:

![alt text](https://git.generalassemb.ly/hrl-dev/project_3/blob/master/Images/max_df_35_words.png "Common Words: max_df .35")

Here, we have some very interesting developments that we may have missed by not doing max_df = .35 before. Not so interesting on the n-gram front, but we now have "light," "bed," "blood," and my favorite, "phone." Phones are tricky things in the horror universe: they're a connection, but a distant one. The person on the other end of the line can't help you if you're in a dire situation. On the other hand, you never really know where the caller is, so they may be closer than you think - thus the famous "the call is coming from inside the house" story, and the iconic opening sequence of the 1996 film Scream. 

I also love "light" as a common topic, because while the trope may be that turning on the light can save you from what lurks in the dark, how much scarier are horrific things done in broad daylight? Case in point: the 2019 A24 film Midsommar, otherwise known as the lightest, brightest horror movie I've ever seen.

Looking over these comparisons, it appears that our most common (interesting) topics that we can report to the publisher are "game," "cabin," "video," "woods," "creature," "phone," "light," a strong parental theme, and a trend towards rather "adult" topics. That's a hefty list. I hope they like it!

## Modeling

In doing a little further research into the r/nosleep and r/creepypasta subreddits, I came to a conclusion: r/nosleep is a place for interesting, original, new horror stories, and r/creepypasta is kind of the car glovebox of the horror community - people just throw whatever in there and then forget about it. Creepypasta, as an idea, has existed for almost 20 years, but the subreddit is not the best repository for new horror stories. 

In coming to this conclusion, I determined that it would be important to convince the horror novel publisher that not only was nosleep better than creepypasta, but nosleep and creepypasta had fundamentally different content, and nosleep was the place to look for good original horror. So, I decided to use a few different classifier models to see if my hypothesis was correct.

Because I care about the classification of nosleep posts, I optimized my models both for accuracy and sensitivity. Accuracy matters for reasons that should be obvious - I want my classifier model to classify things accurately - and the sensitivity score is important to me because while I think there COULD be posts in r/creepypasta that are decent, original horror stories like the ones in r/nosleep, I think most of the posts in r/creepypasta are not that kind of high-quality original horror. Therefore, I would expect more r/creepypasta posts to be misidentified as r/nosleep posts than the opposite, so r/nosleep posts - my positive class - being correctly identified is more important to me. 

I found that I had a baseline accuracy score of 0.510338 to beat - i.e., if I built a very simple model that predicted 0 (r/creepypasta), the majority class, for every single post, it would be right 51% of the time. Well, let's do this.

I started with CountVectorizer for lemmatization, stemming, stripping, and word-limitation, and I paired it in a pipeline with Multinomial Naïve Bayes. I chose MNB because I felt it would be an excellent binary classifier for the type of data I'm dealing with. I GridSearched over a series of parameters in both the CV and MNB, but ended up with a non-ideal accuracy and sensitivity score.

Following that, I tried a Term Frequency-Inverse Document Frequency (TF-IDF) Vectorizer in combination with my MNB model and got a slightly better accuracy score and sensitivity, followed by a TF-IDF/Random Forest pipeline with a significantly better accuracy and sensitivity, and finally, a TF-IDF/Logistic Regression pipeline that produced slightly worse scores than the Random Forest. Let's look at those results in a table:

| Vectorizer & Model   | Accuracy Score  | Sensitivity Score |
| ------------- |:-------------------:| ---------:|
| CountVectorizer/Multinomial Naïve Bayes | 0.69123 | 0.65988 |
| TF-IDF Vectorizer/Multinomial Naïve Bayes   | 0.75018 | 0.84302 |
| TF-IDF Vectorizer/Random Forest   | 0.79235 | 0.91162 |
| TF-IDF Vectorizer/Logistic Regression   | 0.78456 | 0.85756 |


So, the Random Forest decision tree model with the TF-IDF vectorizer performed best on my data. It was able to accurately place posts back into their home subreddits using only their language 79.2% of the time. And, as I predicted, it was able to accurately call a nosleep post a nosleep post 91% of the time - because they're outputting better, more original horror! The creepypasta posts were messing up the accuracy in general, and the specificity was likely much lower, and I blame this on confused redditors posting their original horror stories in the wrong place.

## Conclusion and Recommendations

What we can bring to the horror novel publisher is the following information: a list of truly interesting horror topics: "game," "cabin," "video," "woods," "creature," "phone," "light," and "blood," to name a few - that are very popular on two horror-themed subreddits, and the knowledge that while a creepypasta post might be mistaken for a nosleep post, a nosleep post is rarely mistaken for a creepypasta post - because nosleep is better. Get your trending horror ideas from nosleep (but no plagiarizing!).

To make this model even better, I would recommend refining the natural language processing... process using more advanced tools. I've heard that SpaCy, for example, is a strong competitor to nltk, and might even be faster in some cases. It would be interesting to test out some of its abilities. A neural network trained on tens of thousands of posts would probably be even better at accurately classifying which posts belong in which subreddit than my Random Forest model trained on a little more than a thousand posts from each, so it would be interesting to test those capabilities.

And of course, the trending topics of horror are always changing. Pulling posts a month from now may lead to completely different results than the posts in my data. And pulling the posts only from the top submissions in each subreddit, as opposed to just the most recent submissions, might give some insight into the most popular topics in self-published horror.

