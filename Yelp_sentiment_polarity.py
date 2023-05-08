# In[5]:
import pandas as pd
import seaborn as sns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC


# In[2]:
# Read business DataFrame
business_df = pd.read_csv('yelp_business.csv',usecols=[
    'business_id', 'state','categories','stars'])

# In[3]:
## Select US Resturants by state
us_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
us_business_df=business_df.loc[business_df['state'].isin(us_states)]
business_df.head()
us_restaurants_df=us_business_df[us_business_df['categories'].str.contains('Restaurants')].copy()


# In[7]:
# Filter by cuisine_type
us_restaurants_df.loc[:, 'category'] = pd.Series(dtype='object')
cuisine_types = ['American', 'Mexican', 'Italian', 'Japanese','Chinese', 'Thai', 'Mediterranean',
                'French', 'Vietnamese', 'Greek', 'Indian', 'Korean', 'Hawaiian', 'African',
                'Spanish', 'Middle_eastern']

for cuisine_type in cuisine_types:
    us_restaurants_df.loc[us_restaurants_df.categories.str.contains(cuisine_type),'category'] = cuisine_type


# In[6]:
## Drop null values and unused columns
us_restaurants_df=us_restaurants_df.dropna(axis=0, subset=['category'])
del us_restaurants_df['categories']
us_restaurants_df=us_restaurants_df.reset_index(drop=True)
us_restaurants_df.head(10)


# In[9]:
print("DF Shape ", us_restaurants_df.shape)
print("Duplicate Values ", us_restaurants_df.business_id.duplicated().sum())
print(us_restaurants_df.dtypes)
print("Null Values ",us_restaurants_df.isnull().sum())


# In[11]:
# Read Yelp Review DF
review_df = pd.read_csv('yelp_review.csv',usecols=[
    'review_id', 'business_id', 'stars', 'text'])
review_df.head()

# In[12]:
print("Null Values",review_df.isnull().sum())
print("Duplicate Values", review_df.review_id.duplicated().sum())


# In[14]:
# merge restaurant Data and reviews
restaurant_review_df = pd.merge(us_restaurants_df, review_df, on = 'business_id')
print(len(restaurant_review_df))
restaurant_review_df.rename(columns={'stars_x':'avg_star','stars_y':'rating'}, inplace=True)
restaurant_review_df['review_count'] = restaurant_review_df.text.str.replace('\n','').                                           str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','').map(lambda x: len(x.split()))
    

# In[16]:
# Label reviews as positive or negative
restaurant_review_df['labels'] = ''
restaurant_review_df.loc[restaurant_review_df.rating >=3, 'labels'] = 'positive'
restaurant_review_df.loc[restaurant_review_df.rating <3, 'labels'] = 'negative'
restaurant_review_df.head()


# In[18]:
plt.figure(figsize=(11,7))
us_restaurant_count = us_restaurants_df.category.value_counts()
sns.countplot(y='category',data=us_restaurants_df, 
              order = us_restaurant_count.index)
plt.xlabel('Restaurants Count')
plt.ylabel('Cuisine Type')
plt.title('Restaurant Count by Cuisine Type')
for  i, v in enumerate(us_restaurants_df.category.value_counts()):
    plt.text(v, i, str(v))


# In[20]:
plt.figure(figsize=(11,6))
us_restaurant_count = us_restaurants_df.state.value_counts()
sns.barplot(x=us_restaurant_count.index, y=us_restaurant_count.values)
plt.ylabel('Restaurant Count')
plt.xlabel('US State')
plt.title('Restaurant Count by US State')
for  i, v in enumerate(us_restaurant_count):
    plt.text(i, v, str(v))


# In[23]:
plt.figure(figsize=(11,7))
us_restaurant_count = restaurant_review_df.rating.value_counts()
sns.countplot(y='rating',data=restaurant_review_df, 
              order = us_restaurant_count.index)
plt.xlabel('Restaurants Count')
plt.ylabel('Rating')
plt.title('Ratings for Restaurants')
for  i, v in enumerate(restaurant_review_df.rating.value_counts()):
    plt.text(v, i, str(v))


# In[16]:

review_counts = pd.pivot_table(restaurant_review_df, values=["review_id"], index=["category"], columns=["labels"], aggfunc=len, margins=True)
t_percent = review_counts.div(review_counts.iloc[:,-1], axis=0).iloc[:-1,-2].sort_values(ascending=False)
ax=sns.barplot(x=t_percent.index, y= t_percent.values)
plt.xlabel('Cuisine Type')
plt.ylabel('Percentage')
plt.title('Percentage of Positive Reviews')
plt.ylim([0.60, 0.90])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
for  i, v in enumerate(t_percent.round(2)):
    plt.text(i, v, str(v))
    

# In[18]:
details = restaurant_review_df.groupby(['category','labels'])['review_count'].mean().unstack()
plt.figure(figsize=(11,8))
sns.heatmap(details, cmap='BrBG')

# In[21]:
# Convert text to lower case
restaurant_review_df.text = restaurant_review_df.text.str.lower()
restaurant_review_df['removed_punct_text']= restaurant_review_df.text.str.replace('\n','').                                           str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','')


# In[22]:
# Import files which contains common meaningless words like good
likes = pd.read_csv('positive_words.txt', sep=",", header=None)[0].tolist()
dislikes = pd.read_csv('negative_words.txt', sep=",", header=None)[0].tolist()


# In[23]:
# Get DF by Cuisine Type
def get_cuisine_review(types):
    dataframe = restaurant_review_df[['removed_punct_text','labels']][restaurant_review_df.category==types]
    print(len(dataframe))
    dataframe.reset_index(drop=True, inplace =True)
    dataframe.rename(columns={'removed_punct_text':'text'}, inplace=True)
    return dataframe


def get_likes_dislikes_words(review_txt):
    words = [word for word in review_txt.split() if word in likes+dislikes]
    words = ' '.join(words)
    return words


# In[50]:
# Calculate Polarity
def get_polarity(df):
    df.text = df.text.apply(get_likes_dislikes_words)
    review_list = list(df['text'])
    label_list = list(df['labels'])
    
    cv = CountVectorizer()
    word_freq = cv.fit_transform(review_list)
    
    svc = LinearSVC()
    svc.fit(word_freq, label_list)
    feature_weight = svc.coef_[0]

    cuisine_polarity = pd.DataFrame({'score': feature_weight, 'word': cv.get_feature_names_out()})
    cuisine_reviews = pd.DataFrame(word_freq.toarray(), columns = cv.get_feature_names_out())
    cuisine_reviews['labels'] = label_list
    cuisine_frequency = cuisine_reviews[cuisine_reviews['labels'] =='positive'].sum()[:-1]
    
    cuisine_polarity.set_index('word', inplace=True)
    cuisine_polarity['frequency'] = cuisine_frequency
    
    cuisine_polarity.score = cuisine_polarity.score.astype(float)
    cuisine_polarity.frequency = cuisine_polarity.frequency.astype(int)
    
    cuisine_polarity['polarity'] = cuisine_polarity.score * cuisine_polarity.frequency / cuisine_reviews.shape[0]
    cuisine_polarity.polarity = cuisine_polarity.polarity.astype(float)
    
    return cuisine_polarity


# In[51]:
# Plot chart polarity and top words
def visualise_result(top_words, category):
    plt.figure(figsize=(11,6))
    sns.barplot(y=top_words.index, x=top_words.values)
    plt.xlabel('Polarity Score')
    plt.ylabel('Word')
    plt.title('Top Positive & Negative Words in the review of %s Restaurants ' % category)


# In[55]:
# Get Top words
def get_top_words(dataset, label, number=20):
    if label == 'positive':
        dataframe = dataset[dataset.polarity>0].sort_values('polarity',ascending = False)[:number]
    else:
        dataframe = dataset[dataset.polarity<0].sort_values('polarity')[:number]
    return dataframe

# In[52]:
# Get Polarity per Cuisine Type
cuisine_run = ['Japanese','Chinese', 'Thai',
                'French', 'Vietnamese', 'Greek', 'Indian']

for cuisine_type in cuisine_run:
    training_data = get_cuisine_review(cuisine_type)
    polarity_score = get_polarity(training_data)
    ps = get_top_words(polarity_score, 'positive')
    ng = get_top_words(polarity_score,'negative')
    top_review_words = polarity_score.loc[list(ps.index.values) + list(ng.index.values), 'polarity']
    visualise_result(top_review_words,cuisine_type)
