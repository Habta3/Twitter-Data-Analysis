import json
import pandas as pd
from textblob import TextBlob


def read_json(json_file: str)->list:
    """
    json file reader to open and read json files into a list
    Args:
    -----
    json_file: str - path of a json file
    
    Returns
    -------
    length of the json file and a list of json
    """
    
    tweets_data = []
    for tweets in open(json_file,'r'):
        tweets_data.append(json.loads(tweets))
    
    
    return len(tweets_data), tweets_data

class TweetDfExtractor:
    """
    this function will parse tweets json into a pandas dataframe
    
    Return
    ------
    dataframe
    """
    def __init__(self, tweets_list):
        
        self.tweets_list = tweets_list

    # an example function
    def find_statuses_count(self)->list:
        statuses_count=[]
        for twt in self.tweets_list:
            statuses_count.append(twt['user']['statuses_count'])
            return statuses_count
        
    def find_full_text(self)->list:
        text = []
        for twt in self.tweets_list:
            print(twt)
            text.append(twt)
            return text
       
    
    def find_sentiments(self, text)->list:
        polarity=[]
        subjectivity = []
        for twt in self.tweets_list:
            blob = TextBlob(twt)
            sentiment = blob.sentiment
            polarity.append(sentiment.polarity)
            subjectivity.append(sentiment.subjectivity)
        return polarity, subjectivity

    def find_created_time(self)->list:
        created_at = []
        for twt in self.tweets_list:
            created_at.append(twt['created_at'])
        return created_at

    def find_source(self)->list:
        source = []
        for twt in self.tweets_list:
            source.append(twt['source'])
        return source
    def find_screen_name(self)->list:
        screen_name = []
        for scr_name in self.tweets_list:
            screen_name.append(scr_name["user"]["screen_name"])
        return screen_name

    def find_followers_count(self)->list:
        followers_count=[]
        for follower in self.tweets_list:
            followers_count.append(follower["user"]["followers_count"])
        return followers_count

    def find_friends_count(self)->list:
        friends_count = []
        for fr_count in self.tweets_list:
            friends_count.append(fr_count["user"]["friends_count"])
        return friends_count

    def is_sensitive(self)->list:
        try:
            is_sensitive = [x['possibly_sensitive'] for x in self.tweets_list]
        except KeyError:
            is_sensitive = None
        return is_sensitive

    def find_favourite_count(self)->list:
        favourite_count = []
        for fav_count in self.tweets_list:
            favourite_count.append(fav_count["user"]["favourites_count"])
        return favourite_count
    def find_retweet_count(self)->list:
        retweet_count = []
        for ret_count in self.tweets_list:
            retweet_count.append(ret_count["retweet_count"])
        return retweet_count
    def find_hashtags(self)->list:
        hashtags = []
        for hash_tag in self.tweets_list:
            hashtags.append(", ".join([item["text"] for item in hash_tag["entities"]["hashtags"]]))
        return hashtags

    def find_mentions(self)->list:
        mentions = []
        for mention_count in self.tweets_list:
            mentions.append(", ".join([item["screen_name"] for item in mention_count["entities"]["user_mentions"]]))
        return mentions

    def find_location(self)->list:
        try:
            location = self.tweets_list['user']['location']
        except TypeError:
            location = ''
        
        return location

    def get_tweet_df(self, save=False)->pd.DataFrame:
        """required column to be generated you should be creative and add more features"""
        
        columns = ['created_at', 'source', 'original_text','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count', 
            'original_author', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place']
        
        created_at = self.find_created_time()
        source = self.find_source()
        text = self.find_full_text()
        polarity, subjectivity = self.find_sentiments(text)
        lang = self.find_lang()
        fav_count = self.find_favourite_count()
        retweet_count = self.find_retweet_count()
        screen_name = self.find_screen_name()
        followers_count = self.find_followers_count()
        friends_count = self.find_friends_count()
        sensitivity = self.is_sensitive()
        hashtags = self.find_hashtags()
        mentions = self.find_mentions()
        location = self.find_location()
        data = zip(created_at, source, text, polarity, subjectivity,lang, fav_count, retweet_count, screen_name, followers_count, friends_count, sensitivity, hashtags, mentions, location)
        df = pd.DataFrame(data=data, columns=columns)

        if save:
            df.to_csv('processed_tweet_data.csv', index=False)
            print('File Successfully Saved.!!!')
        
        return df

                
if __name__ == "__main__":
    # required column to be generated you should be creative and add more features
    columns = ['created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count', 
    'original_author', 'screen_count', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']
    _, tweet_list = read_json("africa_twitter_data.json")
    tweet = TweetDfExtractor(tweet_list)
    tweet_df = tweet.get_tweet_df() 

    # use all defined functions to generate a dataframe with the specified columns above