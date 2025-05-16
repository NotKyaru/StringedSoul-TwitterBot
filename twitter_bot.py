import tweepy
import pandas as pd
import numpy as np
import re
import time
import random
import schedule
from datetime import datetime, timedelta
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import json
import logging

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("twitter_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

class TwitterBot:
    def __init__(self, api_key, api_secret, access_token, access_token_secret):
        """Initialize the Twitter bot with API credentials."""
        try:
            # Authenticate with Twitter
            auth = tweepy.OAuth1UserHandler(
                api_key, api_secret, access_token, access_token_secret
            )
            self.api = tweepy.API(auth)
            # Test authentication
            self.api.verify_credentials()
            logger.info("Authentication successful")
            
            # Initialize the data storage
            self.tweets_df = None
            self.topics = []
            self.hashtags = []
            self.posting_times = []
            self.config = {
                "posts_per_day": 3,
                "min_hashtags": 1,
                "max_hashtags": 5,
                "min_post_length": 80,
                "max_post_length": 240
            }
            
            # Load config if available
            self.load_config()
            
        except Exception as e:
            logger.error(f"Error initializing Twitter bot: {e}")
            raise
    
    def load_config(self):
        """Load configuration from file if available."""
        try:
            if os.path.exists("bot_config.json"):
                with open("bot_config.json", "r") as f:
                    self.config = json.load(f)
                logger.info("Configuration loaded from file")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open("bot_config.json", "w") as f:
                json.dump(self.config, f)
            logger.info("Configuration saved to file")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def fetch_user_tweets(self, count=200):
        """Fetch user's tweets to analyze posting pattern."""
        try:
            logger.info(f"Fetching {count} tweets for analysis")
            tweets = []
            for tweet in tweepy.Cursor(self.api.user_timeline, count=count, tweet_mode="extended").items(count):
                # Skip retweets and replies
                if not tweet.full_text.startswith("RT @") and tweet.in_reply_to_status_id is None:
                    tweets.append({
                        "id": tweet.id,
                        "text": tweet.full_text,
                        "created_at": tweet.created_at,
                        "favorite_count": tweet.favorite_count,
                        "retweet_count": tweet.retweet_count
                    })
            
            self.tweets_df = pd.DataFrame(tweets)
            logger.info(f"Fetched {len(tweets)} original tweets")
            
            # Save the tweets data for future reference
            if len(tweets) > 0:
                self.tweets_df.to_csv("tweets_data.csv", index=False)
                logger.info("Tweets data saved to CSV")
            
            return self.tweets_df
        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")
            return None
    
    def analyze_posting_pattern(self):
        """Analyze when the user typically posts."""
        try:
            if self.tweets_df is None or len(self.tweets_df) == 0:
                logger.warning("No tweets to analyze")
                return
                
            logger.info("Analyzing posting pattern")
            
            # Extract hour of day
            self.tweets_df['hour'] = self.tweets_df['created_at'].dt.hour
            
            # Count posts by hour
            hour_counts = self.tweets_df['hour'].value_counts().sort_index()
            
            # Find the most common posting hours
            top_hours = hour_counts.sort_values(ascending=False).head(self.config["posts_per_day"])
            self.posting_times = top_hours.index.tolist()
            
            logger.info(f"Most common posting times (hours): {self.posting_times}")
            return self.posting_times
        except Exception as e:
            logger.error(f"Error analyzing posting pattern: {e}")
            return []
    
    def extract_topics_and_hashtags(self):
        """Extract common topics and hashtags from tweets."""
        try:
            if self.tweets_df is None or len(self.tweets_df) == 0:
                logger.warning("No tweets to analyze for topics and hashtags")
                return
                
            logger.info("Extracting topics and hashtags")
            
            # Extract hashtags
            all_hashtags = []
            for text in self.tweets_df['text']:
                hashtags = re.findall(r'#(\w+)', text)
                all_hashtags.extend(hashtags)
            
            # Get most common hashtags
            hashtag_counts = Counter(all_hashtags)
            self.hashtags = [tag for tag, _ in hashtag_counts.most_common(20)]
            
            # Extract topics (significant words)
            stop_words = set(stopwords.words('english'))
            all_words = []
            
            for text in self.tweets_df['text']:
                # Remove URLs and mentions
                clean_text = re.sub(r'http\S+|@\S+|#\S+', '', text)
                # Tokenize
                tokens = word_tokenize(clean_text.lower())
                # Remove punctuation and stopwords
                words = [word for word in tokens if word not in stop_words 
                         and word not in string.punctuation
                         and len(word) > 3]
                all_words.extend(words)
            
            # Get most common words
            word_counts = Counter(all_words)
            self.topics = [word for word, _ in word_counts.most_common(50)]
            
            logger.info(f"Extracted {len(self.hashtags)} hashtags and {len(self.topics)} topics")
            
            # Save topics and hashtags
            with open("topics_hashtags.json", "w") as f:
                json.dump({
                    "topics": self.topics,
                    "hashtags": self.hashtags
                }, f)
            
            logger.info("Topics and hashtags saved to file")
            
            return self.topics, self.hashtags
        except Exception as e:
            logger.error(f"Error extracting topics and hashtags: {e}")
            return [], []
    
    def analyze_engagement(self):
        """Analyze which posts get the most engagement."""
        try:
            if self.tweets_df is None or len(self.tweets_df) == 0:
                logger.warning("No tweets to analyze for engagement")
                return
                
            logger.info("Analyzing engagement patterns")
            
            # Add engagement score (likes + 2*retweets)
            self.tweets_df['engagement'] = self.tweets_df['favorite_count'] + 2 * self.tweets_df['retweet_count']
            
            # Sort by engagement
            top_tweets = self.tweets_df.sort_values('engagement', ascending=False).head(10)
            
            # Analyze top tweets for patterns
            top_tweet_features = {
                "avg_length": top_tweets['text'].apply(len).mean(),
                "hashtag_count": top_tweets['text'].apply(lambda x: len(re.findall(r'#\w+', x))).mean(),
                "has_url": top_tweets['text'].apply(lambda x: 1 if 'http' in x else 0).mean(),
                "top_words": []
            }
            
            # Extract common words in top tweets
            top_words = []
            stop_words = set(stopwords.words('english'))
            
            for text in top_tweets['text']:
                clean_text = re.sub(r'http\S+|@\S+|#\S+', '', text)
                tokens = word_tokenize(clean_text.lower())
                words = [word for word in tokens if word not in stop_words 
                         and word not in string.punctuation
                         and len(word) > 3]
                top_words.extend(words)
            
            top_tweet_features["top_words"] = [word for word, _ in Counter(top_words).most_common(20)]
            
            logger.info("Engagement analysis complete")
            
            return top_tweet_features
        except Exception as e:
            logger.error(f"Error analyzing engagement: {e}")
            return None
    
    def generate_tweet(self):
        """Generate a tweet based on user's style."""
        try:
            if not self.topics or not self.hashtags:
                logger.warning("Topics or hashtags not available for tweet generation")
                return None
                
            logger.info("Generating new tweet")
            
            # Generate a tweet template
            templates = [
                "Just thinking about {topic1} and {topic2} today. {hashtags}",
                "I've been working on {topic1} lately. Thoughts? {hashtags}",
                "Does anyone else find that {topic1} is related to {topic2}? {hashtags}",
                "New insight: {topic1} could transform how we think about {topic2}. {hashtags}",
                "Interesting observation about {topic1} this week. {hashtags}",
                "Question of the day: How does {topic1} affect {topic2}? {hashtags}",
                "{topic1} is trending, but what about {topic2}? {hashtags}",
                "Hot take: {topic1} is underrated. {hashtags}",
                "My perspective on {topic1} has changed because of {topic2}. {hashtags}",
                "I'm researching {topic1} right now. Any resources to recommend? {hashtags}"
            ]
            
            # Select random template
            template = random.choice(templates)
            
            # Select random topics
            topic1 = random.choice(self.topics)
            topic2 = random.choice([t for t in self.topics if t != topic1])
            
            # Select random hashtags
            num_hashtags = random.randint(self.config["min_hashtags"], self.config["max_hashtags"])
            selected_hashtags = random.sample(self.hashtags, min(num_hashtags, len(self.hashtags)))
            hashtag_text = " ".join([f"#{tag}" for tag in selected_hashtags])
            
            # Generate tweet
            tweet = template.format(topic1=topic1, topic2=topic2, hashtags=hashtag_text)
            
            # Ensure tweet is within character limit
            if len(tweet) > self.config["max_post_length"]:
                tweet = tweet[:self.config["max_post_length"]-3] + "..."
            
            logger.info(f"Generated tweet: {tweet}")
            return tweet
        except Exception as e:
            logger.error(f"Error generating tweet: {e}")
            return None
    
    def post_tweet(self, tweet_text=None):
        """Post a tweet to Twitter."""
        try:
            if tweet_text is None:
                tweet_text = self.generate_tweet()
                
            if not tweet_text:
                logger.warning("No tweet text to post")
                return False
                
            logger.info(f"Posting tweet: {tweet_text}")
            
            # Post to Twitter
            self.api.update_status(tweet_text)
            
            logger.info("Tweet posted successfully")
            return True
        except Exception as e:
            logger.error(f"Error posting tweet: {e}")
            return False
    
    def schedule_posts(self):
        """Schedule posts for the day."""
        try:
            logger.info("Setting up scheduled posts")
            schedule.clear()
            
            # If no posting times are determined, use default spread
            if not self.posting_times:
                self.posting_times = [9, 13, 17]  # Default posting times
            
            # Schedule posts
            for i, hour in enumerate(self.posting_times[:self.config["posts_per_day"]]):
                schedule.every().day.at(f"{hour:02d}:00").do(self.post_tweet)
                logger.info(f"Scheduled post {i+1} at {hour:02d}:00")
            
            logger.info(f"Scheduled {self.config['posts_per_day']} posts for the day")
        except Exception as e:
            logger.error(f"Error scheduling posts: {e}")
    
    def run(self):
        """Run the bot continuously."""
        try:
            logger.info("Starting Twitter bot")
            
            # Fetch and analyze tweets if we don't have data
            if self.tweets_df is None:
                self.fetch_user_tweets()
                self.analyze_posting_pattern()
                self.extract_topics_and_hashtags()
                self.analyze_engagement()
            
            # Schedule posts
            self.schedule_posts()
            
            logger.info("Bot running, waiting for scheduled times to post")
            
            # Keep running
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            raise

if __name__ == "__main__":
    # Load credentials from environment variables or config file
    try:
        # Try to load from config file
        if os.path.exists("twitter_credentials.json"):
            with open("twitter_credentials.json", "r") as f:
                creds = json.load(f)
                api_key = creds.get("api_key")
                api_secret = creds.get("api_secret")
                access_token = creds.get("access_token")
                access_token_secret = creds.get("access_token_secret")
        else:
            # Try to load from environment variables
            api_key = os.environ.get("TWITTER_API_KEY")
            api_secret = os.environ.get("TWITTER_API_SECRET")
            access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
            access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
            
        if not all([api_key, api_secret, access_token, access_token_secret]):
            raise ValueError("Twitter credentials not found")
        
        # Initialize and run the bot
        bot = TwitterBot(api_key, api_secret, access_token, access_token_secret)
        bot.run()
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        print(f"Error: {e}")
