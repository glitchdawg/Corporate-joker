import praw
import config

reddit =praw.Reddit(client_id=config.C_id,client_secret=config.secret,user_agent=config.agent)

import pandas as pd
posts=[]

jokes_sub=reddit.subreddit('Jokes')
for post in jokes_sub.hot(limit=250000):
    posts.append([post.title,post.score,post.id,post.selftext])
posts =pd.DataFrame(posts,columns=['title','upvotes','id','body'])
posts.to_csv('dataset.csv')