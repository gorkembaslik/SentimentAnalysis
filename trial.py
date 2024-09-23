# For Fetching Comments 
from googleapiclient.discovery import build 
# For filtering comments 
import re 
# For filtering comments with just emojis 
import emoji
# Analyze the sentiments of the comment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# For visualization 
import matplotlib.pyplot as plt
# import HttpError
from googleapiclient.errors import HttpError

import pandas as pd
import os
import requests

csv_file = 'youtube_data.csv'

def sentiment_scores(comment, polarity):
 
    # Creating a SentimentIntensityAnalyzer object.
    sentiment_object = SentimentIntensityAnalyzer()
 
    sentiment_dict = sentiment_object.polarity_scores(comment)
    polarity.append(sentiment_dict['compound'])
 
    return polarity

def is_video_sponsored(title, description):
    # Define a list of keywords that indicate sponsorship
    sponsorship_keywords = [
        "sponsored",
        "#ad",
        "advertisement",
        "partnership",
        "collaboration",
        "brought to you by",
        "thank you to",
        "in partnership with",
        "use code",
        "impressive product",
        "as pr"
    ]
    not_sponsorship_keywords = [
        "not sponsored",
        "unsponsored"
    ]
    
    # Convert title and description to lower case for case-insensitive matching
    title_lower = title.lower()
    description_lower = description.lower()

    for keyword in not_sponsorship_keywords:
        if keyword in title_lower or keyword in description_lower:
            return False
    
    # Check for keywords in the title and description
    for keyword in sponsorship_keywords:
        if keyword in title_lower or keyword in description_lower:
            return True
            
    return False

def get_category_name(category_id, api_key):
    url = f"https://www.googleapis.com/youtube/v3/videoCategories"
    params = {
        'part': 'snippet',
        'id': category_id,
        'key': api_key
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            category_title = data['items'][0]['snippet']['title']
            return category_title
        else:
            return "Category not found."
    else:
        return f"Error: {response.status_code}"

API_KEY = ''# Put in your API Key

youtube = build('youtube', 'v3', developerKey=API_KEY) # initializing Youtube API

with open("videos.txt", "r") as file:
    lines = file.readlines()
    line_num = 0
    for line in lines:
        line_num +=1
        # Taking the link from the file and slicing for video id
        video_id = line[-12:-1]
        print(f"\nvideo id: " + video_id)

        # Getting the channelId of the video uploader
        video_response = youtube.videos().list(
            part='snippet, statistics',
            id=video_id
        ).execute()

        # Splitting the response for channelID and other metrics
        video_snippet = video_response['items'][0]['snippet']
        video_statistics = video_response['items'][0]['statistics']

        uploader_channel_id = video_snippet['channelId']
        title = video_snippet['title']
        description = video_snippet['description']
        category_id = video_snippet['categoryId']

        videoCategory = get_category_name(category_id, API_KEY)
        view_count = video_statistics['viewCount']
        likes = video_statistics['likeCount']
        commentsCount = video_statistics['commentCount']

        # dislikes = video_statistics['dislikeCount'] youtube disabled the dislike data
        print(f"channel id: {uploader_channel_id}")
        print(f"Title: {title}")
        print(f"View Count: {view_count}")
        print(f"Likes: {likes}")
        print(f"Description: {description}\n\n")
        # print(f"Dislikes: {dislikes}\n\n")

        # Fetch comments
        print("Fetching Comments...")
        comments = []
        nextPageToken = None
        while len(comments) < 600:
            ## catch the video with disabled comments.
            try:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100, # You can fetch up to 100 comments per request
                    pageToken=nextPageToken
                )
                response = request.execute()
                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    # Check if the comment is not from the video uploader
                    if comment['authorChannelId']['value'] != uploader_channel_id:
                        comments.append(comment['textDisplay'])
                nextPageToken = response.get('nextPageToken')

                if not nextPageToken:
                    break
            except HttpError as e:
                if e.resp.status == 403 and "commentsDisabled" in str(e):
                    print(f"Comments are disabled for video id: {video_id}")
                    break
                else:
                    raise e
        # Print the 5 comments
        # print(comments[:5], end="\n\n")
        hyperlink_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        threshold_ratio = 0.65

        relevant_comments = []

        # Inside your loop that processes comments
        for comment_text in comments:

            comment_text = comment_text.lower().strip()

            emojis = emoji.emoji_count(comment_text)

            # Count text characters (excluding spaces)
            text_characters = len(re.sub(r'\s', '', comment_text))

            if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
                if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
                    relevant_comments.append(comment_text)

        # Print the relevant comments
        #print(relevant_comments[:5], end="\n\n")

        f = open(f"ytcomments{line_num}.txt", 'w', encoding='utf-8')
        for comment in enumerate(relevant_comments):
            f.write(str(comment)+"\n")
        f.close()
        print("Comments stored successfully!")

        polarity = []
        positive_comments = []
        negative_comments = []
        neutral_comments = []
        
        f = open(f"ytcomments{line_num}.txt", 'r', encoding='utf-8')
        comments = f.readlines()
        f.close()
        print("Analysing Comments...")
        for index, items in enumerate(comments):
            polarity = sentiment_scores(items, polarity)
        
            if polarity[-1] > 0.05:
                positive_comments.append(items)
            elif polarity[-1] < -0.05:
                negative_comments.append(items)
            else:
                neutral_comments.append(items)
 
        # Print polarity
        # print(polarity[:5])

        avg_polarity = round(sum(polarity)/len(polarity), 2)
        print("Average Polarity:", avg_polarity)
        if avg_polarity > 0.05:
            print("The Video has got a Positive response")
        elif avg_polarity < -0.05:
            print("The Video has got a Negative response")
        else:
            print("The Video has got a Neutral response")

        print("The comment with most positive sentiment:", comments[polarity.index(max(
            polarity))], "with score", max(polarity), "and length", len(comments[polarity.index(max(polarity))]))
        print("The comment with most negative sentiment:", comments[polarity.index(min(
            polarity))], "with score", min(polarity), "and length", len(comments[polarity.index(min(polarity))]))
        
        
        # creating csv file for the list of videos

        video_data = {
            "Video_ID": video_id,
            "Title": title,
            "Views": view_count,
            "Likes": likes,
            "Comments": commentsCount,
            "Sponsored": is_video_sponsored(title, description),  # Can be defined manually if a video is sponsored or not
            "Sentiment_Score": avg_polarity,
            "Category": videoCategory
        }

        # Convert to DataFrame
        df = pd.DataFrame([video_data])

        # Save to CSV
        if line_num == 1: # not os.path.isfile(csv_file):
            # If file does not exist, write the file and include the header
            df.to_csv(csv_file, index=False, mode='w', header=True)
        else:
            # If file exists, append to the file without writing the header
            df.to_csv(csv_file, index=False, mode='a', header=False)

        print("Data saved to youtube_data.csv")

        '''
        positive_count = len(positive_comments)
        negative_count = len(negative_comments)
        neutral_count = len(neutral_comments)

        # labels and data for Bar chart
        labels = ['Positive', 'Negative', 'Neutral']
        comment_counts = [positive_count, negative_count, neutral_count]

        # Creating bar chart
        plt.bar(labels, comment_counts, color=['blue', 'red', 'grey'])

        # Adding labels and title to the plot
        plt.xlabel('Sentiment')
        plt.ylabel('Comment Count')
        plt.title('Sentiment Analysis of Comments')

        # Displaying the chart
        plt.show()

        # labels and data for Bar chart
        labels = ['Positive', 'Negative', 'Neutral']
        comment_counts = [positive_count, negative_count, neutral_count]

        plt.figure(figsize=(10, 6)) # setting size

        # plotting pie chart
        plt.pie(comment_counts, labels=labels)

        # Displaying Pie Chart
        plt.show()
        '''




        



"""
# Taking input from the user and slicing for video id
video_id = input('Enter Youtube Video URL: ')[-11:]
print("video id: " + video_id)

# Getting the channelId of the video uploader
video_response = youtube.videos().list(
	part='snippet',
	id=video_id
).execute()

# Splitting the response for channelID
video_snippet = video_response['items'][0]['snippet']
uploader_channel_id = video_snippet['channelId']
print("channel id: " + uploader_channel_id)
"""
