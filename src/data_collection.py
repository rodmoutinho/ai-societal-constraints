import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import time

# Replace with your API key
api_key = "api"

# Configure the YouTube service
youtube = build("youtube", "v3", developerKey=api_key)

# Defina o diretório de saída
output_dir = r"C:\Users\isabe\Documents\Phd\Text Mining\data"
output_file = os.path.join(output_dir, "youtube_comments_with_likes.csv")

def search_videos(query, max_pages=20):
    """Searches for videos on YouTube and retrieves video details using pagination, including location if available."""
    videos = []
    next_page_token = None
    page_count = 0

    while page_count < max_pages:
        print(f"Fetching page {page_count + 1} for video search results...")
        search_request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=50,  # Max results per page
            pageToken=next_page_token
        )
        search_response = search_request.execute()
        
        for item in search_response.get('items', []):
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            channel_title = item['snippet']['channelTitle']
            published_date = item['snippet']['publishedAt']
            description = item['snippet']['description']
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Get video details, including statistics and location
            video_details = youtube.videos().list(
                part="statistics,recordingDetails",
                id=video_id
            ).execute()
            
            # Extract view count, like count and location details
            video_stats = video_details['items'][0]['statistics']
            view_count = int(video_stats.get('viewCount', 0))
            like_count = int(video_stats.get('likeCount', 0))  # Likes do vídeo

            if view_count > 0:
                location = video_details['items'][0].get('recordingDetails', {}).get('location', {})
                latitude = location.get('latitude', None)
                longitude = location.get('longitude', None)

                videos.append({
                    "video_id": video_id,
                    "title": title,
                    "channel_title": channel_title,
                    "description": description,
                    "view_count": view_count,
                    "like_count": like_count,  # Inclui os likes do vídeo
                    "published_date": published_date,
                    "url": url,
                    "latitude": latitude,
                    "longitude": longitude
                })

        print(f"Total videos collected so far: {len(videos)}")

        next_page_token = search_response.get("nextPageToken")
        if not next_page_token:
            print("No more pages available.")
            break

        page_count += 1
        time.sleep(1)  # Pause to manage API quota

    print(f"Finished fetching videos. Total videos retrieved: {len(videos)}")
    return videos

def get_video_comments(video_id, retries=3):
    """Collects comments from a specific video, including likes and dates of the comments."""
    comments = []
    print(f"Fetching comments for video ID: {video_id}")
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
        response = request.execute()

        while response:
            for item in response.get("items", []):
                top_level_comment = item["snippet"]["topLevelComment"]["snippet"]
                comment = top_level_comment["textDisplay"]
                comment_date = top_level_comment["publishedAt"]  # Retrieve comment date
                comment_likes = int(top_level_comment.get("likeCount", 0))  # Likes do comentário

                comments.append({
                    "comment": comment,
                    "comment_date": comment_date,  # Adiciona a data do comentário
                    "comment_likes": comment_likes  # Adiciona os likes do comentário
                })

            if "nextPageToken" in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response["nextPageToken"],
                    maxResults=100
                )
                response = request.execute()
            else:
                break

        print(f"Total comments collected for video {video_id}: {len(comments)}")
    
    except HttpError as e:
        error_message = str(e)
        if "commentsDisabled" in error_message:
            print(f"Comments are disabled for video: {video_id}")
        elif "processingFailure" in error_message:
            if retries > 0:
                print(f"Transient error occurred. Retrying... ({retries} retries left)")
                time.sleep(2)  # Wait before retrying
                return get_video_comments(video_id, retries - 1)
            else:
                print(f"Failed to fetch comments for video {video_id} after multiple retries.")
        else:
            print(f"An error occurred: {error_message}")
            raise

    return comments

def main():
    query = "artificial intelligence"
    videos = search_videos(query, max_pages=20)

    all_comments = []

    for idx, video in enumerate(videos, start=1):
        video_id = video["video_id"]
        title = video["title"]
        channel_title = video["channel_title"]
        description = video["description"]
        view_count = video["view_count"]
        like_count = video["like_count"]  # Obtém os likes do vídeo
        published_date = video["published_date"]
        url = video["url"]
        latitude = video["latitude"]
        longitude = video["longitude"]
        
        print(f"\nProcessing video {idx}/{len(videos)}: {title}")
        comments = get_video_comments(video_id)
        
        for comment_data in comments:
            all_comments.append({
                "video_id": video_id,
                "title": title,
                "channel_title": channel_title,
                "description": description,
                "view_count": view_count,
                "like_count": like_count,  # Inclui os likes do vídeo
                "video_published_date": published_date,
                "comment": comment_data["comment"],
                "comment_date": comment_data["comment_date"],  # Inclui a data do comentário
                "comment_likes": comment_data["comment_likes"],  # Inclui os likes do comentário
                "url": url,
                "latitude": latitude,
                "longitude": longitude
            })

    # Export comments and video information to the specified directory
    df = pd.DataFrame(all_comments)
    
    # Verifica se o diretório existe, caso contrário, cria
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nComments exported to {output_file}")

if __name__ == "__main__":
    main()
