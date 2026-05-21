import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from transformers import pipeline


# =========================
# 1. 기본 설정
# =========================

def load_saved_api_key():
    # 환경변수 우선, 없으면 프로젝트 루트 .env 또는 API 키 파일에서 읽습니다.
    api_key = os.getenv("YOUTUBE_API_KEY")
    if api_key:
        return api_key.strip()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "YOUTUBE_API_KEY":
                    return value.strip().strip('"').strip("'")

    for filename in ["youtube_api_key.txt", "api_key.txt"]:
        key_path = os.path.join(base_dir, filename)
        if os.path.exists(key_path):
            with open(key_path, encoding="utf-8") as f:
                value = f.read().strip()
                if value:
                    return value

    return None

API_KEY = load_saved_api_key() or "example"

if API_KEY == "example":
    raise ValueError(
        "YOUTUBE_API_KEY 환경변수가 없습니다. "
        "터미널에서 export YOUTUBE_API_KEY='본인_YouTube_API_KEY' 를 실행하거나, "
        "프로젝트 루트에 .env 파일을 만들어 YOUTUBE_API_KEY=example 형태로 저장하세요."
    )

BASE_URL = "https://www.googleapis.com/youtube/v3"

START_DATE = "2025-01-01T00:00:00Z"

TOP_N_VIDEOS = 10
MAX_COMMENTS_PER_VIDEO = 300

# 공식 YouTube 채널 handle 또는 channel_id 입력
# 가능하면 실행 전 각 그룹의 공식 YouTube 채널 handle을 직접 확인하세요.
# handle 예시: @JYPEntertainment, @SMTOWN, @BLACKPINK 등
#
# 중요:
# 일부 그룹은 그룹 단독 공식 채널이 없거나 회사 채널에 영상이 올라올 수 있습니다.
# 이 경우 group별 channel_handle을 회사 공식 채널로 두고,
# 영상 제목에 그룹명 키워드를 포함하는 방식으로 필터링할 수 있습니다.

GROUP_CHANNELS = {
    "JYP": {
        "NMIXX": {
            "channel_handle": "@NMIXXOfficial",
            "title_keywords": ["NMIXX", "엔믹스"]
        },
        "Stray Kids": {
            "channel_handle": "@StrayKids",
            "title_keywords": ["Stray Kids", "SKZ", "스트레이 키즈", "스트레이키즈"]
        },
        "TWICE": {
            "channel_handle": "@TWICE",
            "title_keywords": ["TWICE", "트와이스"]
        },
    },
    "SM": {
        "NCT": {
            "channel_handle": "@NCTsmtown",
            "title_keywords": ["NCT", "엔시티"]
        },
        "aespa": {
            "channel_handle": "@aespa",
            "title_keywords": ["aespa", "에스파"]
        },
        "RIIZE": {
            "channel_handle": "@RIIZE_official",
            "title_keywords": ["RIIZE", "라이즈"]
        },
    },
    "YG": {
        "BABYMONSTER": {
            "channel_handle": "@BABYMONSTER",
            "title_keywords": ["BABYMONSTER", "베이비몬스터", "BAEMON"]
        },
        "TREASURE": {
            "channel_handle": "@TREASURE",
            "title_keywords": ["TREASURE", "트레저"]
        },
        "BLACKPINK": {
            "channel_handle": "@BLACKPINK",
            "title_keywords": ["BLACKPINK", "블랙핑크"]
        },
    },
    "HYBE": {
        "SEVENTEEN": {
            "channel_handle": "@SEVENTEEN",
            "title_keywords": ["SEVENTEEN", "세븐틴"]
        },
        "TXT": {
            "channel_handle": "@TXT_bighit",
            "title_keywords": ["TXT", "TOMORROW X TOGETHER", "투모로우바이투게더", "투바투"]
        },
        "ENHYPEN": {
            "channel_handle": "@ENHYPENOFFICIAL",
            "title_keywords": ["ENHYPEN", "엔하이픈"]
        },
    }
}


# =========================
# 2. YouTube API 함수
# =========================

def yt_get(endpoint, params, max_retries=3):
    params = dict(params)
    params["key"] = API_KEY

    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(max_retries):
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()

        if response.status_code in [403, 429, 500, 503]:
            wait_time = 5 * (attempt + 1)
            print(f"API 오류 {response.status_code}. {wait_time}초 후 재시도합니다.")
            time.sleep(wait_time)
            continue

        raise Exception(
            f"YouTube API 요청 실패: {response.status_code}\n"
            f"URL: {response.url}\n"
            f"응답: {response.text}"
        )

    raise Exception("YouTube API 재시도 횟수를 초과했습니다.")


def get_channel_id_from_handle(channel_handle):
    """
    YouTube handle로 channel_id를 가져옵니다.
    """
    data = yt_get(
        "channels",
        {
            "part": "id,snippet,contentDetails",
            "forHandle": channel_handle
        }
    )

    items = data.get("items", [])

    if not items:
        raise ValueError(f"채널을 찾지 못했습니다: {channel_handle}")

    item = items[0]
    channel_id = item["id"]
    title = item["snippet"]["title"]
    uploads_playlist_id = item["contentDetails"]["relatedPlaylists"]["uploads"]

    return channel_id, title, uploads_playlist_id


def get_upload_videos_from_playlist(uploads_playlist_id, max_pages=20):
    """
    업로드 playlist에서 영상 목록을 가져옵니다.
    playlistItems.list는 영상의 기본 정보만 가져오므로,
    이후 videos.list로 조회수/댓글 수 등을 추가 조회합니다.
    """
    all_items = []
    next_page_token = None

    for _ in range(max_pages):
        params = {
            "part": "snippet,contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": 50
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        data = yt_get("playlistItems", params)
        items = data.get("items", [])

        all_items.extend(items)

        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            break

    return all_items


def get_video_details(video_ids):
    """
    videos.list로 조회수, 좋아요 수, 댓글 수, 업로드일 등을 가져옵니다.
    한 번에 최대 50개씩 조회합니다.
    """
    details = []

    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]

        data = yt_get(
            "videos",
            {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(chunk),
                "maxResults": 50
            }
        )

        details.extend(data.get("items", []))

    return details


def get_comments_for_video(video_id, max_comments=300):
    """
    특정 영상의 최상위 댓글을 가져옵니다.
    댓글이 비활성화된 영상은 빈 리스트를 반환합니다.
    """
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "textFormat": "plainText",
            "order": "relevance"
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            data = yt_get("commentThreads", params)
        except Exception as e:
            print(f"댓글 수집 실패 video_id={video_id}: {e}")
            break

        for item in data.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "comment_id": item["id"],
                "comment_text": top_comment.get("textDisplay", ""),
                "comment_author": top_comment.get("authorDisplayName", ""),
                "comment_published_at": top_comment.get("publishedAt", ""),
                "comment_updated_at": top_comment.get("updatedAt", ""),
                "comment_like_count": top_comment.get("likeCount", 0)
            })

            if len(comments) >= max_comments:
                break

        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            break

        time.sleep(0.2)

    return comments


# =========================
# 3. 필터링 함수
# =========================

def contains_keyword(text, keywords):
    if not text:
        return False

    text_lower = text.lower()

    for keyword in keywords:
        if keyword.lower() in text_lower:
            return True

    return False


def parse_int(value):
    try:
        return int(value)
    except Exception:
        return 0


# =========================
# 4. 감성분석 모델
# =========================

print("감성분석 모델을 불러오는 중입니다...")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

LABEL_MAP = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def analyze_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral", 0.0

    text = text.replace("\n", " ").strip()
    text = text[:500]

    try:
        result = sentiment_model(text)[0]
        raw_label = result["label"]
        score = result["score"]
        label = LABEL_MAP.get(raw_label, raw_label.lower())
        return label, score
    except Exception:
        return "neutral", 0.0


# =========================
# 5. 평판도 계산 함수
# =========================

def calculate_reputation_score(pos_count, neg_count, alpha=1):
    score = 50 + 50 * ((pos_count - neg_count) / (pos_count + neg_count + alpha))
    score = max(0, min(100, score))
    return round(score, 2)


def safe_divide(a, b):
    return round(a / b, 4) if b else 0


# =========================
# 6. 전체 실행
# =========================

selected_video_rows = []
comment_rows = []

for company, groups in GROUP_CHANNELS.items():
    for group_name, config in groups.items():
        channel_handle = config["channel_handle"]
        title_keywords = config["title_keywords"]

        print(f"\n===== {company} / {group_name} / {channel_handle} 분석 시작 =====")

        try:
            channel_id, channel_title, uploads_playlist_id = get_channel_id_from_handle(channel_handle)

            playlist_items = get_upload_videos_from_playlist(uploads_playlist_id)

            video_ids = []
            basic_video_map = {}

            for item in playlist_items:
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})

                video_id = content_details.get("videoId")
                title = snippet.get("title", "")
                description = snippet.get("description", "")
                published_at = content_details.get("videoPublishedAt", snippet.get("publishedAt", ""))

                if not video_id:
                    continue

                if published_at < START_DATE:
                    continue

                # 그룹 단독 채널이면 사실상 필요 없지만,
                # 회사 채널이나 혼합 채널을 쓸 경우를 대비해 제목/설명 키워드 필터 적용
                if not (
                    contains_keyword(title, title_keywords)
                    or contains_keyword(description, title_keywords)
                ):
                    # 단독 채널인데 필터 때문에 너무 많이 빠지면 아래 continue를 주석 처리하세요.
                    continue

                video_ids.append(video_id)
                basic_video_map[video_id] = {
                    "title": title,
                    "description": description,
                    "published_at": published_at
                }

            if not video_ids:
                print(f"{group_name}: 2025년 이후 조건에 맞는 영상을 찾지 못했습니다.")
                continue

            details = get_video_details(video_ids)

            video_detail_rows = []

            for video in details:
                video_id = video["id"]
                snippet = video.get("snippet", {})
                stats = video.get("statistics", {})

                row = {
                    "company": company,
                    "group": group_name,
                    "channel_handle": channel_handle,
                    "channel_id": channel_id,
                    "channel_title": channel_title,
                    "video_id": video_id,
                    "video_title": snippet.get("title", ""),
                    "video_published_at": snippet.get("publishedAt", ""),
                    "view_count": parse_int(stats.get("viewCount", 0)),
                    "like_count": parse_int(stats.get("likeCount", 0)),
                    "comment_count": parse_int(stats.get("commentCount", 0)),
                }

                video_detail_rows.append(row)

            top_videos = sorted(
                video_detail_rows,
                key=lambda x: x["view_count"],
                reverse=True
            )[:TOP_N_VIDEOS]

            for rank, video in enumerate(top_videos, start=1):
                video["rank_by_views"] = rank
                selected_video_rows.append(video)

                print(
                    f"{group_name} Top {rank}: "
                    f"views={video['view_count']}, comments={video['comment_count']}, "
                    f"title={video['video_title'][:40]}"
                )

                comments = get_comments_for_video(
                    video["video_id"],
                    max_comments=MAX_COMMENTS_PER_VIDEO
                )

                for c in tqdm(comments, desc=f"{group_name} 댓글 감성분석"):
                    sentiment, confidence = analyze_sentiment(c["comment_text"])

                    comment_rows.append({
                        "company": company,
                        "group": group_name,
                        "channel_handle": channel_handle,
                        "channel_id": channel_id,
                        "channel_title": channel_title,
                        "video_id": video["video_id"],
                        "video_title": video["video_title"],
                        "video_published_at": video["video_published_at"],
                        "video_rank_by_views": rank,
                        "video_view_count": video["view_count"],
                        "video_like_count": video["like_count"],
                        "video_comment_count": video["comment_count"],
                        "comment_id": c["comment_id"],
                        "comment_text": c["comment_text"],
                        "comment_author": c["comment_author"],
                        "comment_published_at": c["comment_published_at"],
                        "comment_like_count": c["comment_like_count"],
                        "sentiment": sentiment,
                        "sentiment_confidence": confidence
                    })

                time.sleep(0.5)

        except Exception as e:
            print(f"{company} / {group_name} 분석 중 오류 발생: {e}")


# =========================
# 7. CSV 저장
# =========================

videos_df = pd.DataFrame(selected_video_rows)
comments_df = pd.DataFrame(comment_rows)

videos_df.to_csv("youtube_selected_top_videos.csv", index=False, encoding="utf-8-sig")
comments_df.to_csv("youtube_comment_sentiment_raw.csv", index=False, encoding="utf-8-sig")

print("\n원자료 저장 완료:")
print("- youtube_selected_top_videos.csv")
print("- youtube_comment_sentiment_raw.csv")


# =========================
# 8. 그룹별 평판도 요약
# =========================

summary_rows = []

if not comments_df.empty:
    for (company, group), sub in comments_df.groupby(["company", "group"]):
        total = len(sub)
        pos = len(sub[sub["sentiment"] == "positive"])
        neg = len(sub[sub["sentiment"] == "negative"])
        neu = len(sub[sub["sentiment"] == "neutral"])

        reputation_score = calculate_reputation_score(pos, neg)
        sentiment_clarity = safe_divide(pos + neg, total)

        summary_rows.append({
            "company": company,
            "group": group,
            "total_comments": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "positive_ratio": safe_divide(pos, total),
            "negative_ratio": safe_divide(neg, total),
            "neutral_ratio": safe_divide(neu, total),
            "sentiment_clarity": sentiment_clarity,
            "reputation_score": reputation_score
        })

group_summary_df = pd.DataFrame(summary_rows)

if not group_summary_df.empty:
    group_summary_df = group_summary_df.sort_values(
        by=["company", "reputation_score"],
        ascending=[True, False]
    )

group_summary_df.to_csv(
    "youtube_group_reputation_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 9. 기업별 평판도 요약
# =========================

if not group_summary_df.empty:
    company_equal_weight_df = (
        group_summary_df
        .groupby("company")["reputation_score"]
        .mean()
        .reset_index()
        .rename(columns={"reputation_score": "company_equal_weight_reputation_score"})
    )

    company_equal_weight_df["company_equal_weight_reputation_score"] = (
        company_equal_weight_df["company_equal_weight_reputation_score"].round(2)
    )

    company_equal_weight_df = company_equal_weight_df.sort_values(
        by="company_equal_weight_reputation_score",
        ascending=False
    )

    company_equal_weight_df.to_csv(
        "youtube_company_reputation_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n그룹별 평판도:")
    print(group_summary_df)

    print("\n기업별 평판도:")
    print(company_equal_weight_df)

else:
    print("\n댓글 데이터가 없어 요약 결과를 만들지 못했습니다.")