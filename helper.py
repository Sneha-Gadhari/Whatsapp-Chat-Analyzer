from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from io import BytesIO
import tempfile
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import A4
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from reportlab.platypus import Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import os
extract = URLExtract()

def fetch_stats(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    #fetch nuber of messages
    num_messages = df.shape[0]

    #fetch number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    #fetch number of media shared
    num_media_msg = df[df['message'] == '<Media omitted>\n'].shape[0]

    #fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    return num_messages, len(words),num_media_msg, len(links)

def most_busy_users(df):
    # remove group notifications
    temp = df[df['user'] != 'group_notification']

    x = temp['user'].value_counts().head()
    percent_df = round(
        temp['user'].value_counts() / temp.shape[0] * 100, 2
    ).reset_index().rename(columns={'index': 'name', 'user': 'percent'})

    return x, percent_df

# average message length per user
def avg_message_length(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp['msg_len'] = temp['message'].apply(len)

    return round(temp['msg_len'].mean(), 2)


# most active hour
def most_active_hour(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['hour'].value_counts().sort_index()


def create_wordcloud(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains(r'media omitted|edited', case=False, na=False)]

    def remove_stopwords(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return ' '.join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stopwords)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains(r'media omitted|edited', case=False,na=False)]

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([e['emoji'] for e in emoji.emoji_list(message)])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='time_period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

# user personality tags
def user_personality(df):
    temp = df[df['user'] != 'group_notification']

    personality = {}

    # most talkative
    talkative = temp['user'].value_counts().idxmax()
    personality[talkative] = "📢 Most Talkative"

    # longest messages
    temp['msg_len'] = temp['message'].apply(len)
    long_msg_user = temp.groupby('user')['msg_len'].mean().idxmax()
    personality[long_msg_user] = personality.get(long_msg_user, "") + " 📝 Long Message Sender"

    # emoji lover
    emoji_count = {}
    for user in temp['user'].unique():
        msgs = temp[temp['user'] == user]['message']
        emoji_count[user] = sum(len(emoji.emoji_list(m)) for m in msgs)

    emoji_lover = max(emoji_count, key=emoji_count.get)
    personality[emoji_lover] = personality.get(emoji_lover, "") + " 😂 Emoji Lover"

    # night owl (messages between 12am–5am)
    night_df = temp[(temp['hour'] >= 0) & (temp['hour'] <= 5)]
    if not night_df.empty:
        night_owl = night_df['user'].value_counts().idxmax()
        personality[night_owl] = personality.get(night_owl, "") + " ⏱ Night Owl"

    return personality

# sentiment analysis
def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for msg in df['message']:
        polarity = TextBlob(msg).sentiment.polarity
        if polarity > 0:
            sentiments["Positive"] += 1
        elif polarity < 0:
            sentiments["Negative"] += 1
        else:
            sentiments["Neutral"] += 1

    return sentiments

# report generator

def save_plot(fig, name):
    path = f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

styles = getSampleStyleSheet()

def add_heading(text):
    return Paragraph(f"<b><font size=14>{text}</font></b>", styles["Normal"])

def add_text(text):
    return Paragraph(text, styles["Normal"])

def add_spacer(h=0.3):
    return Spacer(1, h * inch)

def add_image(path):
    img = Image(path)  
    img.hAlign = "CENTER"
    img.drawHeight = img.drawHeight
    img.drawWidth = img.drawWidth
    img.borderPadding = 6
    img.borderColor = colors.black
    img.borderWidth = 1
    return img

def auto_insights(label, value):
    return f"• <b>{label}</b>: {value}"

def generate_complete_pdf_report(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    buffer = BytesIO()
    styles = getSampleStyleSheet()
    story = []
    images = []

    doc = SimpleDocTemplate(buffer, pagesize=A4)

    from reportlab.lib.utils import ImageReader

    def add_plot(fig, title):
        story.append(Paragraph(f"<b>{title}</b>", styles['Heading2']))

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, bbox_inches="tight", dpi=200)
        plt.close(fig)

        img_reader = ImageReader(tmp.name)
        iw, ih = img_reader.getSize()  # REAL pixel size

        max_width = doc.width  # usable page width
        scale = min(1, max_width / iw)  # ONLY scale down if needed

        img = Image(
            tmp.name,
            width=iw * scale,
            height=ih * scale
        )

        images.append(tmp.name)
        story.append(img)
        story.append(Spacer(1, 18))

    # ---------- TITLE ----------
    story.append(Paragraph("<b>WhatsApp Chat Analysis – Full Report</b>", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"User: {selected_user}", styles['Normal']))

    start = df['only_date'].min()
    end = df['only_date'].max()

    story.append(
        Paragraph(
            f"Date Range: {start} to {end}",
            styles['Normal']
        )
    )
    story.append(Spacer(1, 12))

    # ---------- TOP STATS ----------
    num_msgs = df.shape[0]
    words = sum(df['message'].apply(lambda x: len(x.split())))
    media = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = sum(df['message'].apply(lambda x: len(extract.find_urls(x))))

    story.append(Paragraph("<b>Top Statistics</b>", styles['Heading2']))
    story.append(Paragraph(f"Total Messages: {num_msgs}", styles['Normal']))
    story.append(Paragraph(f"Total Words: {words}", styles['Normal']))
    story.append(Paragraph(f"Media Shared: {media}", styles['Normal']))
    story.append(Paragraph(f"Links Shared: {links}", styles['Normal']))

    # ---------- MONTHLY TIMELINE ----------
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    fig, ax = plt.subplots()
    ax.plot(timeline['time'], timeline['message'], color='blue')
    plt.xticks(rotation=90)
    add_plot(fig, "Monthly Timeline")

    # ---------- DAILY TIMELINE ----------
    daily = df.groupby('only_date').count()['message'].reset_index()
    fig, ax = plt.subplots()
    ax.plot(daily['only_date'], daily['message'], color='green')
    plt.xticks(rotation=90)
    add_plot(fig, "Daily Timeline")

    # ---------- WEEKLY ACTIVITY ----------
    week = df['day_name'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(week.index, week.values, color='pink')
    plt.xticks(rotation=45)
    add_plot(fig, "Weekly Activity")

    # ---------- MONTHLY ACTIVITY ----------
    month = df['month'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(month.index, month.values, color='purple')
    plt.xticks(rotation=45)
    add_plot(fig, "Monthly Activity")

    # ---------- HEATMAP ----------
    heatmap = df.pivot_table(index='day_name', columns='time_period',
                             values='message', aggfunc='count').fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap, cmap='YlGnBu', ax=ax)
    add_plot(fig, "Weekly Activity Heatmap")

    # ---------- BUSY USERS ----------
    if selected_user == "Overall":
        x, _ = most_busy_users(df)
        fig, ax = plt.subplots()
        ax.bar(x.index, x.values, color='red')
        plt.xticks(rotation=45)
        add_plot(fig, "Most Busy Users")

    # ---------- WORDCLOUD ----------
    wc = create_wordcloud("Overall", df)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    add_plot(fig, "Wordcloud")

    # ---------- COMMON WORDS ----------
    common = most_common_words("Overall", df)
    fig, ax = plt.subplots()
    ax.barh(common[0], common[1], color='orange')
    add_plot(fig, "Most Common Words")

    # ---------- EMOJI ANALYSIS ----------
    emoji_df = emoji_helper("Overall", df).head()
    fig, ax = plt.subplots()
    ax.pie(emoji_df[1], labels=emoji_df[0], autopct="%0.1f%%")
    add_plot(fig, "Emoji Analysis")

    # ---------- SENTIMENT ----------
    sentiment = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for msg in df['message']:
        p = TextBlob(msg).sentiment.polarity
        if p > 0:
            sentiment["Positive"] += 1
        elif p < 0:
            sentiment["Negative"] += 1
        else:
            sentiment["Neutral"] += 1

    fig, ax = plt.subplots()
    ax.bar(sentiment.keys(), sentiment.values(), color=['green', 'red', 'gray'])
    add_plot(fig, "Sentiment Analysis")

    # ---------- FINAL INSIGHTS ----------
    story.append(add_heading("Chat Insights"))
    story.append(add_spacer(0.2))

    # sentiment insight
    dominant_sentiment = max(sentiment, key=sentiment.get)
    story.append(
        add_text(f"• Overall chat sentiment is <b>{dominant_sentiment}</b>.")
    )
    story.append(add_spacer(0.1))

    # activity insight
    most_active_day = df['day_name'].value_counts().idxmax()
    story.append(
        add_text(f"• Most conversations happen on <b>{most_active_day}</b>.")
    )
    story.append(add_spacer(0.1))

    # night activity insight
    night_msgs = df[(df['hour'] >= 0) & (df['hour'] <= 5)].shape[0]
    night_percent = round((night_msgs / df.shape[0]) * 100, 2)

    if night_percent > 10:
        story.append(
            add_text(f"• <b>{night_percent}%</b> of messages are sent late at night.")
        )
        story.append(add_spacer(0.1))

    # user dominance insight
    top_user = df['user'].value_counts().idxmax()
    share = round((df['user'].value_counts().max() / df.shape[0]) * 100, 2)

    story.append(
        add_text(
            f"• <b>{top_user}</b> contributes <b>{share}%</b> of total messages."
        )
    )

    doc.build(story)

    for img in images:
        os.remove(img)

    buffer.seek(0)
    return buffer
