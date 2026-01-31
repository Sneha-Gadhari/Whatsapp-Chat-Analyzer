import streamlit as st
import preprocessor, helper
from helper import *
from preprocessor import get_date_range
import pandas as pd

st.sidebar.title("Whatsapp chat analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # date range filter
    min_date = df['only_date'].min()
    max_date = df['only_date'].max()

    start_date, end_date = st.sidebar.date_input("Select date range",[min_date, max_date],min_value=min_date,max_value=max_date)

    df = df[(df['only_date'] >= start_date) & (df['only_date'] <= end_date)]

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show analysis"):

        num_messages, words, num_media_msg, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total messages")
            st.title(num_messages)
        with col2:
            st.header("Total words")
            st.title(words)
        with col3:
            st.header("Media shared")
            st.title(num_media_msg)
        with col4:
            st.header("Links shared")
            st.title(num_links)

        # average message length
        avg_len = helper.avg_message_length(selected_user, df)
        st.subheader("Average Message Length")
        st.write(f"{avg_len} characters")

        #monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #activity map
        st.title("Activity map")
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='pink')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly activity map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap, cmap='YlGnBu')
        st.pyplot(fig)

        # most active hour
        st.title("Most Active Hours")
        active_hour = helper.most_active_hour(selected_user, df)

        fig, ax = plt.subplots()
        ax.bar(active_hour.index, active_hour.values)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Number of Messages")
        st.pyplot(fig)

        #finding the busiest users in the group
        if selected_user == 'Overall':
            st.title('Most busy users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        #wordcloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        #most common words
        most_common_df = helper.most_common_words(selected_user, df)

        fig, ax = plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1],color='orange')
        plt.xticks(rotation='vertical')

        st.title("Most common words")
        st.pyplot(fig)

        #emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)

        # sentiment analysis
        st.title("Sentiment Analysis")
        sentiment = helper.sentiment_analysis(selected_user, df)

        fig, ax = plt.subplots()
        ax.bar(sentiment.keys(), sentiment.values(), color=['green', 'red', 'gray'])
        st.pyplot(fig)

        # user personality
        st.title("User Personality Summary")
        personality = helper.user_personality(df)

        for user, tag in personality.items():
            st.write(f"**{user}** : {tag}")

        # ---------------- FINAL INSIGHTS ----------------
        st.markdown("## Chat Insights")

        # sentiment insight
        total_msgs = sum(sentiment.values())
        dominant_sentiment = max(sentiment, key=sentiment.get)

        st.markdown(f"• Overall chat sentiment is **{dominant_sentiment}**.")

        # activity insight
        busy_day = helper.week_activity_map(selected_user, df)
        most_active_day = busy_day.idxmax()

        st.markdown(f"• Most conversations happen on **{most_active_day}**.")

        # night activity insight
        night_msgs = df[(df['hour'] >= 0) & (df['hour'] <= 5)].shape[0]
        night_percent = round((night_msgs / df.shape[0]) * 100, 2)

        if night_percent > 10:
            st.markdown(f"• **{night_percent}%** of messages are sent late at night.")

        # user dominance insight (only overall)
        if selected_user == "Overall":
            top_user = df['user'].value_counts().idxmax()
            share = round(
                (df['user'].value_counts().max() / df.shape[0]) * 100, 2
            )
            st.markdown(f"• **{top_user}** contributes **{share}%** of total messages.")

        # FULL PDF REPORT
        def generate_pdf(df, stats, plots):
            file_name = "WhatsApp_Chat_Analysis_Report.pdf"
            doc = SimpleDocTemplate(
                file_name,
                pagesize=A4,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36
            )

            elements = []

            # 🔹 Title
            elements.append(add_heading("WhatsApp Chat Analysis Report"))
            elements.append(add_spacer(0.5))

            # 🔹 Date Range
            start, end = get_date_range(df)
            elements.append(add_text(f"<b>Date Range:</b> {start} to {end}"))
            elements.append(add_spacer())

            # 🔹 Key Insights
            elements.append(add_heading("Key Insights"))
            elements.append(add_spacer(0.2))

            insights = [
                auto_insights("Total Messages", stats["total_messages"]),
                auto_insights("Total Words", stats["total_words"]),
                auto_insights("Media Shared", stats["media"]),
                auto_insights("Links Shared", stats["links"]),
                auto_insights("Most Active User", stats["most_active"])
            ]

            for i in insights:
                elements.append(add_text(i))
                elements.append(add_spacer(0.15))

            elements.append(add_spacer(0.4))

            # 🔹 Graph Section
            elements.append(add_heading("Visual Analysis"))
            elements.append(add_spacer(0.3))

            for title, img_path in plots.items():
                elements.append(add_text(f"<b>{title}</b>"))
                elements.append(add_spacer(0.15))
                elements.append(add_image(img_path))
                elements.append(add_spacer(0.5))  # continuous flow

            doc.build(elements)
            return file_name


        # ---------------- PDF DOWNLOAD BUTTON ----------------
        st.title("Download Report")

        pdf_buffer = helper.generate_complete_pdf_report(selected_user, df)

        st.download_button(
            label="📄 Download Full PDF Report",
            data=pdf_buffer,
            file_name="WhatsApp_Chat_Analysis_Report.pdf",
            mime="application/pdf"
        )

